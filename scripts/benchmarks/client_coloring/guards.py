# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded offline warm-only jobs; no compiler setup or service mutations."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

SERVICES=Path('/home/adamrussell/services')
PROJECT=SERVICES/'LibreWXR'
PRODUCTION=['librewxr-renderer-1','librewxr-pipeline-1']
LABEL='com.marineweatherpro.client-coloring-benchmark'
GIB=1024**3
RESERVE=1024**3


def run(args,**kwargs):
    return subprocess.run(args,check=True,capture_output=True,text=True,timeout=30,**kwargs).stdout


def inspect(name):return json.loads(run(['docker','inspect',name]))[0]


def identities():
    return {name:{k:r[k] for k in ('Id','Image','RestartCount')}|
            dict(started=r['State']['StartedAt'],oom_killed=r['State']['OOMKilled'])
            for name in PRODUCTION for r in [inspect(name)]}


def file_hashes():
    return {name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()
            for name in ('.env','docker-compose.yml','docker-compose.override.yml')}


def health():
    with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=4) as response:data=json.load(response)
    assert data.get('status')=='ok' and data.get('cluster',{}).get('workers_reporting')==2,'Service health changed'


def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024


def wait_budget(mib,deadline):
    until=min(deadline,time.monotonic()+60);required=mib*1024**2+RESERVE;next_progress=0
    while available()<required:
        if time.monotonic()>until:raise RuntimeError(f'RAM budget unavailable: need {required/GIB:.2f} GiB; have {available()/GIB:.2f} GiB. No services were changed.')
        if time.monotonic()>next_progress:
            print(f'Waiting for RAM: {required/GIB:.2f} GiB needed, {available()/GIB:.2f} GiB available (up to 60 seconds).',flush=True);next_progress=time.monotonic()+15
        health();time.sleep(2)


def command(image,name,trial,output,mib,script,args,frozen=None,native=None,environment=None):
    cmd=['docker','run','--pull','never','--name',name,'--label',LABEL+'='+name,
         '--cpus','2','--memory',str(mib)+'m','--memory-swap',str(mib)+'m',
         '--pids-limit','128','--security-opt','no-new-privileges','--workdir','/tmp',
         '--mount','type=bind,src='+str(trial/'runtime')+',dst=/trial,readonly',
         '--mount','type=bind,src='+str(output)+',dst=/results']
    cmd+=['--network','none','--read-only','--cap-drop','ALL','--tmpfs','/tmp:rw,nosuid,nodev,size=128m']
    if frozen is not None:cmd+=['--mount','type=bind,src='+str(frozen)+',dst=/data/cache,readonly']
    if native is not None:cmd+=['--mount','type=bind,src='+str(native)+',dst=/native,readonly']
    env=dict(environment or {})
    env.update(PYTHONPATH='/trial/source:/trial',PYTHONDONTWRITEBYTECODE='1',
               LIBREWXR_LOG_FILE='',LIBREWXR_CACHE_DIR='/tmp/benchmark-cache',LIBREWXR_COLOR_SCHEMES_FILE='',
               LIBREWXR_VOLATILE_CACHE_DIR='/tmp/volatile',LIBREWXR_COORD_STORE_ENABLED='false',LIBREWXR_COORD_CACHE_SIZE='32',
               LIBREWXR_NOWCAST_ENABLED='false',LIBREWXR_REGIONAL_NWP_ENABLED='false',LIBREWXR_SATELLITE_ENABLED='false',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TRIAL_NATIVE_DIR='/native')
    for key,value in sorted(env.items()):cmd+=['-e',key+'='+str(value)]
    return cmd+['--entrypoint','python',image,'/trial/'+script]+args


def remove_owned(name):
    try:r=inspect(name)
    except subprocess.CalledProcessError:return
    assert r['Config']['Labels'].get(LABEL)==name,'Unowned container; refusing removal'
    run(['docker','rm','-f',name])


def supervise(cmd,name,log,deadline,max_seconds=180):
    start=time.monotonic();minimum=available();next_health=0;failures=0;next_progress=start+15
    with log.open('w') as stream:
        proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                spare=available();minimum=min(minimum,spare)
                if spare<RESERVE:raise RuntimeError('Benchmark stopped at host memory reserve')
                now=time.monotonic()
                if now>deadline or now-start>max_seconds:raise RuntimeError('Benchmark step deadline exceeded')
                if now>=next_health:
                    try:health();failures=0
                    except Exception:failures+=1
                    if failures>=3:raise RuntimeError('Benchmark stopped after service health failures')
                    next_health=now+5
                if now>=next_progress:
                    print(f'  Still running: {int(now-start)}s; {spare/GIB:.2f} GiB host RAM available.',flush=True);next_progress=now+15
                time.sleep(.2)
            state=inspect(name)['State']
            result=dict(exit_code=proc.returncode,oom_killed=state['OOMKilled'],minimum_host_available_bytes=minimum)
            if proc.returncode:print(log.read_text()[-3500:],flush=True)
            return result
        finally:
            remove_owned(name)
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
