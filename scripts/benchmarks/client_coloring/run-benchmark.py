# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded, serial offline jobs; never modifies or restarts services."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import time
import uuid
import guards

FROZEN=guards.SERVICES/'radar-controlled-comparison/20261106-153514/frozen'
NATIVE=guards.SERVICES/'librewxr-render-coordinates-8e4543df5a84/results-20261109-151218-5e110e/native'
ORDERS=(('png','data1','data6'),('data1','data6','png'),('data6','png','data1'))
SAFE_ENV={'LIBREWXR_ENABLED_REGIONS','LIBREWXR_RADAR_ENABLED','LIBREWXR_RRQPE_ENABLED','LIBREWXR_NA_SOURCE','LIBREWXR_CA_SOURCE','LIBREWXR_NOISE_FLOOR_DBZ','LIBREWXR_SMOOTH_RADIUS','LIBREWXR_WEBP_QUALITY','LIBREWXR_DPC_ENABLED','LIBREWXR_JMA_ENABLED','LIBREWXR_MMD_ENABLED','LIBREWXR_PAGASA_ENABLED'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def summarize(jobs):
    result={}
    for variant in ORDERS[0]:
        group=[j['measurement'] for j in jobs if j['variant']==variant]
        if not group:continue
        phases={}
        for phase in ('presentation','render_warm','render_cold','pan_cold','reference_client'):
            totals=[{key:sum(statistics.median(s[key] for s in row['samples'] if s['phase']==phase) for row in job['cases']) for key in ('cpu_seconds','wall_seconds')} for job in group]
            phases[phase]={k:statistics.median(t[k] for t in totals) for k in totals[0]}
        result[variant]=dict(repetitions=len(group),phases=phases,bytes_per_24_tiles=sum(sum(row['bytes_per_tile']) for row in group[0]['cases']),server_peak_rss_bytes=statistics.median(j['server_peak_rss_bytes'] for j in group),whole_worker_peak_rss_bytes=statistics.median(j['peak_rss_bytes'] for j in group),palette_switch=group[0]['palette_switch'])
    if 'png' in result:
        for variant in ('data1','data6'):
            if variant in result:
                r=result[variant];base=result['png']
                r['transfer_change_percent']=100*(r['bytes_per_24_tiles']/base['bytes_per_24_tiles']-1)
                r['server_cpu_change_percent']={p:100*(v['cpu_seconds']/base['phases'][p]['cpu_seconds']-1) for p,v in r['phases'].items() if p!='reference_client'}
    return result

def verify_measurement(r,m,variant):
    assert r['status']=='completed' and not r['smoke'] and r['variant']==variant
    assert r['native_sha256']==m['native_sha256'] and r['source_hashes']==m['runtime_hashes'] and r['snapshot_sha256']==m['snapshot_sha256']
    assert r['distinct_frames']==5
    events=dict(line.split() for line in r['cgroup_after'].get('memory.events','').splitlines())
    assert not any(int(events.get(k,0)) for k in ('max','oom','oom_kill')),'Memory pressure invalidated timing'
    if variant=='validate':
        v=r['validation']
        assert v['real_tiles']==len(m['cases'])*6 and v['synthetic_geometries']==5
        assert v['palettes']>0 and v['exact_rgba_comparisons']==(v['real_tiles']+5)*v['palettes']
        return
    assert [row['case'] for row in r['cases']]==m['cases']
    assert 0<r['server_peak_rss_bytes']<=r['peak_rss_bytes']
    phases={'presentation','render_warm','render_cold','pan_cold','reference_client'}
    for row in r['cases']:
        assert len(row['geometry_hashes'])==len(row['bytes_per_tile'])==6 and all(n>0 for n in row['bytes_per_tile'])
        assert len(row['samples'])==m['rounds']*len(phases)
        seen=set()
        for sample in row['samples']:
            key=(sample['phase'],sample['repetition'])
            assert key not in seen and key[0] in phases and 0<=key[1]<m['rounds'];seen.add(key)
            assert sample['cpu_seconds']>0 and sample['wall_seconds']>0
            if key[0]=='render_warm':assert sample['coordinate_plan_misses']==0
        plans=row['plans'];assert plans['bytes']<=plans['max_bytes'] and plans['entries']<=plans['max_entries']

def main():
    if os.geteuid()!=0:raise SystemExit('Run with sudo. No running service will be modified or restarted.')
    trial=Path(__file__).resolve().parent;m=json.loads((trial/'runtime/manifest.json').read_text())
    for name,want in json.loads((trial/'checksums.json').read_text()).items():assert sha(trial/name)==want,name
    assert sha(FROZEN/'state.json')==m['snapshot_sha256'],'Frozen snapshot changed'
    native=json.loads((NATIVE/'native.json').read_text());assert native['sha256']==m['native_sha256'] and native['source_sha256']==m['runtime_hashes']['source/librewxr/tiles/_sampling_native.c']
    assert sha(NATIVE/native['file'])==native['sha256']
    with (guards.SERVICES/'radar-render-benchmark.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not any(n.startswith('radar-') for n in guards.run(['docker','ps','--format','{{.Names}}']).splitlines()),'Another radar benchmark is active'
        before,configuration=guards.identities(),guards.file_hashes();guards.health()
        image=m['dependency_image'];guards.inspect(image) # Existing local dependency image; no pull/build.
        environment={k:v for item in guards.inspect(guards.PRODUCTION[1])['Config'].get('Env',[]) if '=' in item for k,v in [item.split('=',1)] if k in SAFE_ENV}
        result=trial/('results-'+time.strftime('%Y%m%d-%H%M%S',time.gmtime())+'-'+uuid.uuid4().hex[:6]);result.mkdir()
        (trial/'latest-results.txt').write_text(str(result)+'\n')
        report=dict(status='running',manifest=m,before=before,environment=environment,jobs=[])
        def save():
            p=result/'benchmark.json.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(result/'benchmark.json')
        def interrupted(signum,frame):raise KeyboardInterrupt('Benchmark interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
        deadline=time.monotonic()+m['deadline_seconds'];save()
        print('Results:',result,flush=True)
        print('One validation and nine balanced fresh-worker jobs; usually 8–15 minutes, maximum 20 minutes. Two CPUs, 1.5 GiB RAM, no swap/network, read-only weather. No production changes.',flush=True)
        def launch(variant,workspace,stem,max_seconds=180):
            guards.wait_budget(1536,deadline);name='radar-client-color-'+uuid.uuid4().hex[:10]
            cmd=guards.command(image,name,trial,workspace,1536,'worker.py',['--variant',variant,'--output','/results/measurement.json'],frozen=FROZEN,native=NATIVE,environment=environment)
            outcome=guards.supervise(cmd,name,result/(stem+'.log'),deadline,max_seconds=max_seconds)
            assert outcome['exit_code']==0 and not outcome['oom_killed'],'Step failed; inspect '+stem+'.log'
            assert guards.identities()==before and guards.file_hashes()==configuration,'Running setup changed'
            r=json.loads((workspace/'measurement.json').read_text())
            verify_measurement(r,m,variant)
            return outcome,r
        try:
            print('Validating all palettes and exporting the offline comparison client.',flush=True)
            validation=result/'validation';validation.mkdir();controller,r=launch('validate',validation,'validation',300)
            report['validation']=dict(controller=controller,measurement=r);save()
            signatures=None
            for cycle,order in enumerate(ORDERS,1):
                for variant in order:
                    stem=f'{cycle}-{variant}';workspace=result/stem;workspace.mkdir();print('Measuring '+stem,flush=True)
                    controller,r=launch(variant,workspace,stem)
                    current=[(row['case'],row['geometry_hashes']) for row in r['cases']]
                    if signatures is None:signatures=current
                    assert current==signatures,'RGBA changed between variants'
                    report['jobs'].append(dict(cycle=cycle,variant=variant,controller=controller,measurement=r));report['summary']=summarize(report['jobs']);save()
            report.update(status='completed',outputs_identical=True)
        except BaseException as error:
            report.update(status='failed',error=str(error) or type(error).__name__);raise
        finally:
            try:
                report['after']=guards.identities();report['production_unchanged']=report['after']==before and guards.file_hashes()==configuration
                if not report['production_unchanged']:report['status']='failed'
            finally:save()
        assert report['production_unchanged']
        print('Finished. Exact reference images matched; services and settings are unchanged.',flush=True)
        print(json.dumps(report['summary'],indent=2),flush=True)
        print('Offline browser comparison:',validation/'comparison.html',flush=True)
if __name__=='__main__':main()
