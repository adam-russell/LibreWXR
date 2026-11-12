# SPDX-License-Identifier: AGPL-3.0-or-later
"""Controller checks: no Docker or server access required."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import guards

spec=importlib.util.spec_from_file_location('controller',Path(__file__).with_name('run-benchmark.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class Checks(unittest.TestCase):
    def test_offline_limits_and_readonly_inputs(self):
        cmd=guards.command('sha256:test','owned-test',Path('/bundle'),Path('/result'),1536,'worker.py',[],frozen=Path('/weather'),native=Path('/native'))
        for key,value in [('--network','none'),('--cpus','2'),('--memory','1536m'),('--memory-swap','1536m'),('--pull','never')]:self.assertEqual(cmd[cmd.index(key)+1],value)
        self.assertIn('--read-only',cmd);self.assertIn('ALL',cmd)
        for suffix in ('dst=/trial,readonly','dst=/native,readonly','dst=/data/cache,readonly'):self.assertTrue(any(v.endswith(suffix) for v in cmd))
    def test_cannot_remove_unowned_container(self):
        with patch.object(guards,'inspect',return_value={'Config':{'Labels':{guards.LABEL:'someone-else'}}}),patch.object(guards,'run') as call:
            with self.assertRaises(AssertionError):guards.remove_owned('test')
            call.assert_not_called()
    def test_memory_wait_has_deadline(self):
        with patch.object(guards,'available',return_value=0):
            with self.assertRaises(RuntimeError):guards.wait_budget(1536,0)
    def test_balanced_order(self):
        for column in range(3):self.assertEqual({row[column] for row in c.ORDERS},set(c.ORDERS[0]))
    def fixture(self):
        m=dict(cases=[dict(name='wet')],rounds=2,native_sha256='native',runtime_hashes={'source':'hash'},snapshot_sha256='weather')
        samples=[dict(phase=p,repetition=i,cpu_seconds=1.,wall_seconds=1.1,coordinate_plan_misses=0) for p in ('presentation','render_warm','render_cold','pan_cold','reference_client') for i in range(2)]
        r=dict(status='completed',smoke=False,variant='data1',native_sha256='native',source_hashes=m['runtime_hashes'],snapshot_sha256='weather',distinct_frames=5,cgroup_after={},server_peak_rss_bytes=100,peak_rss_bytes=120,cases=[dict(case=m['cases'][0],geometry_hashes=['hash']*6,bytes_per_tile=[123]*6,samples=samples,plans=dict(bytes=100,max_bytes=200,entries=1,max_entries=32))])
        return r,m
    def test_complete_measurement(self):
        r,m=self.fixture();c.verify_measurement(r,m,'data1')
    def test_invalid_measurements_rejected(self):
        def change_row(r,key,value):r['cases'][0][key]=value
        for change in (lambda r:r.update(smoke=True),lambda r:r.update(native_sha256='other'),lambda r:r.update(distinct_frames=6),lambda r:change_row(r,'samples',[]),lambda r:r['cases'][0]['samples'][2].update(coordinate_plan_misses=1),lambda r:r['cases'][0]['samples'][0].update(cpu_seconds=0),lambda r:r['cgroup_after'].update({'memory.events':'oom_kill 1'}),lambda r:r['cases'][0]['plans'].update(bytes=201)):
            r,m=self.fixture();change(r)
            with self.assertRaises(AssertionError):c.verify_measurement(r,m,'data1')
if __name__=='__main__':unittest.main()
