"""启动成本工具的窄守卫反例；不运行实际selected或生产锁。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import measure_native_startup_cost as C


class CostGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.code=self.root/'formal-code';self.code.mkdir();members={}
        for name in ('native_batch_admission.py','bounded_speed_runtime.py','gpu_launch.py'):
            q=self.code/name;q.write_text("raise RuntimeError('default must not import actual modules')\n");members[name]=C.sha_file(q)
        self.bundle=self.put(self.code/'bundle_files.json',members);self.contract=self.put(self.root/'formal-contract.json',dict(schema='redimnet2-production-contract/v2',purpose='silver_annotation',native_formal={'default_enabled':False}))
        selected=dict(schema='annotation-native-selected-pilot-admission/v2',purpose='production',formal_execution_bundle=self.bundle)
        for role,n in (('baseline',3),('main',3),('recovery',2)):
            folder=self.root/role;folder.mkdir();shard=folder/'finite.jsonl';shard.write_text('{}\n'*n)
            index=self.put(folder/'index.json',dict(source_count=n,shards=[dict(path='finite.jsonl',sha256=C.sha_file(shard),count=n)]))
            selected[role+'_track']=self.put(folder/'track.json',dict(schema='redimnet2-annotation-native-batch-track/v1',index=index,
                queue_root=str(folder/'producer/queue'),output_root=str(folder/'producer/labels'),resources_root=str(folder/'producer/resources')))
        self.selected=self.put(self.root/'selected.json',selected);self.out=self.root/'cpu-startup-cost-test'
        self.plan=dict(schema=C.SCHEMA,selected=self.selected,formal_contract=self.contract,formal_bundle=self.bundle,code_root=str(self.code),output_root=str(self.out),max_seconds=900);self.save()
    def put(self,p,value):
        p.write_text(json.dumps(value));return dict(path=str(p),sha256=C.sha_file(p))
    def save(self):self.pin=self.put(self.root/'plan.json',self.plan)
    def test_default_checks_finite_inputs_without_import_spawn_or_output(self):
        with patch.object(C.importlib,'import_module',side_effect=AssertionError('default import forbidden')),patch.object(C.subprocess,'Popen',side_effect=AssertionError('default spawn forbidden')):
            value=C.run(self.pin)
        self.assertFalse(value['executed']);self.assertEqual(value['source_counts'],dict(baseline=3,main=3,recovery=2));self.assertFalse(self.out.exists())
        self.assertEqual(len(C.check(self.pin)['allowed_jsonl']),3)
    def test_bad_budget_and_fixture_purpose_are_refused(self):
        for value in (True,0,901,float('inf')):
            self.plan['max_seconds']=value;self.save()
            with self.subTest(value=value),self.assertRaises(ValueError):C.check(self.pin)
        self.plan['max_seconds']=900;s=json.loads(Path(self.selected['path']).read_text());s['purpose']='cpu_fixture';self.plan['selected']=self.put(Path(self.selected['path']),s);self.save()
        with self.assertRaisesRegex(ValueError,'production'):C.check(self.pin)
    def test_changed_code_and_model_member_are_rejected_before_model_read(self):
        q=self.code/'native_batch_admission.py';q.write_text('# changed')
        with self.assertRaisesRegex(ValueError,'source changed'):C.check(self.pin)
        members=json.loads(Path(self.bundle['path']).read_text());members[q.name]=C.sha_file(q);media=self.code/'model.safetensors';media.write_bytes(b'not-model-data');members[media.name]=C.sha_file(media)
        self.plan['formal_bundle']=self.put(Path(self.bundle['path']),members);s=json.loads(Path(self.selected['path']).read_text());s['formal_execution_bundle']=self.plan['formal_bundle'];self.plan['selected']=self.put(Path(self.selected['path']),s);self.save();original=C.sha_file
        def guarded(path):
            if Path(path)==media:raise AssertionError('model member must reject before hash')
            return original(path)
        with patch.object(C,'sha_file',side_effect=guarded),self.assertRaisesRegex(ValueError,'model/database bundle'):C.check(self.pin)
    def test_full_library_count_and_producer_output_overlap_are_refused(self):
        tpin=json.loads(Path(self.selected['path']).read_text())['main_track'];track=json.loads(Path(tpin['path']).read_text());idx=json.loads(Path(track['index']['path']).read_text());idx['source_count']=4097;idx['shards'][0]['count']=4097;track['index']=self.put(Path(track['index']['path']),idx)
        s=json.loads(Path(self.selected['path']).read_text());s['main_track']=self.put(Path(tpin['path']),track);self.plan['selected']=self.put(Path(self.selected['path']),s);self.save()
        with self.assertRaisesRegex(ValueError,'4096'):C.check(self.pin)
        idx['source_count']=3;idx['shards'][0]['count']=3;track['index']=self.put(Path(track['index']['path']),idx);track['queue_root']=str(self.out);s['main_track']=self.put(Path(tpin['path']),track);self.plan['selected']=self.put(Path(self.selected['path']),s);self.save()
        with self.assertRaisesRegex(ValueError,'producer tree'):C.check(self.pin)
    def test_audit_guard_allows_sqlite_import_and_only_finite_JSONL(self):
        cfg=C.check(self.pin);allowed=next(iter(cfg['allowed_jsonl']));outside=self.root/'other.jsonl';outside.write_text('{}\n');self.out.mkdir()
        program="""import runpy,sys,json,os,fcntl
from pathlib import Path
n=runpy.run_path(sys.argv[1]);out=Path(sys.argv[2]);code=Path(sys.argv[3]);allowed=sys.argv[4];outside=sys.argv[5]
n['_guard'](out,code,{allowed:'synthetic'})
import sqlite3
assert open(allowed).read()
actions=[lambda:sqlite3.connect(':memory:'),lambda:open(outside).read(),lambda:open(str(out/'bad.wav')).read(),lambda:open(str(out/'x.annotation.json')).read(),lambda:open(str(out/'x.annotation.json.commit.json')).read(),lambda:open(str(out/'claim.json'),'w'),lambda:os.mkdir(out/'claim-dir')]
rejected=0
for f in actions:
 try:f()
 except PermissionError:rejected+=1
print(json.dumps({'rejected':rejected,'sqlite_imported':True}))
"""
        r=subprocess.run([sys.executable,'-I','-S','-B','-c',program,str(Path(C.__file__).resolve()),str(self.out),str(self.code),allowed,str(outside)],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['rejected'],7)
    def test_child_checks_exact_controller_bytes_before_compiling(self):
        script=self.root/'changed-controller.py';script.write_text("raise RuntimeError('must not compile mismatched controller')\n")
        r=subprocess.run([sys.executable,'-I','-S','-B','-c',C.CHILD_CODE,str(script),'unused','0'*64,'f'*64],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(r.returncode,0);self.assertIn('cost controller bytes changed',r.stderr);self.assertNotIn('must not compile',r.stderr)
    def test_owned_proc_identity_handles_parentheses_and_rejects_other_PID(self):
        fields=['S']+['0']*19;fields[1]='42';fields[19]='123456'
        raw='123 (cost child (odd) name) '+' '.join(fields)
        self.assertEqual(C._parse_proc_stat(raw,123),dict(pid=123,parent_pid=42,start_ticks=123456))
        with self.assertRaisesRegex(ValueError,'identity'):C._parse_proc_stat(raw,124)
        fields[19]='0'
        with self.assertRaisesRegex(ValueError,'start identity'):C._parse_proc_stat('123 (cost) '+' '.join(fields),123)


if __name__=='__main__':unittest.main()
