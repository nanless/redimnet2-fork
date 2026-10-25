"""归仓入口的无模型、无GPU合同及参数检查。"""
import argparse
import copy
import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import contracts as C
import freeze_contract as F
import cloud_gate as gate
import make_cloud_launcher as launcher
import verify_cloud_mounts as mounts
import validate_two_gpu as supervisor
from gpu_launch import validate_output_paths


def external_contract():
    # This is a format fixture, not downloaded or measured model identity.
    model={'path':'/offline/SYNTHETIC_NOT_MODEL','files':{'model.safetensors':{'bytes':1,
           'sha256':'e6fcf5be2c11b265f7217ba6102917a7fe64438506357be6999f6ad5a707cb5f'}}}
    return {'schema':'redimnet2-production-contract/v2','purpose':'pilot_validation','version':'synthetic-parent',
            'model_API_calls_allowed':0,'release_training_labels':False,'models':{'asr':model,'aligner':model},
            'source_files':{},'decoder':{},'engine_source_files':{},'quality_policy':{},
            'alignment':{'mode':'ultra_no_head','head_used':False,'repo_id':'17slever17/Qwen3-ForcedAligner-Ultra',
                         'revision':'765dc1d714e4c80196782a5df5bfbcb3e1416736'},
            'runtime':{'dtype':'float32','attention':'eager','TF32':False,'max_batch':1,'execution_policy':'single_window_independent',
                       'length_ratio_limit':1,'max_new_tokens':128,'window':{'core_seconds':26,'context_seconds':2}}}


class MaintenanceEntryTests(unittest.TestCase):
    def test_freeze_makes_new_paired_contracts_and_never_rewrites_parent(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);parent=root/'parent.json';parent.write_bytes(C.canonical_bytes(external_contract()));raw=parent.read_bytes()
            r=F.freeze(parent,C.sha_file(parent),root/'new-code','test-maintenance')
            a=C.read_contract(r['contracts']['pilot_validation']['path']);b=C.read_contract(r['contracts']['silver_annotation']['path'])
            self.assertEqual({k:v for k,v in a.items()if k not in {'purpose','version'}},{k:v for k,v in b.items()if k not in {'purpose','version'}})
            self.assertEqual(set(a['engine_source_files']),set(C.ENGINE_MODULES));self.assertEqual(parent.read_bytes(),raw)
            self.assertFalse(r['historical_acceptance_inherited']);self.assertFalse(r['models_executed'])
            with self.assertRaises(RuntimeError):F.freeze(parent,C.sha_file(parent),root/'new-code','test-maintenance')
            spec=importlib.util.spec_from_file_location('frozen_builder',root/'new-code/freeze_contract.py')
            frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
            second=frozen.freeze(parent,C.sha_file(parent),root/'second-code','test-second')
            self.assertEqual(Path(second['frozen_copy']),(root/'second-code').resolve())

    def test_freeze_rejects_wrong_parent_pin_and_path_escape(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'parent.json';p.write_bytes(C.canonical_bytes(external_contract()))
            with self.assertRaises(RuntimeError):F.freeze(p,'0'*64,Path(d)/'out','test')
            with self.assertRaises(RuntimeError):F.freeze(p,C.sha_file(p),HERE/'forbidden-output','test')
            with self.assertRaises(RuntimeError):F.freeze(p,C.sha_file(p),Path(d)/'out','../escape')

    def test_explicit_validation_UUIDs_unique_and_no_hardware_default(self):
        values=['GPU-00000000-0000-4000-8000-000000000001','GPU-00000000-0000-4000-8000-000000000002']
        self.assertEqual(gate.validation_targets({'validation_UUIDs':values}),set(values))
        for bad in [values[:1],[values[0]]*2,['not-a-UUID',values[1]]]:
            with self.assertRaises(RuntimeError):gate.validation_targets({'validation_UUIDs':bad})

    def test_mount_source_type_mode_and_exact_target(self):
        expected={'target':'/root','source':'synthetic-fileset','fstype':'gpfs','mode':'rw'}
        row=dict(expected,options='rw,relatime')
        self.assertEqual(mounts.validate_row(expected,{'filesystems':[row]}),row)
        for field,value in [('source','overlay'),('fstype','overlay'),('target','/'),('options','ro,relatime')]:
            bad=dict(row);bad[field]=value
            with self.assertRaises(RuntimeError):mounts.validate_row(expected,{'filesystems':[bad]})

    def test_generated_shell_carries_explicit_roots_and_preserves_rank(self):
        spec={'cloud':{'task_root':'/synthetic/task','control_root':'/synthetic/task/control','run_root':'/synthetic/task/formal',
                      'python':'/offline/python','device_lock_root':'/synthetic/global-gpu-locks'},'code_root':'/synthetic/task/code',
              'bundle':{'sha256':'a'*64},'silver_contract':{'path':'/synthetic/c.json','sha256':'b'*64},
              'full_index':{'path':'/synthetic/i.json','sha256':'c'*64}}
        with patch.object(launcher,'validate_spec'):
            script,_=launcher.render(spec,'d'*64)
        self.assertIn('--run-root /synthetic/task',script);self.assertIn('--device-lock-root /synthetic/global-gpu-locks',script)
        self.assertIn('auto8',script);self.assertNotIn('unset RANK',script);self.assertIn('noclobber',script)
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'script.sh';p.write_text(script);subprocess.run(['/bin/bash','-n',str(p)],check=True)

    def test_supervisor_command_uses_same_explicit_roots(self):
        with patch.multiple(supervisor,ROOT=Path('/synthetic/task/validation'),BASE=Path('/synthetic/task'),
                            CODE=Path('/synthetic/task/code'),BUNDLE='a'*64,LOCK_ROOT=Path('/synthetic/locks'),
                            UUIDS='GPU-00000000-0000-4000-8000-000000000001,GPU-00000000-0000-4000-8000-000000000002',create=True):
            command=supervisor.args('test',1,('c.json','b'*64),(Path('/synthetic/i.json'),'c'*64))
        self.assertEqual(command[command.index('--run-root')+1],'/synthetic/task')
        self.assertEqual(command[command.index('--device-lock-root')+1],'/synthetic/locks')
        self.assertIn('--full-verify',command)

    def test_supervisor_layout_passes_real_path_guard_but_code_overlap_does_not(self):
        with tempfile.TemporaryDirectory()as d:
            base=Path(d)/'task';code=base/'code';code.mkdir(parents=True)
            args=SimpleNamespace(run_root=str(base),device_lock_root=str(Path(d)/'global-locks'),
                output=str(base/'validation/resources-attempt1'),queue_root=str(base/'validation/queue'),output_root=str(base/'validation/labels'))
            self.assertEqual(validate_output_paths(args,here=code)[0],base.resolve())
            for key,bad in [('output',code/'bad'),('queue_root',code),('output_root',base),('output_root',base/'validation/queue/subdir')]:
                changed=copy.copy(args);setattr(changed,key,str(bad))
                with self.assertRaises(RuntimeError):validate_output_paths(changed,here=code)
            changed=copy.copy(args);changed.device_lock_root=str(base)
            with self.assertRaises(RuntimeError):validate_output_paths(changed,here=code)

    def test_cloud_spec_equal_run_stage_and_equal_shared_lock_refused(self):
        root='/synthetic/task';spec={'schema':'ultra-eight-three-train-spec/v3','code_root':root+'/code',
            'cloud':{'task_root':root,'run_root':root+'/formal','control_root':root+'/control','priority':6,
                     'platform_workers':1,'gpus_per_worker':8,'max_runtime_seconds':2592000,'python':'/offline/python',
                     'device_lock_root':root}}
        with self.assertRaisesRegex(RuntimeError,'shared locks'):launcher.validate_spec(spec)
        spec['cloud']['device_lock_root']='/synthetic/locks'
        for key in ('bundle','pilot_contract','silver_contract','full_index','full_filter_summary','full_filter_source',
                    'full_filter_audit','full_filter_audit_source','diagnostic_protection_reference','supervisor',
                    'sequence_start','sequence_complete','independent_review'):spec[key]={'path':'/synthetic/ref','sha256':'a'*64}
        spec['source_scope']=gate.SOURCE_SCOPE
        spec['vertical']={'run_root':root+'/formal','resources':root+'/formal/resources',
                          'parent_result':{'sha256':'b'*64},'index':{'sha256':'c'*64}}
        with self.assertRaisesRegex(RuntimeError,'formal run overlaps'):launcher.validate_spec(spec)


if __name__=='__main__':unittest.main()
