"""未来云任务名约束；不重命名历史任务，不执行或提交任何平台任务。"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'tools/open_annotation';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from tools.task_names import DEFAULT_NAMES,validate_task_name,suggested_task_name
import make_admission_spec as admission
import make_cloud_launcher as launcher


class FutureTaskNameTests(unittest.TestCase):
    def test_defaults_and_explicit_names_preserve_the_prefix(self):
        for workload in DEFAULT_NAMES:self.assertTrue(suggested_task_name(workload).startswith('zhounan_'))
        self.assertEqual(suggested_task_name('annotation','zhounan_标注对照-v2'),'zhounan_标注对照-v2')
        for bad in ('old-name','zhounan_',' zhounan_good','zhounan_good ', 'zhounan_bad\nvalue',None,False,31):
            with self.subTest(bad=bad),self.assertRaises(ValueError):validate_task_name(bad)

    def test_new_admission_name_flows_to_future_platform_plan_selection(self):
        a=SimpleNamespace(task_name='zhounan_annotation-custom',validation_uuids='GPU-a,GPU-b',not_before_utc='2026-10-27T00:00:00Z',
            mount_profile=SimpleNamespace(read_text=lambda:'{"mounts":[]}'),baseline_receipt=None)
        for key in ('validation_root','code_root','data_root','task_root','pilot_contract','silver_contract','full_index',
                    'full_filter_summary','full_filter_source','full_filter_audit','full_filter_audit_source',
                    'diagnostic_protection_reference','supervisor','independent_review','run_root','control_root',
                    'python','device_lock_root','native_comparison'):
            setattr(a,key,Path('/synthetic')/key)
        for key in ('vertical','main','recovery'):
            setattr(a,key+'_run',key);setattr(a,key+'_index',key+'/index.json');setattr(a,key+'_count',1)
        with patch.object(admission,'reference',side_effect=lambda p:{'path':str(p),'sha256':'a'*64}),\
             patch.object(admission,'validate_spec'),patch.object(admission,'verify',return_value={'passed':True}):
            spec,_=admission.build(a)
        self.assertEqual(spec['cloud']['platform_task_name'],a.task_name)
        self.assertEqual(launcher.platform_task_name(spec),a.task_name)
        with self.assertRaisesRegex(RuntimeError,'differs'):launcher.platform_task_name(spec,'zhounan_other')

    def test_bad_name_is_rejected_before_admission_inputs_are_read(self):
        with patch.object(admission,'reference')as read:
            with self.assertRaises(ValueError):admission.build(SimpleNamespace(task_name='bad'))
            read.assert_not_called()
        with self.assertRaises(ValueError):launcher.validate_spec({'cloud':{'platform_task_name':'bad'}})

    def test_old_metadata_without_name_receives_only_a_future_suggestion(self):
        original={'cloud':{}};before=json.dumps(original)
        self.assertTrue(launcher.platform_task_name(original).startswith('zhounan_'))
        self.assertEqual(json.dumps(original),before)

    def test_launcher_writes_new_plan_name_without_rewriting_pinned_spec(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);spec={'cloud':{'task_root':'/synthetic/task','run_root':'/synthetic/task/run','mounts':[]},
                'full_index':{'path':'/synthetic/index','sha256':'a'*64}}
            p=root/'spec.json';raw=json.dumps(spec).encode();p.write_bytes(raw);out=root/'new-plan'
            args=['make_cloud_launcher.py','--spec',str(p),'--spec-sha',launcher.sha(raw),'--output-dir',str(out),
                  '--task-name','zhounan_requested_plan']
            with patch.object(sys,'argv',args),patch.object(launcher,'render',return_value=('#!/bin/bash\n',{})),\
                 patch.object(launcher.Reader,'ref',return_value={'shards':[{}]}),contextlib.redirect_stdout(io.StringIO()):
                launcher.main()
            plan=json.loads((out/'platform-plan.json').read_bytes())
            self.assertEqual(plan['platform_task_name'],'zhounan_requested_plan');self.assertEqual(p.read_bytes(),raw)

    def test_help_from_independent_cwd_has_no_repository_environment_dependency(self):
        with tempfile.TemporaryDirectory()as d:
            env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
            for script in ('make_admission_spec.py','make_cloud_launcher.py'):
                r=subprocess.run([sys.executable,'-B',str(HERE/script),'--help'],cwd=d,env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr);self.assertIn('--task-name',r.stdout)
            # PTN构建器本来就是包入口；显式定位仓库后从独立cwd运行同一模块。
            code='import runpy,sys;sys.path.insert(0,sys.argv.pop(1));sys.argv=["ptn-build","--help"];runpy.run_module("tools.ptn_resume.build",run_name="__main__")'
            r=subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],cwd=d,env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);self.assertIn('--task-name',r.stdout)

    def test_package_imports_and_bad_CLI_names_work_without_repository_cwd(self):
        with tempfile.TemporaryDirectory()as d:
            env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
            code='import sys;sys.path.insert(0,sys.argv[1]);import tools.open_annotation.make_admission_spec,tools.open_annotation.make_cloud_launcher;assert "torch"not in sys.modules'
            r=subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],cwd=d,env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            output=Path(d)/'not-created'
            r=subprocess.run([sys.executable,'-B',str(HERE/'make_cloud_launcher.py'),'--spec',str(Path(d)/'missing.json'),
                '--spec-sha','a'*64,'--output-dir',str(output),'--task-name','old-prefix'],cwd=d,env=env,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('zhounan_',r.stderr);self.assertNotIn('FileNotFoundError',r.stderr)
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
