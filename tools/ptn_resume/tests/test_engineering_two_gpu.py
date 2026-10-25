"""Portable runner contract tests; toy states never count as real B6/W2 success."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from tools.ptn_resume import engineering_two_gpu as E


class EngineeringContractTests(unittest.TestCase):
    def test_import_is_stdlib_only_and_never_initializes_torch(self):
        run=subprocess.run([sys.executable,'-B','-c',
            'import sys;from tools.ptn_resume import engineering_two_gpu;assert "torch" not in sys.modules'],
            capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        self.assertEqual(run.returncode,0,run.stderr)

    def test_cleanup_and_entire_resource_return_keep_both_signal_handlers_ignored(self):
        # 只注入处理器字典和假子进程；不发送信号，不读/proc或GPU。
        import signal
        from word_ft_v3 import resources as R
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();package=root/'package';source=root/'source';run=root/'run'
            for p in (package,source,run):p.mkdir()
            settings=SimpleNamespace(PACKAGE=package,SOURCE=source,RUN=run,PYTHON=sys.executable,DECODER=root/'decoder')
            args=SimpleNamespace(output=root/'new',expected_uuids=['one','two'],max_seconds=5,
                max_attempts_per_update=1,lock_root=root/'locks',manifest_sha='0'*64,
                contract_sha='1'*64,terminal_evidence=None,case='finite')
            prior={signal.SIGTERM:object(),signal.SIGINT:object()};handlers=dict(prior);events=[]
            def install(sig,handler):
                old=handlers[sig];handlers[sig]=handler;return old
            def ignored(phase):
                self.assertEqual(handlers,{s:signal.SIG_IGN for s in prior});events.append(phase)
            class Locks:
                evidence={}
                def __init__(self,*_):pass
                def __enter__(self):events.append('locked');return self
                def __exit__(self,*_):
                    self_test.assertEqual(handlers,prior);events.append('released')
            self_test=self;probe={'devices':[{'uuid':'one'},{'uuid':'two'}]};calls=[0]
            def double_probe(*_):
                calls[0]+=1
                if calls[0]==1:return probe
                ignored('return-probe')
                if calls[0]==2:raise OSError('synthetic unknown GPU state')
                return probe
            real_write=E.write_new
            def write(path,value):
                if path.name=='parent-result.json':ignored('result-publish')
                return real_write(path,value)
            with patch.object(E,'imports',return_value=settings),patch.object(E,'sys',SimpleNamespace(
                    executable=sys.executable,modules={},path=list(sys.path))), \
                 patch.object(R,'DeviceLocks',Locks),patch.object(R,'double_probe',side_effect=double_probe), \
                 patch.object(R,'protected_path',return_value=None),patch.object(R,'create_ticket',return_value={}), \
                 patch.object(R,'stable_identity',side_effect=lambda d:[v['uuid']for v in d]), \
                 patch.object(R,'process_start_time',side_effect=OSError('synthetic missing start')), \
                 patch.object(E.signal,'signal',side_effect=install), \
                 patch.object(E.subprocess,'Popen',return_value=SimpleNamespace(pid=12345)), \
                 patch.object(E,'cleanup_all',side_effect=lambda *_:ignored('cleanup')), \
                 patch.object(E.time,'sleep',side_effect=lambda _:ignored('return-wait')), \
                 patch.object(E,'write_new',side_effect=write):
                with self.assertRaisesRegex(ValueError,'engineering case failed'):E.parent(args)
            self.assertEqual(handlers,prior)
            self.assertEqual(events,['locked','cleanup','return-probe','return-wait','return-probe','result-publish','released'])
            result=json.loads((root/'new'/'parent-result.json').read_text())
            self.assertFalse(result['passed']);self.assertTrue(result['resource_returned'])

    def test_actual_relative_deployment_imports_resource_helpers_outside_repository_cwd(self):
        # 真实部署保留tools/ptn_resume与仓库根word_ft_v3，不能依赖cwd或PYTHONPATH。
        code = ('import runpy,sys;from pathlib import Path;'
                'm=runpy.run_path(sys.argv[1],run_name="engineering_import_only");'
                'root=m["REPO"];sys.path.insert(0,str(root));'
                'from word_ft_v3 import resources;'
                'assert Path(resources.__file__).resolve()==root/"word_ft_v3/resources.py";'
                'assert all((root/p).is_file() for p in '
                '("word_ft_v3/resources.py","word_ft_v3/topology.py","word_ft_v3/__init__.py"));'
                'assert "torch" not in sys.modules')
        with tempfile.TemporaryDirectory() as tmp:
            env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
            run=subprocess.run([sys.executable,'-B','-c',code,str(Path(E.__file__).resolve())],
                               cwd=tmp,capture_output=True,text=True,env=env)
        self.assertEqual(run.returncode,0,run.stderr)

    def test_new_output_and_protected_overlaps_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();package=root/'package';package.mkdir();source=package/'source';source.mkdir();run=root/'run'
            for value in (package,source,source/'out',root):
                with self.subTest(path=value),self.assertRaises(ValueError):E.output_guard(value,package,source,run,lambda x:None)
            out=root/'new';self.assertEqual(E.output_guard(out,package,source,run,lambda x:None),out)
            out.mkdir()
            with self.assertRaises(ValueError):E.output_guard(out,package,source,run,lambda x:None)
            link=root/'link';link.symlink_to(root/'missing',target_is_directory=True)
            with self.assertRaises(ValueError):E.output_guard(link,package,source,run,lambda x:None)

    def test_json_evidence_never_overwrites_existing_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'evidence.json';E.write_new(file,{'synthetic':True});before=file.read_bytes()
            with self.assertRaises(FileExistsError):E.write_new(file,{'synthetic':False})
            self.assertEqual(file.read_bytes(),before)

    def test_ready_ACK_final_name_is_published_only_after_complete_temporary_JSON(self):
        with tempfile.TemporaryDirectory() as tmp:
            final=Path(tmp)/'ready.json';value={'rank':0,'synthetic_engineering_only':True};link=E.os.link;seen=[]
            def inspect_then_link(source,target):
                self.assertFalse(Path(target).exists())
                self.assertEqual(json.loads(Path(source).read_text()),value)
                self.assertTrue(Path(source).read_bytes().endswith(b'\n'));seen.append(True)
                return link(source,target)
            with patch.object(E.os,'link',side_effect=inspect_then_link):E.write_new(final,value)
            self.assertEqual(seen,[True]);self.assertEqual(json.loads(final.read_text()),value)
            self.assertFalse(final.with_name('ready.json.part').exists())

    def test_bounded_attempts_count_real_updates_not_scaler_attempts(self):
        for success in (False,True):
            with self.subTest(success=success),tempfile.TemporaryDirectory() as tmp:
                rows=[{'actual_optimizer_step':False},{'actual_optimizer_step':success}]
                with patch.object(E,'run_step',side_effect=rows) as step:
                    kwargs=dict(ddp=None,model=None,optimizer=None,scaler=None,criterion=None,wave=None,targets=None,
                        scheduler=None,margin=None,iteration=705047,N=None,rank=0,output=Path(tmp),phase='toy',maximum=2)
                    if success:
                        actual,clock=E.run_until_update(**kwargs);self.assertEqual(clock,705049)
                        self.assertEqual(sum(r['actual_optimizer_step']for r in actual),1)
                    else:
                        with self.assertRaises(ValueError):E.run_until_update(**kwargs)
                    self.assertEqual(step.call_count,2);self.assertEqual(len(list(Path(tmp).glob('attempt-*.json'))),2)

    def test_cleanup_unknown_nonce_and_repeated_presence_error_do_not_escape_locks(self):
        with tempfile.TemporaryDirectory() as tmp:
            events=[];seen=iter([PermissionError('query denied'),None,True,False,False])
            def presence(pid):
                value=next(seen);events.append(value)
                if isinstance(value,Exception):raise value
                return value
            stop=unittest.mock.Mock(side_effect=PermissionError('ownership denied'))
            E.cleanup_child(SimpleNamespace(pid=333,poll=lambda:None),10,'nonce',Path(tmp),stop=stop,
                presence=presence,sleep=lambda _:None)
            self.assertEqual(len(events),5);stop.assert_called_once()
            record=json.loads((Path(tmp)/'cleanup-unproven-333.json').read_text())
            self.assertTrue(record['locks_retained']);self.assertFalse(record['unknown_processes_signalled'])

    def test_cleanup_missing_initial_start_requires_recovery_or_wait_without_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            stop=unittest.mock.Mock();states=iter([True,False,False])
            recover=unittest.mock.Mock(side_effect=PermissionError('start unreadable'))
            E.cleanup_child(SimpleNamespace(pid=333,poll=lambda:None),None,'nonce',Path(tmp),stop=stop,
                recover=recover,presence=lambda _:next(states),sleep=lambda _:None)
            recover.assert_called_once();stop.assert_not_called()

    def test_independent_rank_sessions_cleaned_even_after_launcher_exited(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher=SimpleNamespace(pid=100,poll=lambda:0);seen=[]
            ranks={200:{'pgid':200,'leader_start':22},300:{'pgid':300,'leader_start':23}}
            (Path(tmp)/'cuda-ack.json').write_text('{}')
            with patch.object(E,'discover_owned_groups',side_effect=[ranks,{}]),\
                 patch.object(E,'cleanup_child',side_effect=lambda actor,*_:seen.append(actor.pid)):
                E.cleanup_all(launcher,20,'nonce',Path(tmp),ranks)
            self.assertEqual(seen,[200,300,100])

    def test_unrelated_new_process_unreadable_environ_never_read_or_block_cleanup(self):
        rows={100:{'pgid':100,'ppid':1,'start':20},101:{'pgid':101,'ppid':100,'start':21},
              900:{'pgid':900,'ppid':1,'start':22}}
        def proof(pid,nonce):
            if pid==900:raise PermissionError('unrelated nondumpable user environ')
            return {'pid':pid,'pgid':pid,'leader_start':rows[pid]['start']}
        with patch.object(E,'process_table',return_value=rows),patch.object(E,'process_proof',side_effect=proof) as read:
            actual=E.discover_owned_groups('nonce',20,{},100)
        self.assertEqual(set(actual),{100,101});self.assertEqual([c.args[0] for c in read.call_args_list],[100,101])


@unittest.skipUnless(importlib.util.find_spec('torch'),'existing CPU torch required')
class StepStateTests(unittest.TestCase):
    def setUp(self):
        import torch
        sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'applications/speaker_rec'))
        from wespeaker_lite.utils import numerical_failure
        self.torch=torch;self.N=numerical_failure;torch.set_num_threads(1)
        self.model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4),torch.nn.Dropout(.3),torch.nn.Linear(4,2)).train()
        self.opt=torch.optim.SGD(self.model.parameters(),lr=.01,momentum=.9)
        self.scaler=torch.amp.GradScaler('cpu',enabled=True)
        self.wave=torch.arange(32,dtype=torch.float32).reshape(8,4)/20

    def step(self):
        loss=self.model(self.wave).square().mean();self.opt.zero_grad();self.scaler.scale(loss).backward()
        self.scaler.step(self.opt);self.scaler.update();return float(loss.detach())

    def test_actual_toy_save_reload_second_step_is_bitwise_equal(self):
        self.step();first=E.make_step_state(self.model,self.opt,self.scaler,self.N)
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'step.pt';self.torch.save(first,file)
            expected_loss=self.step();expected=E.make_step_state(self.model,self.opt,self.scaler,self.N)
            state=self.torch.load(file,weights_only=True,map_location='cpu',mmap=True)
            # Fresh constructor consumes RNG and creates unrelated initial weights.
            self.model=self.torch.nn.Sequential(self.torch.nn.Linear(4,4),self.torch.nn.BatchNorm1d(4),
                self.torch.nn.Dropout(.3),self.torch.nn.Linear(4,2)).train()
            self.opt=self.torch.optim.SGD(self.model.parameters(),lr=.3,momentum=.9)
            self.scaler=self.torch.amp.GradScaler('cpu',init_scale=16,enabled=True)
            E.restore_step_state(self.model,self.opt,self.scaler,state,self.N)
            observed_loss=self.step();observed=E.make_step_state(self.model,self.opt,self.scaler,self.N)
        self.assertEqual(observed_loss,expected_loss)
        def exact(a,b):
            if self.torch.is_tensor(a):self.assertTrue(self.torch.equal(a,b));return
            if isinstance(a,dict):self.assertEqual(set(a),set(b));[exact(a[k],b[k]) for k in a];return
            if isinstance(a,(list,tuple)):self.assertEqual(len(a),len(b));[exact(x,y)for x,y in zip(a,b)];return
            self.assertEqual(a,b)
        exact(observed,expected)

    def test_wrong_format_and_optimizer_order_refused(self):
        self.step();state=E.make_step_state(self.model,self.opt,self.scaler,self.N)
        wrong=copy.deepcopy(state);wrong['synthetic_engineering_only']=False
        with self.assertRaises(ValueError):E.restore_step_state(self.model,self.opt,self.scaler,wrong,self.N)
        wrong=copy.deepcopy(state);wrong['optimizer_parameter_names'][0].reverse()
        with self.assertRaises(ValueError):E.restore_step_state(self.model,self.opt,self.scaler,wrong,self.N)

    def test_real_cpu_AMP_skip_has_no_optimizer_hook_then_normal_update_calls_once(self):
        called=[];hook=self.opt.register_step_post_hook(lambda *_:called.append(True))
        before=self.scaler.get_scale();self.opt.zero_grad()
        self.scaler.scale(self.model(self.wave).sum()*float('inf')).backward()
        self.scaler.step(self.opt);self.scaler.update()
        self.assertEqual(called,[]);self.assertLess(self.scaler.get_scale(),before)
        self.step();self.assertEqual(called,[True]);hook.remove()


if __name__=='__main__':unittest.main()
