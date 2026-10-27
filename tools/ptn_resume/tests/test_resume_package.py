"""Synthetic source trees, terminals and tiny CPU states. Never use old runtime."""
import contextlib
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import os
import unittest
from unittest.mock import patch
import torch
import yaml
from tools.ptn_resume import contracts as C,build as B,state_gate as G


def pin(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return {'path':str(path),'sha256':pin(path)}
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(C.canonical(value));return path


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.backbone=torch.nn.Linear(2,2);self.projection=torch.nn.Linear(2,3)
    def forward(self,x):return self.projection(self.backbone(x))


class Scheduler:
    def __init__(self,opt):self.opt=opt;self.value=None;self.last=None
    def step(self,i):
        self.last=i;self.value=.72
        for group in self.opt.param_groups:group['lr']=.72
    def get_lr(self):return self.value


class Margin:
    def step(self,i):self.last=i
    def get_margin(self):return .2


class ResumeTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory()as temp:
            r=Path(temp).resolve();source=r/'old-source';source.mkdir();legacy=r/'legacy';legacy.mkdir();old=r/'original';prev=r/'failed'
            model=Model();opt=torch.optim.SGD([{'params':model.backbone.parameters()},{'params':model.projection.parameters()}],lr=.72,momentum=.9)
            model(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad()
            scaler=torch.amp.GradScaler('cpu',enabled=True)
            ck=prev/'ptn/models/model_47.pt';ck.parent.mkdir(parents=True);torch.save({'epoch':47,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict()},ck)
            spk=write(prev/'ptn/spk2id.json',{'synthetic':0})
            names=[C.PATCH_NAMES[0]]+['synthetic_%03d.py'%i for i in range(188)];rows=[]
            for name in names:
                path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'# synthetic source\n');path.chmod(0o444)
                rows.append({'relative':name,'sha256':pin(path),'bytes':path.stat().st_size,'mode':0o444})
            s95=r/'source95.sha256';s95.write_text(''.join(x['sha256']+'  '+x['relative']+'\n'for x in rows[:95]))
            original_config={'checkpoint':str(old/'ptn/models/model_35.pt'),'exp_dir':str(prev/'ptn'),
                'num_epochs':120,'enable_amp':True,'optimizer_args':{'lr':.06,'weight_decay':2e-5,'momentum':.9,'nesterov':True},'exact_type_probe':{'integer':96,'bool':False,'float_small':1e-5,'float_integral':12.0,'null':None,'nested':[True,0,2e-5]},'dataloader_args':{'batch_size':96,'num_workers':24},
                'dataset_args':{'num_frms':200,'legacy_aug':True},'scheduler_args':{'epoch_iter':15001,'scale_ratio':12},'projection_args':{'num_class':357822}}
            for name in C.RUNTIME_NAMES:(legacy/name).write_bytes(b'# historical synthetic wrapper\n' if name.endswith('.py') else b'# synthetic\n')
            write(legacy/'audio_selfcheck.json',{});write(legacy/'dependencies.json',{});write(legacy/'failed_task_terminal.json',{});write(legacy/'parent_terminal_attestation.json',{})
            write(legacy/'source_189.json',rows);(legacy/'resume.yaml').write_text(yaml.safe_dump(original_config,sort_keys=True))
            entries={p.name:pin(p)for p in legacy.iterdir()};write(legacy/'bundle_files.json',entries)
            terminals=[]
            for task,run,state in [('t-20261019232824-8zqrl',old,'Terminated'),('t-20261024154424-vggvm',prev,'Failed')]:
                p=write(r/(state+'.json'),{'schema':'ptn-predecessor-terminal/v1','task_id':task,'run_root':str(run),'state':state,'active_workers':0,
                    'worker_states':[{'name':'worker_0','state':state}],'observed_utc':dt.datetime.now(dt.timezone.utc).isoformat()});terminals.append(ref(p))
            patches=[]
            for i,name in enumerate(C.PATCH_NAMES):
                p=r/('patch%d.py'%i);p.write_text('# reviewed synthetic patch\nvalue='+str(i)+'\n');patches.append({'relative':name,**ref(p)})
            runtime=[]
            for name in ('entry.py','worker.py'):
                p=r/('new-'+name);p.write_text('# new synthetic wrapper\n');runtime.append({'name':name,**ref(p)})
            spec={'schema':'ptn-resume-build-spec/v1','legacy_bundle_root':str(legacy),'source_root':str(source),'source95':ref(s95),
                  'checkpoint':ref(ck),'speaker_map':ref(spk),'original_terminal':terminals[0],'predecessor_terminal':terminals[1],
                  'new_run_root':str(r/'new-run'),'patches':patches,'runtime_files':runtime}
            constants={'BASE':str(r),'ORIGINAL':str(old),'PREDECESSOR':str(prev),'SOURCE':str(source),'CHECKPOINT':str(ck),
                'CK_SHA':pin(ck),'CHECKPOINT_BYTES':ck.stat().st_size,'SPK_SHA':pin(spk),'LEGACY_BUNDLE_SHA':pin(legacy/'bundle_files.json'),
                'SOURCE189_SHA':pin(legacy/'source_189.json'),'SOURCE95_SHA':pin(s95)}
            with patch.multiple(C,**constants),patch.multiple(G,MODEL_STATES=4,OPTIMIZER_STATES=4,PARAMETER_COUNT=4,GROUP_SIZES=[2,2],HEAD_SHAPE=[3,2]):
                yield r,spec,model,opt,scaler

    def test_new_package_preserves_original_source_and_has_two_explicit_changes(self):
        with self.fixture()as(r,spec,*_):
            before={p:pin(p)for p in Path(spec['source_root']).rglob('*')if p.is_file()}
            result=B.build(spec,r/'package');c=G.verify_package(r/'package/resume-contract.json',result['contract_sha256'])
            self.assertTrue(result['platform_task_name'].startswith('zhounan_'));self.assertNotIn('platform_task_name',c)
            self.assertFalse(result['runtime_admission']);self.assertFalse(Path(spec['new_run_root']).exists())
            self.assertEqual(before,{p:pin(p)for p in before})
            self.assertEqual(len(json.loads((r/'package/source_manifest.json').read_bytes())),190)
            self.assertEqual(c['clock']['absolute_iteration'],705047)
            self.assertEqual(set(c['runtime_files']),{'entry.py','worker.py','state_gate.py','ptn_contracts.py'})

    def test_future_platform_name_is_receipt_only_and_invalid_name_reads_nothing(self):
        with patch.object(B,'validate_materials')as read:
            with self.assertRaisesRegex(ValueError,'zhounan_'):B.build({},Path('/unused/new-package'),task_name='old-prefix')
            read.assert_not_called()
        with self.fixture()as(r,spec,*_):
            result=B.build(spec,r/'package',task_name='zhounan_ptn_custom')
            self.assertEqual(result['platform_task_name'],'zhounan_ptn_custom')
            self.assertNotIn('platform_task_name',json.loads((r/'package/resume-contract.json').read_bytes()))

    def test_deployed_gate_imports_only_its_frozen_helper_without_repository(self):
        with self.fixture()as(r,spec,*_):
            B.build(spec,r/'package');runtime=r/'package/runtime'
            code="import sys,pathlib;sys.path.insert(0,sys.argv[1]);import state_gate,ptn_contracts;assert pathlib.Path(ptn_contracts.__file__).parent==pathlib.Path(sys.argv[1]);assert 'torch'not in sys.modules;print('standalone-gate-imported')"
            env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
            result=subprocess.run([sys.executable,'-B','-c',code,str(runtime)],cwd=r,env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('standalone-gate-imported',result.stdout)

    def test_resume_yaml_preserves_scientific_floats_and_exact_nested_types(self):
        with self.fixture()as(r,spec,*_):
            B.build(spec,r/'package');actual=yaml.safe_load((r/'package/resume.yaml').read_bytes())
            expected=yaml.safe_load((Path(spec['legacy_bundle_root'])/'resume.yaml').read_bytes())
            for key in ('lr','weight_decay','momentum'):
                self.assertIs(type(actual['optimizer_args'][key]),float)
                self.assertEqual(actual['optimizer_args'][key],expected['optimizer_args'][key])
            self.assertEqual(C.canonical(actual['exact_type_probe']),C.canonical(expected['exact_type_probe']))
            self.assertIs(type(actual['exact_type_probe']['integer']),int)
            self.assertIs(type(actual['exact_type_probe']['bool']),bool)
            self.assertIs(type(actual['exact_type_probe']['float_integral']),float)
            parameter=torch.nn.Parameter(torch.ones(1));opt=torch.optim.SGD([parameter],**actual['optimizer_args'])
            self.assertEqual(opt.param_groups[0]['weight_decay'],2e-5)

    def test_complete_real_tiny_CPU_model_optimizer_scaler_comparison(self):
        with self.fixture()as(r,spec,model,opt,scaler):
            result=B.build(spec,r/'package');c=G.verify_package(r/'package/resume-contract.json',result['contract_sha256'])
            scheduler=Scheduler(opt);margin=Margin()
            value=G.verify_restored(r/'package/resume-contract.json',result['contract_sha256'],model=model,optimizer=opt,scaler=scaler,
                scheduler=scheduler,margin=margin,config=yaml.safe_load((r/'package/resume.yaml').read_bytes()),speaker_map_path=spec['speaker_map']['path'])
            self.assertEqual(scheduler.last,705047);self.assertEqual(margin.last,705047)
            self.assertTrue(value['ordered_optimizer_exact']);self.assertFalse(value['runtime_admission'])

    def test_mutated_model_and_reordered_parameter_bindings_refused(self):
        for mode in ('model','order','momentum','scaler'):
            with self.subTest(mode=mode),self.fixture()as(r,spec,model,opt,scaler):
                result=B.build(spec,r/'package')
                if mode=='model':
                    with torch.no_grad():next(model.parameters()).add_(1)
                if mode=='order':opt.param_groups[0]['params'].reverse()
                if mode=='momentum':next(iter(opt.state.values()))['momentum_buffer'].add_(1)
                if mode=='scaler':scaler.load_state_dict({**scaler.state_dict(),'scale':2.})
                with self.assertRaises(ValueError):G.verify_restored(r/'package/resume-contract.json',result['contract_sha256'],model=model,optimizer=opt,scaler=scaler,
                    scheduler=Scheduler(opt),margin=Margin(),config=yaml.safe_load((r/'package/resume.yaml').read_bytes()),speaker_map_path=spec['speaker_map']['path'])

    def test_no_unlisted_or_duplicate_patch_and_no_source_drift(self):
        for mode in ('extra','duplicate','drift','member','mode','no-op'):
            with self.subTest(mode=mode),self.fixture()as(r,spec,*_):
                if mode=='extra':spec['patches'][0]['relative']='other.py'
                if mode=='duplicate':spec['patches'][1]=copy.deepcopy(spec['patches'][0])
                if mode=='drift':
                    target=Path(spec['source_root'])/'synthetic_000.py';target.chmod(0o644);target.write_text('changed')
                if mode=='member':(Path(spec['source_root'])/'extra.py').write_text('extra')
                if mode=='no-op':spec['patches'][0]={'relative':C.PATCH_NAMES[0],**ref(Path(spec['source_root'])/C.PATCH_NAMES[0])}
                if mode=='mode':(Path(spec['source_root'])/'synthetic_000.py').chmod(0o644)
                with self.assertRaises(ValueError):B.build(spec,r/'package')
                self.assertFalse((r/'package').exists())

    def test_stale_nonterminal_wrong_task_and_bool_active_workers_rejected(self):
        for change in ({'observed_utc':'2000-01-01T00:00:00+00:00'},{'state':'Running'},{'task_id':'other'},{'active_workers':False}):
            with self.subTest(change=change),self.fixture()as(r,spec,*_):
                p=Path(spec['predecessor_terminal']['path']);value=json.loads(p.read_bytes());value.update(change);write(p,value);spec['predecessor_terminal']=ref(p)
                with self.assertRaises(ValueError):B.build(spec,r/'package')
                self.assertFalse((r/'package').exists())

    def test_new_fresh_terminal_refs_after_queue_delay_do_not_rewrite_contract(self):
        with self.fixture()as(r,spec,*_):
            result=B.build(spec,r/'package');contract_path=r/'package/resume-contract.json'
            original_bytes=contract_path.read_bytes();c=G.verify_package(contract_path,result['contract_sha256'])
            later=dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=2)
            with self.assertRaises(ValueError):C.verify_fresh_terminals(c,now=later)
            refs={}
            for key in ('original','predecessor'):
                old=c['terminals'][key]['reference'];value=json.loads(Path(old['path']).read_bytes());value['observed_utc']=later.isoformat()
                refs[key]=ref(write(r/('fresh-'+key+'.json'),value))
            proof=C.verify_fresh_terminals(c,now=later,evidence=refs)
            self.assertEqual(proof['predecessor']['reference'],refs['predecessor'])
            self.assertEqual(original_bytes,contract_path.read_bytes())
            with self.assertRaises(ValueError):C.verify_fresh_terminals(c,now=later,evidence=True)

    def test_wrong_SHA_symlink_or_existing_run_never_creates_package(self):
        for mode in ('sha','symlink','existing'):
            with self.subTest(mode=mode),self.fixture()as(r,spec,*_):
                if mode=='sha':spec['checkpoint']['sha256']='0'*64
                if mode=='symlink':
                    p=r/'linked';p.symlink_to(spec['source_root'],target_is_directory=True);spec['source_root']=str(p)
                if mode=='existing':Path(spec['new_run_root']).mkdir()
                with self.assertRaises(ValueError):B.build(spec,r/'package')
                self.assertFalse((r/'package').exists())

    def test_old_runtime_and_overlapping_destination_refused(self):
        with self.fixture()as(r,spec,*_):
            spec['runtime_files'][0]={'name':'entry.py',**ref(Path(spec['legacy_bundle_root'])/'entry.py')}
            with self.assertRaises(ValueError):B.build(spec,r/'package')
        with self.fixture()as(r,spec,*_):
            with self.assertRaises(ValueError):B.build(spec,Path(spec['source_root'])/'new-package')

    def test_resealing_mutable_bundle_index_cannot_admit_extra_source_or_runtime(self):
        for relative in ('source/extra_191.py','runtime/yaml.py'):
            with self.subTest(relative=relative),self.fixture()as(r,spec,*_):
                result=B.build(spec,r/'package');new=r/'package'/relative;new.write_text('# injected extra\n')
                index=r/'package/bundle_files.json';value=json.loads(index.read_bytes());value[relative]=pin(new);write(index,value)
                with self.assertRaises(ValueError):G.verify_package(r/'package/resume-contract.json',result['contract_sha256'])

    def test_clock_and_package_tampering_rejected(self):
        with self.fixture()as(r,spec,*_):
            result=B.build(spec,r/'package');(r/'package/runtime/entry.py').write_text('# changed\n')
            with self.assertRaises(ValueError):G.verify_package(r/'package/resume-contract.json',result['contract_sha256'])
        for change in ({'start_epoch':36},{'absolute_iteration':525035},{'optimizer':'fresh'},{'rng_restored':True}):
            clock={**C.CLOCK,**change}
            with self.subTest(change=change),self.assertRaises(ValueError):C.validate_clock(clock)


if __name__=='__main__':unittest.main()
