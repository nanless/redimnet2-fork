"""Build a new immutable model47 package, without importing or executing trainers."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from . import contracts as C

SPEC_KEYS={'schema','legacy_bundle_root','source_root','source95','checkpoint','speaker_map',
           'original_terminal','predecessor_terminal','new_run_root','patches','runtime_files'}


def legacy_files(root):
    root=C.physical(root);raw=(root/'bundle_files.json').read_bytes()
    C.need(hashlib.sha256(raw).hexdigest()==C.LEGACY_BUNDLE_SHA,'historical bundle index differs')
    entries=C.json_bytes(raw);C.need(isinstance(entries,dict)and len(entries)==16,'original 16 members required')
    C.need({p.name for p in root.iterdir()}==set(entries)|{'bundle_files.json'},'unlisted original bundle member')
    for name,pin in entries.items():
        C.need(str(C.relative(name))==name and '/'not in name and C.is_sha(pin),'bad legacy member')
        actual,_=C.hash_file(root/name);C.need(actual==pin,'legacy member changed: '+name)
    C.need(entries.get('source_189.json')==C.SOURCE189_SHA,'source189 historical pin differs')
    ledger=C.json_bytes((root/'source_189.json').read_bytes())
    return root,entries,ledger


def verify_source(root,ledger,source95):
    root=C.physical(root);C.need(root.is_dir(),'old source tree missing')
    C.need(isinstance(ledger,list)and len(ledger)==189,'original 189 ledger required')
    by_name={}
    for row in ledger:
        C.need(isinstance(row,dict) and {'relative','sha256','bytes','mode'}<=set(row),'invalid original source row')
        rel=str(C.relative(row['relative']));C.need(rel not in by_name and C.is_sha(row['sha256']),'duplicate/bad source pin')
        C.need(type(row['bytes'])is int and row['bytes']>=0 and type(row['mode'])is int,'invalid source size/mode')
        p=C.physical(root/rel);actual,snapshot=C.hash_file(p)
        C.need(actual==row['sha256'] and snapshot[2]==row['bytes'] and stat.S_IMODE(p.stat().st_mode)==row['mode'],'original source changed: '+rel)
        by_name[rel]=row
    actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()or p.is_symlink()}
    C.need(actual==set(by_name),'source189 membership differs')
    rows=C.read_ref(source95,C.SOURCE95_SHA).decode().splitlines();C.need(len(rows)==95,'original source95 count differs')
    seen=set()
    for line in rows:
        pin,name=line.split(None,1);name=name.lstrip('* ')
        path=Path(name)
        rel=str(path.relative_to('/root/code/github_repos/redimnet2-fork')) if path.is_absolute()else str(path)
        C.relative(rel);C.need(rel not in seen and rel in by_name and by_name[rel]['sha256']==pin,'source95/189 mismatch');seen.add(rel)
    return by_name


def config_from_legacy(root,new_run):
    import yaml
    original=yaml.safe_load((root/'resume.yaml').read_bytes());C.need(isinstance(original,dict),'legacy config object required')
    C.need(original.get('checkpoint')==C.ORIGINAL+'/ptn/models/model_35.pt' and original.get('exp_dir')==C.PREDECESSOR+'/ptn','wrong legacy config lineage')
    C.need(original.get('model_ckpt')is None and original['num_epochs']==120 and original['enable_amp']is True and
           original['dataloader_args']['batch_size']==96 and original['dataloader_args']['num_workers']==24 and
           original['dataset_args']['num_frms']==200 and original['dataset_args']['legacy_aug']is True and
           original['scheduler_args']['epoch_iter']==15001 and original['scheduler_args']['scale_ratio']==12 and
           original['projection_args']['num_class']==357822,'legacy training geometry differs')
    changed=copy.deepcopy(original);changed['checkpoint']=C.CHECKPOINT;changed['exp_dir']=str(new_run/'ptn')
    return changed


def validate_materials(spec,now=None):
    C.need(isinstance(spec,dict) and set(spec)==SPEC_KEYS and spec.get('schema')=='ptn-resume-build-spec/v1','wrong build spec keys/schema')
    legacy,legacy_index,ledger=legacy_files(spec['legacy_bundle_root'])
    by_name=verify_source(spec['source_root'],ledger,spec['source95'])
    C.need(spec['checkpoint']=={'path':C.CHECKPOINT,'sha256':C.CK_SHA},'model47 path/pin is fixed')
    actual,checkpoint_stat=C.hash_file(C.CHECKPOINT);C.need(actual==C.CK_SHA and checkpoint_stat[2]==C.CHECKPOINT_BYTES,'model47 bytes differ')
    C.read_ref(spec['speaker_map'],C.SPK_SHA)
    C.need(spec['speaker_map']['path']==C.PREDECESSOR+'/ptn/spk2id.json','speaker map must be actual failed-run map')
    original_terminal=C.terminal(spec['original_terminal'],'t-20261019232824-8zqrl',C.ORIGINAL,{'Terminated'},now)
    predecessor_terminal=C.terminal(spec['predecessor_terminal'],'t-20261024154424-vggvm',C.PREDECESSOR,{'Failed'},now)
    run=C.physical(spec['new_run_root']);C.need(not run.exists(),'new training run must not exist')
    for forbidden in [C.ORIGINAL,C.PREDECESSOR,spec['source_root'],spec['legacy_bundle_root']]:
        old=C.physical(forbidden);C.need(run!=old and old not in run.parents and run not in old.parents,'run overlaps old/source tree')
    patches=spec['patches'];C.need(isinstance(patches,list) and len(patches)==2,'exactly executor replacement and new helper required')
    patch_by_name={}
    for item in patches:
        C.need(isinstance(item,dict)and set(item)=={'relative','path','sha256'},'bad patch reference')
        name=str(C.relative(item['relative']));C.need(name in C.PATCH_NAMES and name not in patch_by_name,'unreviewed/duplicate source patch')
        raw=C.read_ref({'path':item['path'],'sha256':item['sha256']});compile(raw,name,'exec');patch_by_name[name]=item
    C.need(set(patch_by_name)==set(C.PATCH_NAMES) and C.PATCH_NAMES[0]in by_name and C.PATCH_NAMES[1]not in by_name,'patch source topology differs')
    C.need(patch_by_name[C.PATCH_NAMES[0]]['sha256']!=by_name[C.PATCH_NAMES[0]]['sha256'],'executor patch must actually differ from old source')
    runtime=spec['runtime_files'];C.need(isinstance(runtime,list)and 2 <= len(runtime) <= len(C.RUNTIME_NAMES),'all new wrapper files required')
    runtime_by_name={}
    for item in runtime:
        C.need(isinstance(item,dict)and set(item)=={'name','path','sha256'},'bad runtime reference')
        name=item['name'];C.need(name in C.RUNTIME_NAMES and name not in runtime_by_name,'unexpected runtime member')
        raw=C.read_ref({'path':item['path'],'sha256':item['sha256']})
        if name.endswith('.py'):compile(raw,name,'exec')
        # Old35 wrappers cannot masquerade as newly reviewed47 wrappers.
        C.need(item['sha256']!=legacy_index[name] if name in ('entry.py','strict_restore.py','worker.py','cpu_check.py')else True,'unchanged model35 runtime refused: '+name)
        runtime_by_name[name]=item
    C.need({'entry.py','worker.py'}<=set(runtime_by_name),'new entry and worker required')
    return {'legacy':legacy,'ledger':ledger,'by_name':by_name,'run':run,'patches':patch_by_name,'runtime':runtime_by_name,
            'config':config_from_legacy(legacy,run),'checkpoint_stat':checkpoint_stat,
            'terminals':{'original':original_terminal,'predecessor':predecessor_terminal}}


def yaml_config_bytes(config):
    import yaml
    raw=yaml.safe_dump(config,sort_keys=True,allow_unicode=True).encode('utf-8')
    restored=yaml.safe_load(raw)
    C.need(C.canonical(restored)==C.canonical(config),'YAML configuration type/value roundtrip differs')
    return raw


def write_new(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def build(spec,output,now=None):
    checked=validate_materials(spec,now)
    out=C.physical(output);C.need(not out.exists(),'package destination must be new')
    protected=[C.physical(spec[k])for k in ('source_root','legacy_bundle_root','new_run_root')]+[C.physical(C.ORIGINAL),C.physical(C.PREDECESSOR)]
    C.need(all(out!=p and p not in out.parents and out not in p.parents for p in protected),'package overlaps training/source tree')
    C.need(out.parent.is_dir(),'package parent must already exist');out.mkdir()
    before_rows=[];delta=[]
    for name,row in checked['by_name'].items():
        new=checked['patches'].get(name)
        source=Path(new['path'])if new else Path(spec['source_root'])/name
        raw=C.read_ref({'path':str(source),'sha256':new['sha256']if new else row['sha256']})
        target=out/'source'/name;write_new(target,raw);target.chmod(row['mode'])
        before_rows.append({'relative':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mode':row['mode']})
        if new:delta.append({'relative':name,'change':'replace','old_sha256':row['sha256'],'new_sha256':new['sha256']})
    helper=checked['patches'][C.PATCH_NAMES[1]];raw=C.read_ref({'path':helper['path'],'sha256':helper['sha256']})
    target=out/'source'/C.PATCH_NAMES[1];write_new(target,raw);target.chmod(0o444)
    before_rows.append({'relative':C.PATCH_NAMES[1],'sha256':helper['sha256'],'bytes':len(raw),'mode':0o444})
    delta.append({'relative':C.PATCH_NAMES[1],'change':'add','old_sha256':None,'new_sha256':helper['sha256']})
    for name,item in checked['runtime'].items():write_new(out/'runtime'/name,C.read_ref({'path':item['path'],'sha256':item['sha256']}))
    write_new(out/'resume.yaml',yaml_config_bytes(checked['config']))
    write_new(out/'source_manifest.json',C.canonical(sorted(before_rows,key=lambda r:r['relative'])))
    write_new(out/'source_delta.json',C.canonical({'original_source189_sha256':C.SOURCE189_SHA,'members_before':189,'members_after':190,'changes':delta}))
    # Freeze actual maintained gate bytes beside entry; no mutable repo import.
    gate_pins={}
    for src,name in [('contracts.py','ptn_contracts.py'),('state_gate.py','state_gate.py')]:
        raw=(Path(__file__).parent/src).read_bytes();write_new(out/'runtime'/name,raw);gate_pins[name]=hashlib.sha256(raw).hexdigest()
    contract={'schema':'ptn-resume-model47/v1','runtime_admission':False,'clock':C.CLOCK,
        'checkpoint':{'path':C.CHECKPOINT,'sha256':C.CK_SHA,'bytes':C.CHECKPOINT_BYTES},'speaker_map':spec['speaker_map'],
        'new_run_root':str(checked['run']),'source_root':str(out/'source'),
        'source_manifest_sha256':C.hash_file(out/'source_manifest.json')[0],
        'source_delta_sha256':C.hash_file(out/'source_delta.json')[0],
        'legacy':{'bundle_files_sha256':C.LEGACY_BUNDLE_SHA,'source189_sha256':C.SOURCE189_SHA,'source95':spec['source95'],'bundle_root':str(checked['legacy'])},
        'resume_config_path':str(out/'resume.yaml'),'resume_config_sha256':C.hash_file(out/'resume.yaml')[0],
        'terminals':checked['terminals'],'checkpoint_stat_at_build':checked['checkpoint_stat'],
        'required_existing_locks':[str(Path(C.ORIGINAL)/p)for p in ('.reproduction.lock','ptn/.cloud_train.lock')]+[str(Path(C.PREDECESSOR)/p)for p in ('.reproduction.lock','ptn/.cloud_train.lock')],
        'logical_chain_lock':C.BASE+'/.ptn-logical-chain.lock','admission_lock':C.BASE+'/.ptn-resume-admission.lock',
        'runtime_files':{**{name:item['sha256']for name,item in checked['runtime'].items()},**gate_pins},
        'must_recheck_under_exclusive_locks':['fresh control-plane terminal','actual original four stable inode locks and logical/admission locks','full CPU model+ordered SGD+scaler restore','actual frozen scheduler705047 and margin','all eight ranks and 192 data workers','no unknown writer or resource PID'],
        'historical_rng_missing':True,'partial_epoch48_discarded':True,'LM_started':False,
        'builder_sha256':C.hash_file(Path(__file__).resolve())[0]}
    write_new(out/'resume-contract.json',C.canonical(contract))
    # This is a fresh manifest for this package, never an edit of an old pin.
    members={str(p.relative_to(out)):C.hash_file(p)[0]for p in out.rglob('*')if p.is_file()}
    write_new(out/'bundle_files.json',C.canonical(members))
    C.need(not checked['run'].exists(),'builder unexpectedly created training output')
    return {'package':str(out),'contract_sha256':C.hash_file(out/'resume-contract.json')[0],
            'bundle_manifest_sha256':C.hash_file(out/'bundle_files.json')[0], 'runtime_admission':False,
            'actual_state_restore_performed':False,'source_members':190}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args(argv)
    spec=C.json_bytes(C.physical(a.spec).read_bytes());result=build(spec,a.output_dir);print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
