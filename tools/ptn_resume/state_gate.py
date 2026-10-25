"""Exact CPU/state restoration gate. No model construction, forward or execution."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
try:
    from . import contracts as C
except ImportError:
    import ptn_contracts as C

MODEL_STATES=823
OPTIMIZER_STATES=584
PARAMETER_COUNT=585
GROUP_SIZES=[583,2]
HEAD_SHAPE=[357822,192]


def exact(actual, expected, path='state'):
    import torch
    if torch.is_tensor(expected):
        C.need(torch.is_tensor(actual) and actual.dtype==expected.dtype and actual.shape==expected.shape,path+': tensor geometry differs')
        C.need(not(expected.is_floating_point()or expected.is_complex())or bool(torch.isfinite(expected).all()),path+': saved tensor nonfinite')
        C.need(torch.equal(actual.detach().cpu(),expected.detach().cpu()),path+': tensor values differ')
    elif isinstance(expected,dict):
        C.need(isinstance(actual,dict)and set(actual)==set(expected),path+': dictionary keys differ')
        for key in expected:exact(actual[key],expected[key],path+'.'+str(key))
    elif isinstance(expected,(tuple,list)):
        C.need(type(actual)is type(expected)and len(actual)==len(expected),path+': sequence differs')
        for i,(a,b)in enumerate(zip(actual,expected)):exact(a,b,path+'.'+str(i))
    else:
        C.need(type(actual)is type(expected)and actual==expected,path+': scalar differs')
        C.need(not isinstance(expected,float)or math.isfinite(expected),path+': nonfinite scalar')


def verify_package(contract_path, contract_sha):
    c=C.read_contract(contract_path,contract_sha);base=C.physical(contract_path).parent
    manifest=C.json_bytes((base/'bundle_files.json').read_bytes());C.need(isinstance(manifest,dict),'new bundle manifest required')
    actual={str(p.relative_to(base))for p in base.rglob('*')if p.is_file()or p.is_symlink()}
    C.need(actual==set(manifest)|{'bundle_files.json'},'new package membership differs')
    for rel,pin in manifest.items():
        C.relative(rel);h,_=C.hash_file(base/rel);C.need(h==pin,'package file drift: '+rel)
    C.need(manifest.get('resume-contract.json')==contract_sha,'new manifest contract pin differs')
    source=C.physical(c['source_root']);C.need(source==base/'source','source must be independent package tree')
    raw=(base/'source_manifest.json').read_bytes();C.need(hashlib.sha256(raw).hexdigest()==c['source_manifest_sha256'],'new source ledger differs')
    rows=C.json_bytes(raw);C.need(isinstance(rows,list)and len(rows)==190,'new source count must be190')
    names={row['relative']for row in rows}
    C.need(len(names)==190,'duplicate new source ledger name')
    actual_source={str(p.relative_to(source))for p in source.rglob('*')if p.is_file()or p.is_symlink()}
    C.need(actual_source==names,'source tree has missing/unlisted member')
    expected_members={'source/'+name for name in names}|{'runtime/'+name for name in c['runtime_files']}|{
        'resume.yaml','source_manifest.json','source_delta.json','resume-contract.json'}
    C.need(set(manifest)==expected_members,'new bundle manifest admits unapproved extra member')
    for row in rows:
        rel=str(C.relative(row['relative']));h,s=C.hash_file(source/rel)
        C.need(h==row['sha256']and s[2]==row['bytes']and (source/rel).stat().st_mode&0o7777==row['mode'],'new source ledger mismatch: '+rel)
    raw=(base/'source_delta.json').read_bytes();C.need(hashlib.sha256(raw).hexdigest()==c['source_delta_sha256'],'source delta differs')
    delta=C.json_bytes(raw)
    C.need(delta.get('original_source189_sha256')==C.SOURCE189_SHA and delta.get('members_before')==189 and delta.get('members_after')==190,'wrong source delta base')
    changes=delta.get('changes');C.need(isinstance(changes,list)and len(changes)==2 and
        {r['relative']for r in changes}==set(C.PATCH_NAMES),'unapproved source delta')
    C.need(C.hash_file(c['resume_config_path'])[0]==c['resume_config_sha256'],'resume config changed')
    for name,pin in c['runtime_files'].items():C.need(C.hash_file(base/'runtime'/name)[0]==pin,'runtime identity differs')
    return c


def verify_restored(contract_path,contract_sha,*,model,optimizer,scaler,scheduler,margin,config,speaker_map_path,terminal_evidence=None):
    """Caller constructs/loads CPU objects; this gate independently reads model47.

    It does not accept restoration booleans or a caller-supplied payload. The CPU
    payload is safe-weights-only/mmap; failures never fall back to arbitrary pickle.
    Rank executions may compare GPU objects via exact CPU readback before release.
    External forward/update blocking, stable locks, decoder/maps, resources and
    all-rank consensus are separate executor obligations, never proven here.
    """
    import torch
    c=verify_package(contract_path,contract_sha)
    terminals=C.verify_fresh_terminals(c,evidence=terminal_evidence)
    C.need(not torch.cuda.is_initialized() if all(p.device.type=='cpu'for p in model.parameters())else True,'CPU state check inherited CUDA context')
    ck=C.physical(C.CHECKPOINT);h,before=C.hash_file(ck);C.need(h==C.CK_SHA and before[2]==C.CHECKPOINT_BYTES,'model47 pin/size changed')
    payload=torch.load(ck,map_location='cpu',weights_only=True,mmap=True)
    C.need(isinstance(payload,dict)and set(payload)=={'epoch','model','optimizer','scaler'}and type(payload['epoch'])is int and payload['epoch']==47,'not a complete same-stage model47')
    C.need(isinstance(payload['model'],dict)and len(payload['model'])==MODEL_STATES and
        isinstance(payload['optimizer'],dict)and len(payload['optimizer'].get('state',{}))==OPTIMIZER_STATES,'saved state denominator differs')
    import yaml
    expected=yaml.safe_load(Path(c['resume_config_path']).read_bytes())
    C.need(C.canonical(config)==C.canonical(expected),'effective config differs from fixed new resume config')
    saved_map=C.physical(c['speaker_map']['path']);actual_map=C.physical(speaker_map_path)
    C.need(actual_map in {saved_map,C.physical(Path(c['new_run_root'])/'ptn/spk2id.json')}and
        C.hash_file(saved_map)[0]==C.SPK_SHA and C.hash_file(actual_map)[0]==C.SPK_SHA,'saved/recomputed speaker map identity differs')
    C.need(list(model.projection.weight.shape)==HEAD_SHAPE,'classification head shape changed')
    C.need(len(model.state_dict())==MODEL_STATES,'model state denominator differs');exact(model.state_dict(),payload['model'],'model')
    params=list(model.backbone.parameters())+list(model.projection.parameters())
    C.need(len(params)==PARAMETER_COUNT and [len(g['params'])for g in optimizer.param_groups]==GROUP_SIZES,'optimizer group geometry differs')
    C.need([id(p)for g in optimizer.param_groups for p in g['params']]==[id(p)for p in params],'optimizer parameter order/binding differs')
    C.need(type(optimizer)is torch.optim.SGD,'same-stage optimizer must be actual SGD')
    exact(optimizer.state_dict(),payload['optimizer'],'optimizer')
    C.need(len(optimizer.state)==OPTIMIZER_STATES,'optimizer momentum missing')
    for param,state in optimizer.state.items():
        C.need(set(state)=={'momentum_buffer'}and state['momentum_buffer'].shape==param.shape and bool(torch.isfinite(state['momentum_buffer']).all()),'momentum geometry/nonfinite')
    C.need(scaler.is_enabled(),'AMP scaler disabled');exact(scaler.state_dict(),payload['scaler'],'scaler')
    # Actual frozen scheduler methods, not a newly reimplemented formula.
    scheduler.step(705047);margin.step(705047)
    C.need(abs(scheduler.get_lr()-.72)<1e-12 and all(abs(g['lr']-.72)<1e-12 for g in optimizer.param_groups)and abs(margin.get_margin()-.2)<1e-12,'actual frozen LR/margin clock differs')
    C.need(C.snapshot(ck.stat())==before,'checkpoint changed during full restoration comparison')
    return {'schema':'ptn-model47-exact-state-proof/v1','contract_sha256':contract_sha,
        'checkpoint_sha256':C.CK_SHA,'speaker_map_sha256':C.SPK_SHA,'actual_speaker_map_path':str(actual_map),'saved_epoch':47,'start_epoch':48,
        'absolute_iteration':705047,'model_state_count':MODEL_STATES,'ordered_parameters':PARAMETER_COUNT,
        'optimizer_state_count':OPTIMIZER_STATES,'actual_lr':scheduler.get_lr(),'actual_margin':margin.get_margin(),
        'model_exact':True,'ordered_optimizer_exact':True,'scaler_exact':True,'actual_scaler':scaler.state_dict(),
        'state_gate_sha256':C.hash_file(Path(__file__).resolve())[0], 'checkpoint_stat':before,
        'actual_terminal_evidence':terminals,
        'model_forward_or_optimizer_update_called_by_gate':False,'locks_acquired_by_gate':False,
        'runtime_admission':False,'all_rank_admission':False,'rng_restored':False}
