"""为同机有限转写实验准备三域完整原句PCM和真实本地依赖描述，不执行模型。

只读取显式索引中的最多三个指定分片，各取至多两个20秒内的完整来源。
原音、旧标注和旧合同不改；来源资格不释放。默认选择不能代表全库质量。
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import sys

if __package__:
    from . import audio_windows
    from .asr_stage_store import physical, require, digest, canonical
    from .vllm_backend import validate_spec, _file_sha, POLICY
else:
    import audio_windows
    from asr_stage_store import physical, require, digest, canonical
    from vllm_backend import validate_spec, _file_sha, POLICY

DOMAINS={'voxceleb2','voxblink2','cnceleb2'}


def pinned(path,expected):
    p=physical(path);raw=p.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected,'input SHA differs: '+str(p))
    return json.loads(raw)


def json_new(path,value):
    with path.open('xb') as f:f.write(canonical(value))
    return {'path':str(path),'sha256':_file_sha(path)}


def local_engine_description(contract):
    """静态读取已安装官方源码与模型配置，不import qwen或vllm。"""
    roots={}
    for name in ('qwen_asr','vllm'):
        found=importlib.util.find_spec(name)
        require(found is not None and found.submodule_search_locations,'local package missing: '+name)
        roots[name]=physical(next(iter(found.submodule_search_locations)))
    qwen=roots['qwen_asr'];engine=roots['vllm']
    utils=qwen/'inference/utils.py';tree=ast.parse(utils.read_text())
    languages=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='SUPPORTED_LANGUAGES' for x in node.targets):
            languages=ast.literal_eval(node.value)
        elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=='SUPPORTED_LANGUAGES':
            languages=ast.literal_eval(node.value)
    require(isinstance(languages,(list,tuple,set)) and languages,'supported language literal unavailable')
    model=contract['models']['asr'];root=physical(model['path'])
    config=json.loads((root/'config.json').read_text());eos=config.get('eos_token_id')
    # tokenizer_config may be the authoritative place for special tokens.
    if eos is None:
        tc=json.loads((root/'tokenizer_config.json').read_text());token=tc.get('eos_token')
        token=token.get('content') if isinstance(token,dict) else token
        ids={int(i) for i,value in tc.get('added_tokens_decoder',{}).items()
             if isinstance(value,dict) and value.get('content')==token}
        if (root/'tokenizer.json').is_file():
            tok=json.loads((root/'tokenizer.json').read_text())
            ids.update(x['id'] for x in tok.get('added_tokens',[]) if x.get('content')==token)
        if (root/'added_tokens.json').is_file():
            added=json.loads((root/'added_tokens.json').read_text())
            if token in added:ids.add(added[token])
        require(len(ids)==1,'actual EOS token not uniquely identified');eos=next(iter(ids))
    eos=[eos] if type(eos)is int else eos
    files=model['files'];pins={k:(v['sha256'] if isinstance(v,dict) else v) for k,v in files.items()}
    sources=[qwen/'inference/qwen3_asr.py',utils,engine/'entrypoints/llm.py',engine/'sampling_params.py']
    runtime=contract['runtime']
    return validate_spec({'schema':'annotation-vllm-asr-spec/v1','purpose':'pilot_validation',
        'model_API_calls_allowed':0,'release_training_labels':False,
        'model':{'path':str(root),'files_sha256':pins},
        'engine':{'backend':'vllm','version':md.version('vllm'),'dtype':'bfloat16',
            'tensor_parallel_size':1,'gpu_memory_utilization':0.15,'max_model_len':2048},
        'qwen_asr_version':md.version('qwen-asr'),'decode':{'temperature':0.0,
            'max_tokens':runtime['max_new_tokens'],'seed':0,'n':1,'ignore_eos':False},
        'max_batch_rows':2,'normalization_profile':runtime.get('normalization_profile',POLICY),
        'preprocessing':{'model_sample_rate':16000,'pcm_policy':'float32le_mono/v1',
            'window_policy_sha256':digest(runtime['window'])},
        'batch_policy':'fixed_cohort_until_durable; immutable_published_result_reuse/v1',
        'eos_token_ids':eos,'supported_languages':sorted(languages),'force_language':None,'context':'',
        'source_files':{str(p):_file_sha(p) for p in sources}})


def prepare(*,index_path,index_sha,contract_path,contract_sha,ordinals,output,engine_describer=local_engine_description):
    import numpy as np
    import soundfile as sf
    index=pinned(index_path,index_sha);contract=pinned(contract_path,contract_sha)
    require(index['source_count']==11521178 and set(index['domains'])==DOMAINS,'original three-training-domain index required')
    require(contract.get('release_training_labels') is False,'no training-label release permitted')
    require(len(ordinals)==3 and len(set(ordinals))==3 and all(type(i)is int and 0<=i<len(index['shards']) for i in ordinals),
            'three explicit distinct original shard ordinals required')
    out=physical(output);require(not out.exists() and out.parent.is_dir(),'new external output directory required')
    for p in (physical(index_path).parent,physical(contract_path).parent):
        require(out!=p and p not in out.parents and out not in p.parents,'output overlaps original inputs')
    spec=engine_describer(contract)
    require(spec['decode']['max_tokens']<=512,'generation cap exceeds bounded validation')
    out.mkdir();(out/'pcm').mkdir();selected=[];rejected=[];counts={d:0 for d in DOMAINS};seen=set()
    for ordinal in ordinals:
        descriptor=index['shards'][ordinal];rel=Path(descriptor['path'])
        require(not rel.is_absolute() and '..' not in rel.parts,'unsafe source shard path')
        path=physical(Path(index_path).parent/rel);raw=path.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==descriptor['sha256'],'source shard SHA differs')
        rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
        require(len(rows)==descriptor['count'],'source shard row count differs')
        for source in rows[:64]:
            dataset=source['dataset'];require(dataset in DOMAINS,'extra dataset in selected original shard')
            if counts[dataset]>=2:continue
            reason=None
            try:
                require(isinstance(source.get('source_id'),str) and source['source_id'] and
                        source['source_id'] not in seen,'duplicate or missing source identity')
                require(source.get('source_interval') is None,'whole utterance required, not an arbitrary crop')
                require(not source.get('protection',{}).get('training_forbidden',False),'protected diagnostic source excluded')
                info=sf.info(source['audio_path'])
                require(info.samplerate<=info.frames<=20*info.samplerate,'whole utterance outside 1--20 seconds')
                identity,errors=audio_windows.inspect_source(source)
                require(not errors and identity['decode_status']=='decoded_complete','complete source identity/decode failed')
                windows=audio_windows.plan_windows(identity,source,contract['runtime'])
                require(len(windows)==1,'whole utterance produced multiple windows')
                window=windows[0];wave=audio_windows.read_window(source,identity,window)
                p=out/'pcm'/('%02d.npy'%len(selected));np.save(p,wave,allow_pickle=False)
                selected.append({'source_id':source['source_id'],'window_id':window['window_id'],
                    'input_pcm_sha256':window['input_pcm_sha256'],'model_input_frames':window['model_input_frames'],
                    'model_sample_rate':16000,'pcm_path':str(p),'pcm_file_sha256':_file_sha(p),
                    'source':source,'source_identity':identity,'window':window,'original_shard_ordinal':ordinal,
                    'usable_for_training':False,'usable_for_evaluation':False})
                counts[dataset]+=1;seen.add(source['source_id'])
            except Exception as exc:
                import traceback
                reason={'source_id':source.get('source_id'),'error':str(exc),'traceback':traceback.format_exc()}
            if reason:rejected.append(reason)
    json_new(out/'selection-rejections.json',rejected)
    require(all(v==2 for v in counts.values()),'failed to select two whole utterances in each domain; evidence preserved')
    inputs=json_new(out/'inputs.json',selected);adapter=json_new(out/'adapter-spec.json',spec)
    report={'schema':'annotation-live-input-preparation/v1','original_index':{'path':str(index_path),'sha256':index_sha},
        'original_contract':{'path':str(contract_path),'sha256':contract_sha},'domain_counts':counts,
        'source_count':len(selected),'PCM_seconds':sum(v['model_input_frames'] for v in selected)/16000,
        'source_identity_verified':True,'whole_utterances_only':True,'selection':'first_eligible_at_most_64_rows_per_explicit_shard',
        'quality_representative_sample_claimed':False,'inputs':inputs,'adapter_spec':adapter,
        'GPU_or_model_executed':False,'training_or_evaluation_released':False}
    json_new(out/'preparation.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','index-sha','contract','contract-sha','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--shard-ordinals',required=True,help='三个已经核实的原分片序号，逗号分隔')
    a=p.parse_args();r=prepare(index_path=a.index,index_sha=a.index_sha,contract_path=a.contract,
        contract_sha=a.contract_sha,ordinals=[int(x) for x in a.shard_ordinals.split(',')],output=a.output)
    print(json.dumps(r,ensure_ascii=False))


if __name__=='__main__':main()
