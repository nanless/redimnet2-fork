"""用合成PCM和实际官方Whisper特征器核对批次补零；不加载模型或真实音频。"""
from __future__ import annotations
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

SCHEMA='annotation-native-feature-padding-diagnostic/v1'
LENGTHS=(16000,16128,49408,401408)
BATCHES=(1,4,8)
COMPANION=416000


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def pinned(ref,maximum=1024**2):
    need(isinstance(ref,dict)and set(ref)=={'path','sha256'},'explicit path/SHA required')
    path=Path(ref['path']);need(path.is_absolute()and '..'not in path.parts and all(not x.is_symlink()for x in(path,*path.parents)),'physical absolute input required')
    need(path.stat().st_size<=maximum,'bounded configuration/source required');raw=path.read_bytes();need(sha(raw)==ref['sha256'],'source/configuration SHA differs');return raw


def output_length_function(raw):
    """只编译实际官方文件中的纯算术长度函数，不导入Qwen模型类。"""
    tree=ast.parse(raw);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_get_feat_extract_output_lengths']
    need(len(nodes)==1,'one actual official feature-length function required')
    node=nodes[0]
    allowed=(ast.FunctionDef,ast.arguments,ast.arg,ast.Expr,ast.Constant,ast.Assign,ast.Name,ast.Load,ast.Store,ast.BinOp,ast.Mod,ast.Sub,ast.FloorDiv,ast.Add,ast.Mult,ast.Return)
    need(all(isinstance(n,allowed)for n in ast.walk(node)),'official length function is no longer pure bounded arithmetic')
    scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-official-length-function>','exec'),{'__builtins__':{}},scope)
    return scope[node.name],sha(ast.dump(node,include_attributes=False).encode())


def validate(plan_pin):
    p=json.loads(pinned(plan_pin));need(set(p)=={'schema','contract','feature_source','processing_source','config','output_root'}and p['schema']==SCHEMA,'explicit synthetic feature diagnostic plan required')
    contract=json.loads(pinned(p['contract']));need(contract['purpose']=='pilot_validation','actual frozen native pilot contract required')
    need(p['config']['path']==str(Path(contract['models']['asr']['path'])/'preprocessor_config.json'),'feature config must come from actual ASR model directory')
    config=json.loads(pinned(p['config']));feature=pinned(p['feature_source']);processing=pinned(p['processing_source']);output_length_function(processing)
    need(config.get('sampling_rate',16000)==16000 and config.get('hop_length')==160,'actual16k/hop160 feature configuration required')
    need(config.get('dither',0)==0,'deterministic fixed diagnostic requires actual dither0')
    root=Path(p['output_root']);need(root.is_absolute()and not root.exists()and root.parent.is_dir()and all(not x.is_symlink()for x in(root,*root.parents)),'fresh physical diagnostic output required')
    return p,config,feature,processing


def waveform(length,variant=0):
    import numpy as np
    need(type(length)is int and 400<length<=480000,'bounded synthetic length required')
    x=np.arange(length,dtype=np.float64)/16000
    # 确定性多频波形和非零尾部有意显示反射边界与右补零的区别。
    return (.23*np.sin(2*np.pi*(173+variant)*x)+.11*np.cos(2*np.pi*419*x)+.013).astype(np.float32)


def array_sha(value,dtype='<f4'):
    import numpy as np
    return sha(np.ascontiguousarray(value,dtype=dtype).tobytes())


def compare(a,b):
    import numpy as np
    n=min(a.shape[-1],b.shape[-1]);delta=np.abs(a[...,:n].astype(np.float64)-b[...,:n].astype(np.float64));frames=np.max(delta,axis=0)
    exact=np.flatnonzero(frames!=0);changed=np.flatnonzero(frames>1e-6)
    return dict(common_frames=n,full_valid_shape_equal=a.shape==b.shape,common_prefix_equal=bool(np.array_equal(a[...,:n],b[...,:n])),
        common_prefix_sha_left=array_sha(a[...,:n]),common_prefix_sha_right=array_sha(b[...,:n]),maximum_absolute_difference=float(delta.max()),
        frames_changed_exact=int(len(exact)),frames_changed_above_tolerance=int(len(changed)),difference_tolerance=1e-6,changed_frame_indices=changed.tolist(),
        maximum_difference_mel_frame=[int(x)for x in np.unravel_index(delta.argmax(),delta.shape)],
        region_maximum_absolute_difference={name:float(delta[:,a:b].max())if b>a else 0. for name,a,b in [('first_third',0,n//3),('middle_third',n//3,2*n//3),('last_third',2*n//3,n),('last_five_frames',max(0,n-5),n)]})



def raw_log_probe(extractor,inputs):
    """按已pin官方torch方法逐行复算中间量；与官方最终整数组逐bit互核。"""
    import numpy as np
    import torch
    length=max(len(x)for x in inputs)
    batch=np.stack([np.pad(x,(0,length-len(x)),constant_values=extractor.padding_value)for x in inputs])
    waveform=torch.from_numpy(batch).to('cpu',torch.float32)
    window=torch.hann_window(extractor.n_fft,device='cpu')
    stft=torch.stft(waveform,extractor.n_fft,extractor.hop_length,window=window,return_complex=True)
    magnitudes=stft[...,:-1].abs()**2
    filters=torch.from_numpy(extractor.mel_filters).to('cpu',torch.float32)
    log_spec=torch.clamp(filters.T @ magnitudes,min=1e-10).log10()
    row_max=log_spec.max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)[0]
    final=(torch.maximum(log_spec,row_max-8.)+4.)/4.
    return log_spec.numpy(),final.numpy()


def run(plan_pin,execute=False):
    p,config,feature_raw,processing_raw=validate(plan_pin)
    if not execute:return dict(schema=SCHEMA,checked=True,executed=False,real_audio_read=False,model_weights_loaded=False,GPU_created=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import sys
    blocked=[]
    def no_original_media_or_weights(event,args):
        if event=='open'and args and isinstance(args[0],(str,bytes)):
            name=os.fsdecode(args[0])
            if name.lower().endswith(('.safetensors','.bin','.pt','.wav','.flac','.mp3')):
                blocked.append(name);raise RuntimeError('synthetic diagnostic may not open weights or original media')
    sys.addaudithook(no_original_media_or_weights)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    need(not torch.cuda.is_initialized(),'diagnostic cannot inherit initialized CUDA')
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
    actual=Path(inspect.getsourcefile(WhisperFeatureExtractor)).resolve();need(str(actual)==p['feature_source']['path']and sha(actual.read_bytes())==p['feature_source']['sha256'],'imported official Whisper feature source differs')
    extractor=WhisperFeatureExtractor(**config);need(extractor.hop_length==160 and extractor.sampling_rate==16000,'effective extractor geometry differs')
    output_lengths,function_sha=output_length_function(processing_raw);start=time.monotonic();results=[];saved_arrays={};extra_controls={}
    for length in LENGTHS:
        target=waveform(length);by_batch={};values={};raw_values={}
        for batch in BATCHES:
            inputs=[target]+[waveform(COMPANION,i+1)for i in range(batch-1)]
            encoded=extractor(inputs,sampling_rate=16000,padding=True,truncation=False,return_attention_mask=True,return_tensors='np',device='cpu',do_normalize=False)
            features=np.asarray(encoded['input_features']);mask=np.asarray(encoded['attention_mask']);need(features.shape[0]==batch and mask.shape==(batch,features.shape[-1]),'actual feature/mask geometry differs')
            count=int(mask[0].sum());valid=features[0,:,:count].copy();values[batch]=valid
            raw_log,reconstructed=raw_log_probe(extractor,inputs);need(np.array_equal(features,reconstructed),'intermediate log probe does not exactly reconstruct official full feature array')
            raw_values[batch]=raw_log[0,:,:count].copy();maxpos=np.unravel_index(raw_log[0].argmax(),raw_log[0].shape)
            saved_arrays['n%d_b%d_valid_mel'%(length,batch)]=valid;saved_arrays['n%d_b%d_raw_log_valid'%(length,batch)]=raw_values[batch]
            need(np.isfinite(features).all()and set(np.unique(mask))<={0,1},'invalid actual feature or mask')
            by_batch[str(batch)]=dict(input_PCM_sha256=array_sha(target),padded_PCM_frames=max(len(x)for x in inputs),feature_shape=list(features.shape),mask_shape=list(mask.shape),
                target_feature_mask_sha256=array_sha(mask[0],'<i8'),target_valid_mel_frames=count,target_valid_mel_sha256=array_sha(valid),audio_token_placeholders=int(output_lengths(count)),device='cpu',raw_log_global_max=float(raw_log[0].max()),dynamic_range_log_floor=float(raw_log[0].max()-8.),
                raw_log_global_max_mel_frame=[int(x)for x in maxpos],raw_log_peak_is_inside_valid_mask=bool(maxpos[1]<count),manual_intermediates_reconstruct_official_full_array=True)
        if length==16000:
            same=extractor([target]*4,sampling_rate=16000,padding=True,truncation=False,return_attention_mask=True,return_tensors='np',device='cpu',do_normalize=False)
            bright=extractor([target]+[waveform(COMPANION,i+1)*100 for i in range(3)],sampling_rate=16000,padding=True,truncation=False,return_attention_mask=True,return_tensors='np',device='cpu',do_normalize=False)
            extra_controls=dict(same_length_B1_vs_B4=compare(values[1],np.asarray(same['input_features'])[0,:,:int(np.asarray(same['attention_mask'])[0].sum())]),
                bright_companions_target_B4=compare(values[4],np.asarray(bright['input_features'])[0,:,:int(np.asarray(bright['attention_mask'])[0].sum())]))
        results.append(dict(raw_log_common_prefix_comparisons={'B1_B4':compare(raw_values[1],raw_values[4]),'B1_B8':compare(raw_values[1],raw_values[8])},synthetic_PCM_frames=length,synthetic_PCM_sha256=array_sha(target),batches=by_batch,comparisons={'B1_B4':compare(values[1],values[4]),'B1_B8':compare(values[1],values[8]),'B4_B8':compare(values[4],values[8])}))
    need(not torch.cuda.is_initialized(),'official CPU feature extraction unexpectedly initialized CUDA')
    import resource
    value=dict(schema='annotation-native-feature-padding-diagnostic-result/v1',plan=plan_pin,source_pins={k:p[k]for k in('contract','feature_source','processing_source','config')},
        timing_scope='feature matrix and raw-log probes after framework imports',do_normalize=False,extra_controls=extra_controls,controller_source_sha256=sha(Path(__file__).read_bytes()),official_length_function_AST_sha256=function_sha,config=config,results=results,companion_PCM_frames=COMPANION,
        elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if __import__('sys').platform=='darwin'else 1024),
        blocked_media_or_weight_open_attempts=blocked,numpy_version=np.__version__,torch_version=torch.__version__,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CUDA_initialized=torch.cuda.is_initialized(),model_weights_loaded=False,real_audio_read=False,API_calls=0,GPU_calls=0,production_write=False,human_word_quality_certified=False)
    root=Path(p['output_root']);root.mkdir(mode=0o700);arrays=root/'synthetic-valid-mel-and-raw-log.npz';np.savez(arrays,**saved_arrays);value['saved_synthetic_arrays']=dict(path=str(arrays),sha256=sha(arrays.read_bytes()),bytes=arrays.stat().st_size);raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();(root/'result.json').write_bytes(raw);(root/'operation.json').write_bytes(pinned(plan_pin))
    return dict(output=str(root/'result.json'),sha256=sha(raw),result=value)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--plan',required=True);a.add_argument('--plan-sha',required=True);a.add_argument('--execute',action='store_true');v=a.parse_args()
    print(json.dumps(run({'path':v.plan,'sha256':v.plan_sha},v.execute),ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
