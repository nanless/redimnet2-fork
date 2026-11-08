"""每条沿官方B1取特征和prompt，再在特征层合批；不加载模型。

PCM长度相同不能证明按最长音频补零后提取的mel相同。本协议逐条调用
原官方processor，保留原生有效mel、feature mask及prompt；只补特征尾和
文本左侧。generate入口再从真实张量读回SHA，不能仅信代理自述。
"""
from __future__ import annotations
import copy
import hashlib
import inspect
import json
from pathlib import Path

POLICY='official_b1_features_then_pad/v1'
DECLARATION={'policy':POLICY}
SOURCE_ROLES=('whisper_feature_extractor','sequence_feature_extractor','batch_feature','processor_mixin','audio_utils','qwen_processor','sdk_normalizer')


def digest(value):return hashlib.sha256((json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()


def file_ref(path):
    p=Path(path).resolve();need(p.is_file()and p.stat().st_size<=4*1024**2,'bounded preprocessing source/config required')
    before=p.stat();raw=p.read_bytes();after=p.stat()
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
         (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)and len(raw)==before.st_size,'preprocessing source changed during hash')
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest())


def source_objects():
    """准备/实际活体核验才导入已装类；不构造模型或读取权重。"""
    from transformers import WhisperFeatureExtractor
    from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.processing_utils import ProcessorMixin
    from transformers.audio_utils import mel_filter_bank
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
    from qwen_asr.inference.utils import normalize_audio_input
    return dict(whisper_feature_extractor=WhisperFeatureExtractor,sequence_feature_extractor=SequenceFeatureExtractor,
        batch_feature=BatchFeature,processor_mixin=ProcessorMixin,audio_utils=mel_filter_bank,
        qwen_processor=Qwen3ASRProcessor,sdk_normalizer=normalize_audio_input)


def collect_b1_feature_sources(contract):
    value=dict(schema='annotation-official-b1-feature-sources/v1',files={role:file_ref(inspect.getsourcefile(obj))for role,obj in source_objects().items()},
        processor_config=file_ref(Path(contract['models']['asr']['path'])/'preprocessor_config.json'))
    validate_source_bindings(dict(contract,asr_preprocessing_sources=value));return value


def validate_source_bindings(contract,*,verify_files=True):
    """合同纯CPU校验：七个明确依赖及模型配置；不扫描包、不导入模型。"""
    value=contract.get('asr_preprocessing_sources')
    need(isinstance(value,dict)and set(value)=={'schema','files','processor_config'}and value['schema']=='annotation-official-b1-feature-sources/v1'
         and set(value['files'])==set(SOURCE_ROLES),'explicit seven preprocessing sources required')
    for ref in [*value['files'].values(),value['processor_config']]:
        need(isinstance(ref,dict)and set(ref)=={'path','sha256'}and isinstance(ref['path'],str)and '\x00'not in ref['path']and Path(ref['path']).is_absolute()and '..'not in Path(ref['path']).parts
             and isinstance(ref['sha256'],str)and len(ref['sha256'])==64 and all(v in '0123456789abcdef'for v in ref['sha256']),
             'explicit preprocessing source/config path+SHA required')
        if verify_files:need(file_ref(ref['path'])==ref,'preprocessing source/config bytes differ')
    for role in ('qwen_processor','sdk_normalizer'):
        ref=value['files'][role];pins=contract['source_files']
        need(pins.get(ref['path'])==ref['sha256'],'preprocessing Qwen source does not match original pinned SDK source')
    ref=value['processor_config'];model=contract['models']['asr'];expected=model['files'].get('preprocessor_config.json',{})
    need(Path(ref['path']).name=='preprocessor_config.json'and expected.get('sha256')==ref['sha256'],
         'preprocessing config differs from actual downloaded model files')
    if verify_files:
        need(Path(ref['path'])==(Path(model['path'])/'preprocessor_config.json').resolve(),'actual preprocessing config path differs from model root')
        config=json.loads(Path(ref['path']).read_bytes());need(isinstance(config,dict),'processor config must be JSON object')
    return value


def audio_placeholder_length(feature_frames):
    """当前已pin官方processor/audio encoder完全相同的输出长度整数公式。"""
    need(type(feature_frames)is int and feature_frames>0,'positive B1 feature length required')
    leave=feature_frames%100;features=(leave-1)//2+1
    return ((features-1)//2+1-1)//2+1+(feature_frames//100)*13


def need(ok,message):
    if not ok:raise ValueError(message)


def pcm_sha(wave):
    import numpy as np
    need(isinstance(wave,np.ndarray)and wave.dtype==np.float32 and wave.ndim==1 and len(wave)>0 and np.isfinite(wave).all(),
         'explicit finite mono float32 PCM required')
    return hashlib.sha256(np.ascontiguousarray(wave,dtype='<f4').tobytes()).hexdigest()


def feature_sha(value):
    import numpy as np
    data=value.detach().cpu().contiguous().numpy()
    need(data.dtype==np.float32 and np.isfinite(data).all(),'B1 feature proof requires finite float32 tensors')
    return hashlib.sha256(data.astype('<f4',copy=False).tobytes(order='C')).hexdigest()


class B1FeatureBatchProcessor:
    """透明转发非call接口；固定一次SDK processor调用对应一个完整cohort。"""
    def __init__(self,processor,*,torch_module,normalize_audio,max_requests,batch_feature_type,contract=None,synthetic_fixture=False):
        need(type(max_requests)is int and max_requests in (1,4,8),'declared native width required')
        self.raw=processor;self.torch=torch_module;self.normalizer=normalize_audio;self.maximum=max_requests;self.batch_feature_type=batch_feature_type
        self.expected=None;self.capture=None;self.calls=0;self.observations=0
        need(type(synthetic_fixture)is bool,'explicit synthetic fixture flag required')
        self.contract=contract;self.synthetic=synthetic_fixture;self.source_observation=None
        self._source_stats=None;self._live_identity=None;self._live_first_call_checked=False;self._extractor_config=None
        need(synthetic_fixture or contract is not None,'real preprocessing requires pinned source/config contract')
        need(getattr(processor.tokenizer,'padding_side',None)=='left','official tokenizer must retain explicit left padding')
        self.pad_id=getattr(processor.tokenizer,'pad_token_id',None)
        need(type(self.pad_id)is int and self.pad_id>=0,'actual tokenizer pad token required')
        self.audio_id=processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
        need(type(self.audio_id)is int and self.audio_id>=0,'actual audio placeholder token required')
        self.verify_live_sources(force=True)

    def __getattr__(self,name):return getattr(self.raw,name)

    def verify_live_sources(self,*,force=False,first_call=False):
        if self.synthetic:return
        expected=validate_source_bindings(self.contract,verify_files=False)
        live=(type(self.raw),type(self.raw.feature_extractor),self.batch_feature_type,self.normalizer,
              getattr(self.raw._merge_kwargs,'__func__',self.raw._merge_kwargs),
              getattr(self.raw.feature_extractor.pad,'__func__',self.raw.feature_extractor.pad))
        identity=tuple(id(obj)for obj in live)
        def stamp(path):
            s=Path(path).stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if self._source_stats is not None and not force and not first_call:
            need(identity==self._live_identity and self.raw.feature_extractor.to_dict()==self._extractor_config,
                 'live preprocessing class/function/config changed between cohorts')
            need(all(stamp(p)==v for p,v in self._source_stats.items()),'pinned preprocessing file identity changed; stop and retain evidence')
            return
        objects=source_objects()
        objects.update(whisper_feature_extractor=type(self.raw.feature_extractor),qwen_processor=type(self.raw),
                       batch_feature=self.batch_feature_type,sdk_normalizer=self.normalizer)
        observed={role:file_ref(inspect.getsourcefile(obj))for role,obj in objects.items()}
        need(observed==expected['files'],'actually imported preprocessing classes/functions differ from pins')
        need(str(Path(inspect.getsourcefile(self.raw._merge_kwargs)).resolve())==expected['files']['processor_mixin']['path']and
             str(Path(inspect.getsourcefile(self.raw.feature_extractor.pad)).resolve())==expected['files']['sequence_feature_extractor']['path'],
             'actual processor merge/audio pad method differs from pinned source')
        need(file_ref(expected['processor_config']['path'])==expected['processor_config'],'actual preprocessing config bytes differ')
        config=json.loads(Path(expected['processor_config']['path']).read_bytes());actual=self.raw.feature_extractor.to_dict()
        for key,value in config.items():
            if key in {'processor_class','feature_extractor_type'}:continue
            need(actual.get(key)==value,'actual feature extractor config differs: '+key)
        need(self.raw.feature_extractor.sampling_rate==16000,'actual feature extractor sample rate differs')
        self.source_observation=dict(schema=expected['schema'],files=observed,processor_config=expected['processor_config'],
            actual_feature_extractor_config=actual,verified_from_live_objects=True)
        self._source_stats={ref['path']:stamp(ref['path'])for ref in [*expected['files'].values(),expected['processor_config']]}
        self._live_identity=identity;self._extractor_config=copy.deepcopy(actual)
        if first_call:self._live_first_call_checked=True

    def begin_cohort(self,rows):
        need(isinstance(rows,list)and 1<=len(rows)<=self.maximum,'one explicit finite original cohort required')
        expected=[]
        for row in rows:
            wave=row['audio'];raw_sha=pcm_sha(wave);window=row['window']
            need(raw_sha==window['input_pcm_sha256']and len(wave)==window['model_input_frames']and window['model_sample_rate']==16000,
                 'cohort PCM differs from original window')
            normalized=self.normalizer((wave,16000))
            need(len(normalized)==len(wave),'official SDK normalization unexpectedly changed model input length')
            expected.append(dict(row_id=row['row_id'],original_input_pcm_sha256=raw_sha,sdk_input_pcm_sha256=pcm_sha(normalized),frames=len(wave)))
        need(len({v['row_id']for v in expected})==len(expected),'duplicate cohort row IDs')
        self.expected=expected;self.capture=None;self.calls=0;self.observations=0

    def __call__(self,*,text=None,audio=None,**kwargs):
        torch=self.torch;self.calls+=1
        self.verify_live_sources(first_call=not self._live_first_call_checked)
        need(self.expected is not None and self.calls==1,'SDK split/repeated processor call refused')
        need(kwargs==dict(return_tensors='pt',padding=True),'official B1 processor kwargs changed')
        need(isinstance(text,list)and isinstance(audio,list)and len(text)==len(audio)==len(self.expected),'SDK source/text row mapping changed')
        need(self.raw.tokenizer.padding_side=='left'and self.raw.tokenizer.pad_token_id==self.pad_id,'tokenizer padding changed')
        singles=[];proofs=[]
        for i,(prompt,wave,expected)in enumerate(zip(text,audio,self.expected)):
            need(pcm_sha(wave)==expected['sdk_input_pcm_sha256']and len(wave)==expected['frames'],'actual SDK normalized PCM/order differs')
            single=self.raw(text=[prompt],audio=[wave],**kwargs)
            need(set(single)=={'input_ids','attention_mask','input_features','feature_attention_mask'},'unknown official processor tensor fields')
            ids,mask,features,fmask=(single[k]for k in('input_ids','attention_mask','input_features','feature_attention_mask'))
            need(all(isinstance(v,torch.Tensor)and v.device.type=='cpu'for v in(ids,mask,features,fmask)),
                 'official B1 processor must produce actual CPU tensors')
            need(ids.ndim==mask.ndim==fmask.ndim==2 and ids.shape[0]==mask.shape[0]==fmask.shape[0]==1
                 and tuple(ids.shape)==tuple(mask.shape)and features.ndim==3 and features.shape[0]==1
                 and features.shape[2]==fmask.shape[1]and features.dtype==torch.float32,'B1 processor tensor geometry changed')
            need(ids.dtype==mask.dtype==torch.int64 and fmask.dtype in(torch.int32,torch.int64)and bool((mask==1).all())and bool((fmask==1).all()),
                 'single B1 tensor unexpectedly has padding, mask holes or dtype changes')
            pi,fi=int(ids.shape[1]),int(features.shape[2]);need(pi>0 and fi>0,'empty B1 feature/prompt refused')
            prompt_ids=ids[0].tolist();placeholder_count=sum(t==self.audio_id for t in prompt_ids)
            need(placeholder_count==audio_placeholder_length(fi),'B1 audio placeholders do not bind original valid feature length')
            proofs.append(dict(schema='annotation-official-b1-feature-proof/v1',policy=POLICY,**expected,position=i,
                feature_shape=[int(features.shape[1]),fi],feature_dtype='float32',feature_valid_sha256=feature_sha(features[0]),
                valid_feature_frames=fi,feature_attention_mask=fmask[0].tolist(),baseline_input_token_ids=prompt_ids,
                feature_attention_mask_dtype=str(fmask.dtype).split('.')[-1],
                baseline_nonpad_prompt_tokens=pi,audio_token_id=self.audio_id,audio_placeholder_count=placeholder_count,
                official_processor_rows=1,official_processor_kwargs=copy.deepcopy(kwargs),
                source_bindings_sha256=None if self.synthetic else digest(self.contract['asr_preprocessing_sources']),
                source_bindings_verified=not self.synthetic,
                SDK_normalization_changed_PCM=expected['original_input_pcm_sha256']!=expected['sdk_input_pcm_sha256']))
            singles.append(single)
        max_prompt=max(int(s['input_ids'].shape[1])for s in singles);max_features=max(int(s['input_features'].shape[2])for s in singles)
        data={key:[]for key in('input_ids','attention_mask','input_features','feature_attention_mask')}
        for s,p in zip(singles,proofs):
            left=max_prompt-p['baseline_nonpad_prompt_tokens'];right=max_features-p['valid_feature_frames']
            data['input_ids'].append(torch.nn.functional.pad(s['input_ids'],(left,0),value=self.pad_id))
            data['attention_mask'].append(torch.nn.functional.pad(s['attention_mask'],(left,0),value=0))
            data['input_features'].append(torch.nn.functional.pad(s['input_features'],(0,right),value=0.))
            data['feature_attention_mask'].append(torch.nn.functional.pad(s['feature_attention_mask'],(0,right),value=0))
        result=self.batch_feature_type({key:torch.cat(values,dim=0)for key,values in data.items()})
        self.capture=proofs;return result

    def observe_generate(self,kwargs):
        """从实际 generate 入参读回valid mel/mask/prompt，不信构造时缓存摘要。"""
        torch=self.torch;self.observations+=1
        need(self.calls==1 and self.observations==1 and self.capture is not None,'one actual generate must follow exactly one SDK preprocessing call')
        features=kwargs.get('input_features');fm=kwargs.get('feature_attention_mask');ids=kwargs.get('input_ids');mask=kwargs.get('attention_mask')
        need(all(isinstance(v,torch.Tensor)for v in(features,fm,ids,mask)),'actual generate preprocessing tensors missing')
        n=len(self.capture)
        need(features.ndim==3 and ids.ndim==fm.ndim==mask.ndim==2 and features.shape[0]==ids.shape[0]==fm.shape[0]==mask.shape[0]==n
             and tuple(ids.shape)==tuple(mask.shape)and features.shape[2]==fm.shape[1]and features.dtype==torch.float32
             and ids.dtype==mask.dtype==torch.int64 and fm.dtype in(torch.int32,torch.int64),'actual generate feature/mask/prompt shape or dtype changed')
        result=[]
        for i,baseline in enumerate(self.capture):
            fi,pi=baseline['valid_feature_frames'],baseline['baseline_nonpad_prompt_tokens'];mel=features[i,:,:fi]
            observed_fmask=fm[i].detach().cpu().tolist();observed_mask=mask[i].detach().cpu().tolist();observed_ids=ids[i].detach().cpu().tolist()
            need(int(features.shape[1])==baseline['feature_shape'][0]and feature_sha(mel)==baseline['feature_valid_sha256'],
                 'actual valid mel differs from independent official B1 features')
            need(observed_fmask==[1]*fi+[0]*(int(fm.shape[1])-fi)and str(fm.dtype).split('.')[-1]==baseline['feature_attention_mask_dtype'],
                 'actual feature mask/valid frame count/dtype differs from B1')
            need(observed_mask==[0]*(int(mask.shape[1])-pi)+[1]*pi
                 and observed_ids[-pi:]==baseline['baseline_input_token_ids']
                 and observed_ids[:-pi]==[self.pad_id]*(int(ids.shape[1])-pi),'actual nonpad prompt or left padding differs from B1')
            need(bool((features[i,:,fi:]==0).all()),'actual feature padding is not deterministic zero')
            proof=copy.deepcopy(baseline);proof.update(actual_generate_valid_feature_sha256=feature_sha(mel),
                actual_generate_feature_mask=observed_fmask,actual_generate_feature_padded_frames=int(fm.shape[1]),
                actual_generate_feature_shape=[int(features.shape[1]),int(features.shape[2])],
                actual_generate_feature_dtype=str(features.dtype).split('.')[-1],actual_generate_feature_padding_is_zero=True,
                actual_generate_feature_mask_dtype=str(fm.dtype).split('.')[-1],
                actual_generate_nonpad_prompt_token_ids=observed_ids[-pi:],actual_generate_audio_placeholder_count=sum(t==self.audio_id for t in observed_ids[-pi:]),
                actual_generate_rows=n,actual_generate_row_index=i,verified_from_actual_generate_tensors=True,
                complete_word_boundary_verified=False,human_gold=False)
            result.append(proof)
        return result
