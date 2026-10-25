"""Word-complete units: source audio is immutable; time follows content boundaries."""
from __future__ import annotations
import hashlib,json,math,re,unicodedata
from pathlib import Path

SCHEMA='word-complete-units/v1'
REFERENCE_EVIDENCE={'reference_whole_utterance','reference_timed_segment','human_audited_span'}
CANDIDATE_EVIDENCE='forced_alignment_candidate'

def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def _stat(path):
    s=Path(path).stat();return {'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def _han(c):
    n=ord(c);return c=='〇' or 0x3400<=n<=0x4dbf or 0x4e00<=n<=0x9fff or 0xf900<=n<=0xfaff or 0x20000<=n<=0x2fa1f

def normalize_text(text,language='zh'):
    if not isinstance(text,str):raise ValueError('text must be a string')
    value=unicodedata.normalize('NFKC',text)
    if language=='zh':
        chars=[]
        for c in value:
            if _han(c):chars.append(c)
            elif c.isspace() or unicodedata.category(c).startswith('P'):continue
            else:raise ValueError(f'unknown spoken form in Chinese reference: {c!r}; do not guess numbers/mixed scripts')
        normalized=''.join(chars)
    elif language=='en':
        if any(c.isdigit() or (c.isalpha() and not c.isascii()) for c in value):raise ValueError('unsupported numeric/non-English spoken form')
        words=re.findall(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*",value)
        remainder=re.sub(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*",'',value)
        if any(not c.isspace() and not unicodedata.category(c).startswith('P') for c in remainder):raise ValueError('unsupported English token')
        normalized=' '.join(w.lower().replace('’',"'") for w in words)
    else:raise ValueError('language must be zh or en; mixed-language units require a separate protocol')
    if not normalized:raise ValueError('empty spoken text')
    return normalized

def tokenize_units(text,language='zh'):
    n=normalize_text(text,language);return list(n) if language=='zh' else n.split()
def count_bin(n):
    if isinstance(n,bool) or not isinstance(n,int) or n<1:raise ValueError('positive unit count required')
    return '1' if n==1 else '2-3' if n<=3 else '4-6' if n<=6 else '7+'

def unit_identity(r):
    return {k:r[k] for k in ['audio_sha256','sample_rate','start_sample','end_sample','speaker_id','spoken_text','language','source_utterance_id','evidence','transcript_origin','boundary_review']}

def make_unit(audio_path,speaker_id,text,*,language='zh',start_sample=0,end_sample=None,
              evidence='reference_whole_utterance',transcript_origin='dataset_reference',
              boundary_review=None,source_utterance_id=None,source_session_id=None,
              corpus='unspecified',split='unspecified',provenance=None,alignment=None,vad=None):
    import soundfile as sf
    if speaker_id is None or not str(speaker_id).strip():raise ValueError('speaker_id is required')
    p=Path(audio_path);before=_stat(p);info=sf.info(str(p));checksum=file_sha(p)
    if before!=_stat(p):raise ValueError('source changed while building unit')
    if end_sample is None:end_sample=int(info.frames)
    normalized=normalize_text(text,language);tokens=tokenize_units(normalized,language)
    if boundary_review is None:boundary_review='pending' if evidence==CANDIDATE_EVIDENCE else 'provider_reference' if evidence.startswith('reference_') else 'audited'
    record={'audio_path':str(p.absolute()),'audio_realpath':str(p.resolve()),'audio_sha256':checksum,
      'audio_stat':before,'sample_rate':int(info.samplerate),'source_num_samples':int(info.frames),'source_channels':int(info.channels),
      'start_sample':start_sample,'end_sample':end_sample,'speaker_id':str(speaker_id),'text_original':text,
      'spoken_text':normalized,'language':language,'n_units':len(tokens),'unit_texts':tokens,
      'count_bin':count_bin(len(tokens)),'counting_policy':'han_characters_no_punctuation_v1' if language=='zh' else 'english_words_v1',
      'evidence':evidence,'transcript_origin':transcript_origin,'boundary_review':boundary_review,
      'source_utterance_id':str(source_utterance_id) if source_utterance_id else checksum,
      'source_session_id':str(source_session_id) if source_session_id is not None else None,
      'corpus':corpus,'split':split,'provenance':provenance or {},'alignment':alignment,'vad':vad}
    record['duration_sec']=(end_sample-start_sample)/info.samplerate
    record['n_samples_16k']=math.ceil((end_sample-start_sample)*16000/info.samplerate)
    record['unit_id']=digest(unit_identity(record))
    return validate_unit(record,allow_candidates=True)

def validate_unit(record,allow_candidates=False,verify_audio=False,*,allow_exploratory=None):
    if allow_exploratory is not None:allow_candidates=allow_exploratory
    r=dict(record)
    for name in ['unit_id','speaker_id','audio_path','audio_realpath','audio_sha256','spoken_text','source_utterance_id']:
        if not isinstance(r.get(name),str) or not r[name]:raise ValueError('missing '+name)
    if not re.fullmatch('[0-9a-f]{64}',r['audio_sha256']):raise ValueError('bad audio SHA')
    for name in ['sample_rate','source_num_samples','source_channels','start_sample','end_sample','n_units','n_samples_16k']:
        if isinstance(r.get(name),bool) or not isinstance(r.get(name),int):raise ValueError('integer required: '+name)
    if r['sample_rate']<=0 or r['source_channels']<1 or not 0<=r['start_sample']<r['end_sample']<=r['source_num_samples']:raise ValueError('invalid source interval/header')
    tokens=tokenize_units(r['spoken_text'],r['language'])
    policy='han_characters_no_punctuation_v1' if r['language']=='zh' else 'english_words_v1'
    if r.get('counting_policy')!=policy or normalize_text(r.get('text_original'),r['language'])!=r['spoken_text']:raise ValueError('original text/counting policy differs')
    if r['spoken_text']!=normalize_text(r['spoken_text'],r['language']) or r['n_units']!=len(tokens) or r['unit_texts']!=tokens or r['count_bin']!=count_bin(len(tokens)):raise ValueError('text/unit count differs')
    expected=(r['end_sample']-r['start_sample'])/r['sample_rate']
    if not math.isfinite(r.get('duration_sec',float('nan'))) or not math.isclose(r['duration_sec'],expected,abs_tol=1e-12):raise ValueError('duration differs')
    if r['n_samples_16k']!=math.ceil((r['end_sample']-r['start_sample'])*16000/r['sample_rate']):raise ValueError('resampled sample count differs')
    evidence=r.get('evidence')
    if evidence==CANDIDATE_EVIDENCE:
        if not allow_candidates:raise ValueError('alignment candidate is not a reference word-complete unit')
        if r.get('boundary_review')!='pending' or not isinstance(r.get('alignment'),dict) or not r['alignment']:raise ValueError('candidate must retain pending alignment provenance')
    elif evidence not in REFERENCE_EVIDENCE:raise ValueError('unknown completeness evidence')
    elif evidence.startswith('reference_'):
        if r.get('transcript_origin') not in {'dataset_reference','user_reference','human_transcript'} or r.get('boundary_review')!='provider_reference':raise ValueError('reference evidence/origin differs')
        if evidence=='reference_whole_utterance' and (r['start_sample']!=0 or r['end_sample']!=r['source_num_samples']):raise ValueError('whole reference utterance must preserve the whole original file')
        if evidence=='reference_timed_segment' and not r.get('provenance'):raise ValueError('timed reference needs original annotation provenance')
    elif r.get('boundary_review')!='audited' or not r.get('provenance',{}).get('review_note'):raise ValueError('human audited unit needs its actual review note')
    if r['unit_id']!=digest(unit_identity(r)):raise ValueError('unit identity differs')
    if verify_audio:
        import soundfile as sf
        p=Path(r['audio_path'])
        if str(p.resolve())!=r['audio_realpath'] or _stat(p)!=r['audio_stat'] or file_sha(p)!=r['audio_sha256']:raise ValueError('source audio content/stat changed')
        i=sf.info(str(p))
        if (i.samplerate,i.frames,i.channels)!=(r['sample_rate'],r['source_num_samples'],r['source_channels']):raise ValueError('source audio header changed')
    return r

validate_record=validate_unit

def seal_dataset(records,metadata=None):
    records=[validate_unit(r,allow_candidates=True) for r in records]
    if not records or len({r['unit_id'] for r in records})!=len(records):raise ValueError('empty/duplicate unit dataset')
    body={'schema':SCHEMA,'records':records,'speaker_map':{s:i for i,s in enumerate(sorted({r['speaker_id'] for r in records}))},'metadata':metadata or {}}
    return dict(body,dataset_id=digest(body))

def write_dataset(path,records,metadata=None):
    data=seal_dataset(records,metadata);p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');return data
write_manifest=write_dataset

def load_dataset(path,verify=True,allow_candidates=False):
    obj=json.loads(Path(path).read_text());body={k:v for k,v in obj.items() if k!='dataset_id'}
    if obj.get('schema')!=SCHEMA or obj.get('dataset_id')!=digest(body):raise ValueError('word dataset identity differs')
    if not obj['records'] or len({r['unit_id'] for r in obj['records']})!=len(obj['records']):raise ValueError('empty/duplicate units')
    expected={s:i for i,s in enumerate(sorted({r['speaker_id'] for r in obj['records']}))}
    if obj['speaker_map']!=expected:raise ValueError('speaker map differs')
    for r in obj['records']:validate_unit(r,allow_candidates,verify_audio=verify)
    return obj
load_manifest=load_dataset

def validate_training_dataset(dataset):
    """重分配split不能清除明确的留出、保护或禁止训练证据。

    只识别已经写明的用途字段；不把VoxCeleb的dev目录名猜成评测集，
    也不把显式候选开关解释为覆盖来源禁训资格。
    """
    metadata = dataset.get('metadata', {})
    if not isinstance(metadata, dict) or metadata.get('split') != 'train' or not dataset.get('records'):
        raise ValueError('only train split is allowed')
    # Registered consumer semantics, matching quality_axes.heldout_source.
    # A source label dev is explicit heldout evidence; a pathname containing dev is not.
    heldout = {'test', 'dev', 'eval', 'evaluation', 'validation', 'val', 'valid', 'dev-cal',
               'calibration', 'protected_regression', 'provider_evaluation', 'provider_test', 'provider_validation'}
    heldout_roles = heldout | {'regression'}
    for record in dataset['records']:
        if record.get('split') != 'train':
            raise ValueError('only train split is allowed')
        provenance = record.get('provenance', {})
        if not isinstance(provenance, dict):
            raise ValueError('source provenance must be an object')
        scopes = [metadata, record, provenance]
        for parent in (metadata, provenance):
            for field in ('source', 'source_original'):
                if field in parent:
                    if not isinstance(parent[field], dict):
                        raise ValueError('original source scope must be an object: ' + field)
                    scopes.append(parent[field])
        for scope in scopes:
            if 'split' in scope and (not isinstance(scope['split'], str) or scope['split'] not in {'train', 'provider_train'}):
                raise ValueError('original source split cannot be relabeled as train')
            if 'usable_for_training' in scope and scope['usable_for_training'] is not True:
                raise ValueError('source explicitly lacks training eligibility')
            protection = scope.get('protection', {})
            if not isinstance(protection, dict):
                raise ValueError('source protection must be an object')
            if 'training_forbidden' in protection and protection['training_forbidden'] is not False:
                raise ValueError('protected source cannot be used for training')
            original = scope.get('original_splits', [])
            if 'original_split' in scope:
                original = [*original, scope['original_split']] if isinstance(original, list) else None
            if not isinstance(original, list) or any(not isinstance(x, str) for x in original):
                raise ValueError('original splits must retain string labels')
            if any(x.casefold() in heldout for x in original):
                raise ValueError('original heldout source cannot be relabeled as train')
            roles = scope.get('roles', [])
            if not isinstance(roles, list) or any(not isinstance(x, str) for x in roles):
                raise ValueError('source roles must be a list')
            for raw in roles:
                head, separator, tail = raw.casefold().partition(':')
                if ((head in heldout_roles and (not separator or tail))
                        or (head == 'supplemental' and tail in heldout)):
                    raise ValueError('heldout source role cannot be relabeled as train')
    return dataset


def read_unit(record,*,verify_content=False,return_info=False):
    import numpy as np,soundfile as sf
    r=validate_unit(record,allow_candidates=True,verify_audio=verify_content);p=Path(r['audio_path'])
    if _stat(p)!=r['audio_stat'] or str(p.resolve())!=r['audio_realpath']:raise ValueError('source changed before read')
    wave,rate=sf.read(str(p),start=r['start_sample'],frames=r['end_sample']-r['start_sample'],dtype='float32',always_2d=True)
    if rate!=r['sample_rate'] or wave.shape!=(r['end_sample']-r['start_sample'],r['source_channels']) or not np.isfinite(wave).all():raise ValueError('incomplete/nonfinite source unit')
    if _stat(p)!=r['audio_stat']:raise ValueError('source changed during read')
    mono=wave.mean(axis=1,dtype=np.float32)
    if rate!=16000:
        from scipy.signal import resample_poly
        d=math.gcd(rate,16000);mono=resample_poly(mono,16000//d,rate//d).astype(np.float32)
    if len(mono)!=r['n_samples_16k'] or not np.isfinite(mono).all():raise ValueError('unit resampling length/nonfinite')
    info={'source_samples':len(wave),'output_samples':len(mono),'source_rate':rate,'output_rate':16000,'waveform_padding':False,'waveform_repeat':False,'vad_concatenation':False}
    return (mono,info) if return_info else mono

def protect_output(path):
    p=Path(path).resolve();repo=Path(__file__).resolve().parents[1]
    roots=[repo,Path('/root/code/github_repos/redimnet2-fork'),Path('/root/experiments/redimnet2/b6-mix-official-20261019'),Path('/root/group-shared/voiceprint/data'),Path('/shared/group-shared/voiceprint/data')]
    for q in roots:
        q=q.resolve()
        if p==q or q in p.parents or p in q.parents:raise ValueError('protected output: '+str(p))
    return p
