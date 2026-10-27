"""跨引擎协议的合成文件级测试；不执行模型或读取原声。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_engine_comparison as A
import engine_protocol as P
from contracts import canonical_bytes,digest,job_key,model_provenance
from schema import build_record
from test_quality_schema import fixture


def write(path,value,lines=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=b''.join(canonical_bytes(v) for v in value) if lines else canonical_bytes(value)
    path.write_bytes(data)
    return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest()}


def make_case(root, *, purpose='cpu_fixture'):
    """供集成检查复用，所有文件均在调用者的新临时根，无媒体或模型。"""
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    base=list(fixture());base[4][0].update(model_input_frames=32000,resampling_policy='synthetic-polyphase')
    models={k:{'path':v['path'],'files':v['files']} for k,v in base[2]['provenance']['models'].items()}
    runtime={'dtype':'float32','max_new_tokens':12,'window':{'core_seconds':24,'context_seconds':2},
             'ownership':'synthetic-midpoint','boundary_mapping':base[4][0]['time_mapping'],
             'resampling':'synthetic-polyphase','normalization':'synthetic-same',
             'max_batch':1,'execution_policy':'single_window_independent','seed':0,
             'supported_languages':['Chinese','English']}
    contract={'schema':'redimnet2-production-contract/v2','purpose':purpose,'version':'engine-a',
              'runtime':runtime,'models':models,'decoder':{'name':'synthetic-same'},
              'quality_policy':{'strict':True},'alignment':{'mode':'synthetic-same'},
              'source_files':{'synthetic.py':'1'*64},'engine_source_files':{'synthetic.py':'2'*64}}
    contracts=[deepcopy(contract),deepcopy(contract)]
    contracts[1]['version']='engine-b';contracts[1]['runtime']['asr_compute']={'dtype':'bfloat16','attention':'sdpa'}
    contracts[1]['source_files']={'vllm-synthetic.py':'3'*64}
    contracts[1]['engine_versions']={'vllm':'synthetic-vllm','qwen-asr':'synthetic-qwen'}
    fixed=P.controls_from_contract(contract)
    planned={'source':base[0],'source_identity':base[1],
             'windows':[{k:base[4][0][k] for k in P.WINDOW_FIELDS}|{'fixed_text':'哈哈'}]}
    tracks={};records={};producers={};raw_rows={};metrics={}
    for n,c in zip(('baseline','candidate'),contracts):
        folder=root/n
        cref=write(folder/'contract.json',c)
        variables={'backend':'transformers' if n=='baseline' else 'vllm','code_sha256':P.code_identity(c),
                   'precision':'float32' if n=='baseline' else 'bfloat16',
                   'batching':{'policy':'single' if n=='baseline' else 'dynamic','max_requests':1 if n=='baseline' else 16}}
        producer={'schema':P.PRODUCER_SCHEMA,'producer_id':n,'purpose':purpose,'contract':cref,
                  'bundle_sha256':variables['code_sha256'],
                  'fixed_controls':fixed,'variables':variables,'model_API_calls':0,'release_training_labels':False}
        native_spec=None
        if n=='candidate':
            native_spec=native_spec_for(c,fixed,variables,purpose)
            producer['native_asr_producer_sha256']=digest(native_spec)
            producer['native_asr_spec']=write(folder/'adapter-spec.json',native_spec)
        pref=write(folder/'producer.json',producer)
        args=deepcopy(base)
        args[2]={'sha256':cref['sha256'],'version':c['version'],'api_calls':0,
                 'provenance':{'runtime':c['runtime'],'models':model_provenance(c),'code_sha256':variables['code_sha256']}}
        args[6]['job_key']=job_key(args[0],cref['sha256'])
        rec=build_record(*args)
        sref=write(folder/'records'/'s.json',rec)
        marker={'schema':'redimnet2-sidecar-commit/v1','source_id':base[0]['source_id'],
                'job_key':args[6]['job_key'],'contract_sha256':cref['sha256'],
                'source_record_sha256':digest(base[0]),'sidecar_sha256':sref['sha256'],
                'bytes':Path(sref['path']).stat().st_size}
        mref=write(folder/'records'/'s.json.commit.json',marker)
        row={'source_id':base[0]['source_id'],'sidecar':sref,'commit':mref,
             'execution':{'w000':{'asr':'new','alignment':'new'}}}
        if n=='candidate':
            request_identity={'schema':'annotation-asr-request/v1','source_id':base[0]['source_id'],'window_id':'w000',
                'input_pcm_sha256':'7'*64,'model_input_frames':32000,'model_sample_rate':16000,
                'producer_sha256':producer['native_asr_producer_sha256']}
            native={'schema':'annotation-asr-native-result/v1','request_id':digest(request_identity),'request_identity':request_identity,'cohort_sha256':'8'*64,
                'source_id':base[0]['source_id'], 'window_id':'w000','input_pcm_sha256':'7'*64,
                'model_input_frames':32000,'model_sample_rate':16000,'producer_sha256':producer['native_asr_producer_sha256'],
                'status':'completed','raw_text':'哈哈','normalized_text':'哈哈','language':'Chinese',
                'normalization_audit':rec['windows'][0]['asr']['normalization_audit'],
                'synthetic_fixture':True,'model_API_calls':0,'usable_for_training':False,'usable_for_evaluation':False,
                'native':{'request_id':'synthetic-native-0','generated_token_ids':[1,2,3],'eos_observed':True,'first_eos_index':2,'max_new_tokens':12,
                          'finish_reason':'stop','stop_reason':3,'finished':True,'declared_eos_token_ids':[3]}}
            row['native_asr_refs']={'w000':write(folder/'native.json',native)}
        result_ref=write(folder/'results.jsonl',[row],lines=True)
        met={'schema':'annotation-engine-metrics/v1','producer_sha256':pref['sha256'],
             'source_count':1,'wall_seconds':10 if n=='baseline' else 5,
             'time_basis':'whole_run_wall_including_loading_and_publication','model_API_calls':0,'synthetic_fixture':True}
        for stage in ('asr','alignment'):
            for kind in P.KINDS:met[stage+'_'+kind+'_windows']=int(kind=='new')
        metrics_ref=write(folder/'metrics.json',met)
        tracks[n]={'producer':pref,'results':result_ref,'metrics':metrics_ref}
        records[n]=rec;producers[n]=producer;raw_rows[n]=row;metrics[n]=met
    spec={'schema':P.SCHEMA,'purpose':purpose,'mode':'end_to_end','fixed_controls':fixed,
          'allowed_changes':sorted(P.VARIABLE_KEYS),'inputs':write(root/'inputs'/'sources.jsonl',[planned],lines=True),**tracks}
    ref=write(root/'protocol.json',spec)
    return {'root':root,'spec':spec,'ref':ref,'planned':planned,'records':records,'producers':producers,
            'rows':raw_rows,'metrics':metrics,'contracts':dict(zip(('baseline','candidate'),contracts))}


def native_spec_for(contract,fixed,variables,purpose='cpu_fixture'):
    import test_vllm_backend as F
    cfg=F.spec();cfg['purpose']=purpose;cfg['decode'].update(max_tokens=12,seed=0);cfg['eos_token_ids']=[3]
    cfg['model']={'path':contract['models']['asr']['path'],
        'files_sha256':{k:v['sha256'] for k,v in contract['models']['asr']['files'].items()}}
    cfg['preprocessing']['window_policy_sha256']=fixed['window_policy']['sha256']
    cfg['max_batch_rows']=variables['batching']['max_requests'];cfg['source_files']=contract['source_files']
    cfg['qwen_asr_version']=contract['engine_versions']['qwen-asr']
    return cfg


def rewrite_spec(case):
    case['ref']=write(Path(case['ref']['path']),case['spec']);return case['ref']


def rewrite_record(case,track):
    row=case['rows'][track];row['sidecar']=write(Path(row['sidecar']['path']),case['records'][track])
    marker=json.loads(Path(row['commit']['path']).read_text());marker.update(sidecar_sha256=row['sidecar']['sha256'],bytes=Path(row['sidecar']['path']).stat().st_size)
    row['commit']=write(Path(row['commit']['path']),marker)
    case['spec'][track]['results']=write(Path(case['spec'][track]['results']['path']),[row],lines=True)
    rewrite_spec(case)


class EngineComparisonTests(unittest.TestCase):
    def case(self):
        d=tempfile.TemporaryDirectory();self.addCleanup(d.cleanup)
        return make_case(Path(d.name).resolve()/'case')

    def run_case(self,c):return A.audit(c['ref']['path'],c['ref']['sha256'])

    def test_full_file_pair_has_exact_denominator_and_no_GPU_or_human_admission(self):
        r=self.run_case(self.case())
        self.assertEqual(r['source_count'],1);self.assertEqual(r['counts']['planned_windows'],1)
        self.assertIsNone(r['reported_new_execution_wall_ratio']);self.assertIsNone(r['validated_GPU_speedup'])
        self.assertFalse(r['GPU_runtime_verified']);self.assertFalse(r['formal_admission']);self.assertFalse(r['human_quality_verified'])

    def test_cross_engine_code_precision_and_batching_are_declared_not_bitwise_required(self):
        r=self.run_case(self.case());self.assertEqual(set(r['actual_declared_changes']),P.VARIABLE_KEYS)
        self.assertFalse(r['native_cross_engine_bitwise_required'])

    def test_undeclared_variable_change_is_rejected(self):
        c=self.case();c['spec']['allowed_changes']=['precision'];rewrite_spec(c)
        with self.assertRaisesRegex(RuntimeError,'undeclared'):self.run_case(c)

    def test_quality_weights_prompt_and_cap_controls_cannot_be_redeclared(self):
        for key in ('quality_policy','model_weights','prompt_language_policy','generation'):
            c=self.case();p=c['producers']['candidate'];p['fixed_controls']=deepcopy(p['fixed_controls'])
            p['fixed_controls'][key]['value']['changed']=True;p['fixed_controls'][key]['sha256']=digest(p['fixed_controls'][key]['value'])
            c['spec']['candidate']['producer']=write(Path(c['spec']['candidate']['producer']['path']),p);rewrite_spec(c)
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.run_case(c)

    def test_control_content_SHA_and_unknown_variable_refused(self):
        c=self.case();c['spec']['fixed_controls']['window_policy']['sha256']='f'*64;rewrite_spec(c)
        with self.assertRaisesRegex(RuntimeError,'content/SHA'):self.run_case(c)
        c=self.case();c['spec']['allowed_changes'].append('quality_policy');rewrite_spec(c)
        with self.assertRaises(RuntimeError):self.run_case(c)

    def test_token_cap_change_in_actual_saved_output_is_rejected(self):
        c=self.case();c['records']['candidate']['windows'][0]['asr']['native']['max_new_tokens']=11;rewrite_record(c,'candidate')
        with self.assertRaisesRegex(RuntimeError,'token cap'):self.run_case(c)

    def test_input_pcm_drift_is_not_a_model_difference(self):
        c=self.case();c['planned']['windows'][0]['input_pcm_sha256']='e'*64
        c['spec']['inputs']=write(Path(c['spec']['inputs']['path']),[c['planned']],lines=True);rewrite_spec(c)
        with self.assertRaisesRegex(RuntimeError,'PCM/window'):self.run_case(c)

    def test_results_duplicates_or_missing_source_are_not_silently_dropped(self):
        for rows in ([],[1,1]):
            c=self.case();values=[] if not rows else [c['rows']['candidate']]*2
            c['spec']['candidate']['results']=write(Path(c['spec']['candidate']['results']['path']),values,lines=True);rewrite_spec(c)
            with self.subTest(rows=rows),self.assertRaises(RuntimeError):self.run_case(c)

    def test_changed_text_disables_endpoint_delta_instead_of_reporting_zero(self):
        c=self.case();w=c['records']['candidate']['windows'][0]
        # 同一字串的目标变化单独测试比较器，正式文件还会核自有ASR目标。
        w['alignment']['target_text']='不同'
        with self.assertRaisesRegex(RuntimeError,'own ASR'):
            A.compare_window(c['records']['baseline']['windows'][0],w,c['planned']['windows'][0],'end_to_end')
        w['asr']['normalized_text']='不同'
        r=A.compare_window(c['records']['baseline']['windows'][0],w,c['planned']['windows'][0],'end_to_end')
        self.assertIsNone(r['endpoint_delta_ms']);self.assertFalse(r['alignment_targets_equal'])

    def test_fixed_text_mode_requires_declared_same_target(self):
        c=self.case();c['spec']['mode']='fixed_text_alignment';rewrite_spec(c)
        self.assertEqual(self.run_case(c)['mode'],'fixed_text_alignment')
        c['planned']['windows'][0]['fixed_text']='不同';c['spec']['inputs']=write(Path(c['spec']['inputs']['path']),[c['planned']],lines=True);rewrite_spec(c)
        with self.assertRaisesRegex(RuntimeError,'declared identical'):self.run_case(c)

    def test_unknown_native_zero_is_retained_and_not_human_error_rate(self):
        c=self.case()
        # 比较器保存原始几何；零起点本身可以合法，不把机器差异当人审错误。
        w=deepcopy(c['records']['candidate']['windows'][0]);w['alignment']['native_timestamps_ms'][0]=0
        r=A.compare_window(c['records']['baseline']['windows'][0],w,c['planned']['windows'][0],'end_to_end')
        self.assertEqual(r['endpoint_delta_ms']['native_timestamps_ms'],100)
        self.assertEqual(r['alignment_observations']['candidate']['native_timestamps_ms'][0],0)

    def test_commit_corruption_is_rejected_without_repair(self):
        c=self.case();p=Path(c['rows']['candidate']['commit']['path']);p.unlink()
        with self.assertRaises(FileNotFoundError):self.run_case(c)
        self.assertFalse(p.exists())

    def test_metrics_cannot_turn_cache_consumption_into_new_calls(self):
        c=self.case();row=c['rows']['candidate'];row['execution']['w000']['asr']='cache'
        c['spec']['candidate']['results']=write(Path(c['spec']['candidate']['results']['path']),[row],lines=True);rewrite_spec(c)
        with self.assertRaisesRegex(RuntimeError,'accounting'):self.run_case(c)
        m=c['metrics']['candidate'];m['asr_new_windows']=0;m['asr_cache_windows']=1
        c['spec']['candidate']['metrics']=write(Path(c['spec']['candidate']['metrics']['path']),m);rewrite_spec(c)
        self.assertIsNone(self.run_case(c)['reported_new_execution_wall_ratio'])

    def test_missing_vllm_native_request_is_rejected(self):
        c=self.case();c['rows']['candidate']['native_asr_refs']={};c['spec']['candidate']['results']=write(Path(c['spec']['candidate']['results']['path']),[c['rows']['candidate']],lines=True);rewrite_spec(c)
        with self.assertRaisesRegex(RuntimeError,'preserve native'):self.run_case(c)

    def test_native_request_pcm_and_producer_are_bound(self):
        c=self.case();row=c['rows']['candidate'];p=Path(row['native_asr_refs']['w000']['path']);v=json.loads(p.read_text());v['producer_sha256']='0'*64
        v['request_identity']['producer_sha256']='0'*64;v['request_id']=digest(v['request_identity'])
        row['native_asr_refs']['w000']=write(p,v);c['spec']['candidate']['results']=write(Path(c['spec']['candidate']['results']['path']),[row],lines=True);rewrite_spec(c)
        with self.assertRaisesRegex(RuntimeError,'native ASR producer'):self.run_case(c)

    def test_native_stable_request_ID_cannot_be_detached_from_request_identity(self):
        c=self.case();row=c['rows']['candidate'];p=Path(row['native_asr_refs']['w000']['path']);v=json.loads(p.read_text());v['request_id']='not-the-request-hash'
        row['native_asr_refs']['w000']=write(p,v);c['spec']['candidate']['results']=write(Path(c['spec']['candidate']['results']['path']),[row],lines=True);rewrite_spec(c)
        with self.assertRaisesRegex(RuntimeError,'stable request'):self.run_case(c)

    def test_native_internal_request_ID_is_required(self):
        for value in (None, '', '   ', 123):
            with self.subTest(value=value):
                c=self.case();row=c['rows']['candidate'];p=Path(row['native_asr_refs']['w000']['path']);v=json.loads(p.read_text())
                if value is None:v['native'].pop('request_id')
                else:v['native']['request_id']=value
                row['native_asr_refs']['w000']=write(p,v)
                c['spec']['candidate']['results']=write(Path(c['spec']['candidate']['results']['path']),[row],lines=True);rewrite_spec(c)
                with self.assertRaisesRegex(RuntimeError,'internal request ID'):self.run_case(c)

    def test_native_cohort_SHA_is_required(self):
        for value in (None, '', 'g'*64, '8'*63, 123):
            with self.subTest(value=value):
                c=self.case();row=c['rows']['candidate'];p=Path(row['native_asr_refs']['w000']['path']);v=json.loads(p.read_text())
                if value is None:v.pop('cohort_sha256')
                else:v['cohort_sha256']=value
                row['native_asr_refs']['w000']=write(p,v)
                c['spec']['candidate']['results']=write(Path(c['spec']['candidate']['results']['path']),[row],lines=True);rewrite_spec(c)
                with self.assertRaisesRegex(RuntimeError,'cohort SHA'):self.run_case(c)

    def test_native_adapter_fixed_controls_are_read_not_just_hash_claimed(self):
        for field in ('decode','model','window','dtype','language','batch'):
            c=self.case();producer=c['producers']['candidate'];ref=producer['native_asr_spec'];cfg=json.loads(Path(ref['path']).read_text())
            if field=='decode':cfg['decode']['max_tokens']=11
            elif field=='model':cfg['model']['files_sha256']['weights']='0'*64
            elif field=='window':cfg['preprocessing']['window_policy_sha256']='0'*64
            elif field=='dtype':cfg['engine']['dtype']='float32'
            elif field=='language':cfg['force_language']='English'
            else:cfg['max_batch_rows']=2
            producer['native_asr_spec']=write(Path(ref['path']),cfg);producer['native_asr_producer_sha256']=digest(cfg)
            c['spec']['candidate']['producer']=write(Path(c['spec']['candidate']['producer']['path']),producer)
            c['metrics']['candidate']['producer_sha256']=c['spec']['candidate']['producer']['sha256']
            c['spec']['candidate']['metrics']=write(Path(c['spec']['candidate']['metrics']['path']),c['metrics']['candidate']);rewrite_spec(c)
            with self.subTest(field=field),self.assertRaises(RuntimeError):self.run_case(c)

    def test_native_status_cannot_be_laundered_into_successful_sidecar(self):
        c=self.case();row=c['rows']['candidate'];p=Path(row['native_asr_refs']['w000']['path']);v=json.loads(p.read_text());v['status']='unknown'
        row['native_asr_refs']['w000']=write(p,v);c['spec']['candidate']['results']=write(Path(c['spec']['candidate']['results']['path']),[row],lines=True);rewrite_spec(c)
        with self.assertRaisesRegex(RuntimeError,'native output and final ASR'):self.run_case(c)

    def test_native_EOS_contradiction_cannot_be_hidden_by_final_text(self):
        c=self.case();row=c['rows']['candidate'];p=Path(row['native_asr_refs']['w000']['path']);v=json.loads(p.read_text());v['native']['first_eos_index']=0
        row['native_asr_refs']['w000']=write(p,v);c['spec']['candidate']['results']=write(Path(c['spec']['candidate']['results']['path']),[row],lines=True);rewrite_spec(c)
        with self.assertRaisesRegex(RuntimeError,'contradicts'):self.run_case(c)

    def test_native_EOS_set_cannot_change_from_actual_pinned_adapter_spec(self):
        c=self.case();row=c['rows']['candidate'];p=Path(row['native_asr_refs']['w000']['path']);v=json.loads(p.read_text());v['native']['declared_eos_token_ids']=[2,3]
        v['native']['first_eos_index']=1
        row['native_asr_refs']['w000']=write(p,v);c['spec']['candidate']['results']=write(Path(c['spec']['candidate']['results']['path']),[row],lines=True);rewrite_spec(c)
        with self.assertRaisesRegex(RuntimeError,'EOS declaration'):self.run_case(c)

    def test_sidecar_from_another_bundle_cannot_be_pasted_into_producer(self):
        c=self.case();c['records']['candidate']['provenance']['code_sha256']='0'*64
        c['records']['candidate']['contract']['provenance']['code_sha256']='0'*64
        rewrite_record(c,'candidate')
        with self.assertRaisesRegex(RuntimeError,'frozen bundle'):self.run_case(c)

    def test_failed_source_without_windows_remains_in_full_denominator(self):
        c=self.case();r=c['records']['candidate'];args=list(fixture())
        args[1]['decode_status']='decode_failed';args[2]=deepcopy(r['contract']);args[4]=[];args[5]=[]
        args[6]=deepcopy(r['execution']);args[7]=[{'stage':'source_decode','traceback':'synthetic full error preserved'}]
        c['records']['candidate']=build_record(*args)
        c['rows']['candidate'].update(execution={},native_asr_refs={})
        rewrite_record(c,'candidate')
        for stage in ('asr','alignment'):
            for kind in P.KINDS:c['metrics']['candidate'][stage+'_'+kind+'_windows']=0
        c['spec']['candidate']['metrics']=write(Path(c['spec']['candidate']['metrics']['path']),c['metrics']['candidate']);rewrite_spec(c)
        out=self.run_case(c);self.assertEqual(out['candidate_decision_counts'],{'failed':1});self.assertEqual(out['source_count'],1)
        self.assertEqual(out['counts']['missing_windows'],1);self.assertIn('full error',out['rows'][0]['candidate_errors'][0]['traceback'])
        self.assertIsNone(out['maximum_same_target_endpoint_delta_ms'])

    def test_fake_vllm_durable_stage_native_file_and_engine_audit_integrate(self):
        import numpy as np
        import test_vllm_backend as F
        from asr_stage_store import digest as native_digest
        c=self.case();audio=np.zeros(32000,dtype=np.float32);pcm=hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest()
        c['planned']['windows'][0]['input_pcm_sha256']=pcm
        c['spec']['inputs']=write(Path(c['spec']['inputs']['path']),[c['planned']],lines=True)
        for name in ('baseline','candidate'):
            c['records'][name]['windows'][0]['input_pcm_sha256']=pcm
        cfg=native_spec_for(c['contracts']['candidate'],c['spec']['fixed_controls'],c['producers']['candidate']['variables'])
        class Engine(F.FakeEngine):
            def wait_for_completion(self,use_tqdm=False):
                return [{'request_id':'native-0','finished':True,'prompt_token_ids':[1,2],
                    'outputs':[{'index':0,'text':'Chinese|哈哈','token_ids':[1,2,3],'finish_reason':'stop','stop_reason':3}]}]
        request={'source_id':c['planned']['source']['source_id'],'window_id':'w000','audio':audio,
                 'input_pcm_sha256':pcm,'model_input_frames':32000,'model_sample_rate':16000}
        engine=Engine();adapter=F.adapter(c['root']/'stage',engine,cfg);run=adapter.run_batch([request])
        native=run['results'][0]
        self.assertEqual(engine.calls,1);self.assertEqual(run['metrics']['new_model_rows'],1)
        w=c['records']['candidate']['windows'][0]
        w['asr']={k:native[k] for k in ('status','raw_text','normalized_text','language','normalization_audit','native')}
        c['producers']['candidate']['native_asr_producer_sha256']=native_digest(cfg)
        c['spec']['candidate']['producer']=write(Path(c['spec']['candidate']['producer']['path']),c['producers']['candidate'])
        c['metrics']['candidate']['producer_sha256']=c['spec']['candidate']['producer']['sha256']
        c['spec']['candidate']['metrics']=write(Path(c['spec']['candidate']['metrics']['path']),c['metrics']['candidate'])
        c['rows']['candidate']['native_asr_refs']['w000']=write(c['root']/'candidate/native.json',native)
        for name in ('baseline','candidate'):rewrite_record(c,name)
        result=self.run_case(c);self.assertEqual(result['source_count'],1);self.assertFalse(result['GPU_runtime_verified'])
        snapshot=Path(c['rows']['candidate']['native_asr_refs']['w000']['path']).read_bytes()
        restored=adapter.run_batch([{k:v for k,v in request.items() if k!='audio'}])
        self.assertEqual(engine.calls,1);self.assertEqual(restored['metrics']['stage_cache_hits'],1)
        self.assertEqual(restored['results'],run['results']);self.assertEqual(Path(c['rows']['candidate']['native_asr_refs']['w000']['path']).read_bytes(),snapshot)

    def test_previous_published_bytes_and_reference_are_immutable(self):
        c=self.case();row=c['rows']['candidate'];snap={'schema':'annotation-published-before/v1','entries':[{k:row[k] for k in ('source_id','sidecar','commit')}]}
        c['spec']['published_before']=write(c['root']/'prior.json',snap);rewrite_spec(c)
        r=self.run_case(c);self.assertTrue(r['recovery']['published_references_and_bytes_unchanged'])
        p=Path(row['sidecar']['path']);p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(RuntimeError,'SHA'):self.run_case(c)

    def test_report_output_cannot_overlap_input_or_producer_tree(self):
        c=self.case()
        with self.assertRaisesRegex(RuntimeError,'overlaps'):
            A.audit(c['ref']['path'],c['ref']['sha256'],c['root']/'candidate'/'report.json')
        out=c['root'].parent/'review.json';A.audit(c['ref']['path'],c['ref']['sha256'],out)
        with self.assertRaises(RuntimeError):A.audit(c['ref']['path'],c['ref']['sha256'],out)

    def test_symlink_input_and_malformed_JSON_are_rejected(self):
        c=self.case();alias=c['root'].parent/'alias';alias.symlink_to(c['root'],target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError,'symlink'):
            A.audit(alias/'protocol.json',c['ref']['sha256'])
        p=c['root']/'bad.json';raw=b'{"x":1,"x":2}';p.write_bytes(raw)
        with self.assertRaises(ValueError):P.read_ref({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()})

    def test_CPU_fixture_cannot_relabel_metrics_as_model_execution(self):
        c=self.case();m=c['metrics']['candidate'];m['synthetic_fixture']=False
        c['spec']['candidate']['metrics']=write(Path(c['spec']['candidate']['metrics']['path']),m);rewrite_spec(c)
        with self.assertRaisesRegex(RuntimeError,'synthetic'):self.run_case(c)

    def test_cli_writes_new_report_with_full_file_binding(self):
        c=self.case();out=c['root'].parent/'cli-report.json'
        subprocess.run([sys.executable,str(Path(A.__file__)),'--spec',c['ref']['path'],'--spec-sha',c['ref']['sha256'],'--output',str(out)],check=True,capture_output=True)
        self.assertEqual(json.loads(out.read_text())['source_count'],1)


if __name__=='__main__':unittest.main()
