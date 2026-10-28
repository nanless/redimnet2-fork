"""合成文件核原生批次审计，保留全部分母与错误，不使用模型或数据库。"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(HERE),str(HERE/'tests')]
import audit_native_transformers_batch as A
import native_transformers_batch as S
from contracts import canonical_bytes,digest,model_provenance
from normalization import normalize_with_audit
from test_maintenance_entries import external_contract
from test_audit_speed_comparison import pair,track as old_track


def contract(width=4):
    value=external_contract();value['purpose']='cpu_fixture';value['version']='native-test'
    value['runtime'].update(execution_policy=S.POLICY,asr_compute={'dtype':'float32','attention':'sdpa'},
        asr_batching={'policy':'fixed_cohort_until_durable/v1','max_requests':width,'ordering':'frames_then_row_id_within_source_group/v1'})
    return value


def add_feature_protocol(value):
    value['runtime']['asr_preprocessing']={'policy':'official_b1_features_then_pad/v1'}
    refs={role:dict(path='/SYNTHETIC_CPU_SOURCE/'+role+'.py',sha256='c'*64)for role in A.PREPROCESSING_SOURCE_ROLES}
    for role in ('qwen_processor','sdk_normalizer'):value['source_files'][refs[role]['path']]=refs[role]['sha256']
    value['models']['asr']['files']['preprocessor_config.json']={'bytes':2,'sha256':'b'*64}
    value['asr_preprocessing_sources']=dict(schema='annotation-official-b1-feature-sources/v1',files=refs,
        processor_config=dict(path=value['models']['asr']['path']+'/preprocessor_config.json',sha256='b'*64))
    return value


def feature_proof(native,row,contract):
    """显式CPU合成收据；真正tensor读取另由适配器测试覆盖，不冒充GPU。"""
    active=[v for v,m in zip(native['input_token_ids'],native['input_attention_mask'])if m]
    return dict(schema='annotation-official-b1-feature-proof/v1',policy='official_b1_features_then_pad/v1',
        row_id=row['row_id'],original_input_pcm_sha256=row['input_pcm_sha256'],sdk_input_pcm_sha256=row['input_pcm_sha256'],
        frames=row['model_input_frames'],position=native['actual_generate_row_index'],feature_shape=[2,4],feature_dtype='float32',
        feature_valid_sha256='a'*64,valid_feature_frames=4,feature_attention_mask=[1]*4,
        feature_attention_mask_dtype='int32',actual_generate_feature_mask_dtype='int32',
        baseline_input_token_ids=active,baseline_nonpad_prompt_tokens=len(active),audio_token_id=active[0],audio_placeholder_count=active.count(active[0]),
        official_processor_rows=1,official_processor_kwargs=dict(return_tensors='pt',padding=True),SDK_normalization_changed_PCM=False,
        actual_generate_valid_feature_sha256='a'*64,actual_generate_feature_mask=[1]*4+[0]*2,
        actual_generate_feature_padded_frames=6,actual_generate_feature_shape=[2,6],actual_generate_feature_dtype='float32',
        actual_generate_feature_padding_is_zero=True,actual_generate_nonpad_prompt_token_ids=active,
        actual_generate_audio_placeholder_count=active.count(active[0]),actual_generate_rows=native['actual_generate_rows'],
        actual_generate_row_index=native['actual_generate_row_index'],verified_from_actual_generate_tensors=True,
        source_bindings_sha256=digest(contract['asr_preprocessing_sources']),source_bindings_verified=True)


class NativeAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.c=contract()
        if getattr(self,'FEATURE_PROTOCOL',False):add_feature_protocol(self.c)
        self.contract_sha=digest(self.c);self.bundle_sha='d'*64;self.tree=self.root/self.c['version']
        model_sha=digest(model_provenance(self.c)['asr'])
        producer=digest(dict(contract_sha256=self.contract_sha,bundle_sha256=self.bundle_sha,model_sha256=model_sha,
            execution_policy=S.POLICY,asr_batching=self.c['runtime']['asr_batching']))
        self.store=S.NativeTransformersBatchStore(self.tree/'.native_torch_cohorts'/'claim'/'source-group',producer,
            synthetic_fixture=True,contract_sha256=self.contract_sha,model_sha256=model_sha,max_batch_rows=4)
        self.rows=[]
        for i in range(4):
            source={'source_id':'source'+str(i),'relative_path':str(i)+'.wav'};wave=np.full(160+i,i/10,dtype=np.float32)
            self.rows.append(dict(row_id=source['source_id']+':w',source_id=source['source_id'],window_id='w',
                source_record_sha256=digest(source),input_pcm_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                model_input_frames=len(wave),model_sample_rate=16000,audio=wave,context={'source':source}))
        self.outputs=self.store.run_batch(self.rows,infer_batch=self.infer)
        self.records=[{'source':row['context']['source']}for row in self.rows]
        self.windows=[dict(window_id='w',input_pcm_sha256=row['input_pcm_sha256'],model_input_frames=row['model_input_frames'],
                           model_sample_rate=16000,asr=self.outputs[row['row_id']])for row in self.rows]

    def infer(self,rows):
        ids=[r['row_id']for r in rows];outputs={}
        for i,row in enumerate(rows):
            native=dict(generated_token_ids=[11,2,2],effective_generated_token_ids=[11,2],effective_generated_token_count=2,
                eos_observed=True,first_eos_index=1,eos_token_ids=[2],padding_token_id=2,raw_returned_token_count=3,
                post_eos_token_ids=[2],input_padded_tokens=3,input_attention_mask=[0,1,1],input_nonpad_tokens=2,
                input_token_ids=[0,8,9],
                actual_generate_rows=len(rows),actual_generate_row_index=i,max_new_tokens=self.c['runtime']['max_new_tokens'],
                model_training=False,inference_mode=True,grad_enabled=False)
            native['padding_policy']=dict(original_pad_token_id=None,pad_token_id=2,eos_token_ids=[2],padding_side='left',
                policy='explicit_existing_pad_else_first_configured_EOS/v1')
            if self.c['runtime'].get('asr_preprocessing'):native['preprocessing']=feature_proof(native,row,self.c)
            audit=normalize_with_audit('完整字')
            outputs[row['row_id']]=dict(status='completed',raw_text='完整字',normalized_text='完整字',normalization_audit=audit,
                language='Chinese',native=native,batch_rows=ids,batch_size=len(ids),inference_call=dict(stage='asr',rows=len(ids),
                producer_row_ids=ids,input_frames=[r['model_input_frames']for r in rows],execution_policy=S.POLICY))
        return outputs

    def verify(self,i=0,**changes):
        args=dict(record=self.records[i],window=self.windows[i],contract=self.c,contract_sha=self.contract_sha,
            bundle_sha=self.bundle_sha,tree=self.tree,cache={});args.update(changes)
        return A.verify_native_window(**args)

    def test_real_whole_json_readback_source_pcm_model_and_rows_without_db(self):
        cache={}
        with patch('sqlite3.connect',side_effect=AssertionError('database forbidden')):
            for i in range(4):self.assertEqual(self.verify(i,cache=cache)['row_id'],self.rows[i]['row_id'])
        self.assertEqual(len(cache),1);self.assertFalse(list(self.root.rglob('*.sqlite*')))

    def test_missing_reference_is_not_a_successful_native_window(self):
        self.windows[0]['asr'].pop('durable_native_batch')
        with self.assertRaisesRegex(RuntimeError,'reference missing'):self.verify()

    def test_packet_bytes_plan_or_model_binding_changed_refused(self):
        for change in ('packet','plan','bundle','contract'):
            with self.subTest(change=change):
                if change=='bundle':
                    with self.assertRaisesRegex(RuntimeError,'producer/model/contract'):self.verify(bundle_sha='a'*64)
                elif change=='contract':
                    with self.assertRaisesRegex(RuntimeError,'producer/model/contract'):self.verify(contract_sha='a'*64)
                else:
                    ref=self.windows[0]['asr']['durable_native_batch'];p=Path(ref['packet_path'])if change=='packet'else Path(ref['packet_path']).parent.parent/'plan.json'
                    raw=p.read_bytes();p.write_bytes(raw+b' ')
                    with self.assertRaisesRegex(RuntimeError,'SHA differs'):self.verify()
                    p.write_bytes(raw)

    def test_wrong_source_pcm_or_row_position_refused(self):
        for change in ('source','PCM','position'):
            record=copy.deepcopy(self.records[0]);window=copy.deepcopy(self.windows[0])
            if change=='source':record['source']['relative_path']='another.wav'
            elif change=='PCM':window['input_pcm_sha256']='f'*64
            else:window['asr']['durable_native_batch']['position']=1
            with self.subTest(change=change),self.assertRaisesRegex(RuntimeError,'source/PCM'):self.verify(record=record,window=window)

    def test_effective_tokens_stop_at_first_eos_and_padding_is_not_extra_speech(self):
        native=self.outputs[self.rows[0]['row_id']]['native']
        self.assertEqual(A.validate_native_tokens(native,self.c,4,0),1)
        bad=copy.deepcopy(native);bad['generated_token_ids']=[11,2,88];bad['post_eos_token_ids']=[88]
        with self.assertRaisesRegex(RuntimeError,'after first EOS'):A.validate_native_tokens(bad,self.c,4,0)
        bad=copy.deepcopy(native);bad['effective_generated_token_ids']=[11,2,2]
        with self.assertRaisesRegex(RuntimeError,'effective tokens'):A.validate_native_tokens(bad,self.c,4,0)

    def test_wrong_padding_mask_generation_map_or_training_refused(self):
        for key,value in [('input_attention_mask',[1,0,1]),('actual_generate_rows',1),('actual_generate_row_index',1),('grad_enabled',True)]:
            bad=copy.deepcopy(self.outputs[self.rows[0]['row_id']]['native']);bad[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):A.validate_native_tokens(bad,self.c,4,0)

    def test_failed_decode_retained_without_invented_cohort(self):
        window=dict(window_id='bad',preprocessing_status='failed',asr=dict(status='failed',raw_text='',native={'eos_observed':False}))
        self.assertIsNone(self.verify(window=window))

    def test_unchanged_single_window_auditor_still_refuses_native(self):
        import audit_speed_comparison as old
        with self.assertRaisesRegex(RuntimeError,'single-window'):old.compare_contracts(self.c,self.c,'speed')

    def test_only_batch_width_change_allowed_and_fa_always_one(self):
        a,b=contract(1),contract(8);A.compare_contracts(a,b,'speed')
        b['runtime']['asr_compute']['dtype']='bfloat16'
        with self.assertRaisesRegex(RuntimeError,'only declared ASR batch'):A.compare_contracts(a,b,'speed')
        b=contract(4);b['purpose']='silver_annotation'
        with self.assertRaisesRegex(RuntimeError,'pilot contract'):A.validate_contract(b)

    def test_weighted_actual_call_counts_not_row_count_equals_batch_count(self):
        metrics=dict(asr_call_rows={'4':2,'1':1},asr_rows=9,asr_batches=3,aligner_call_rows={'1':8},aligner_rows=8,aligner_batches=8)
        A.validate_call_counts(metrics,self.c)
        metrics['aligner_call_rows']={'4':2}
        with self.assertRaisesRegex(RuntimeError,'Ultra'):A.validate_call_counts(metrics,self.c)

    def native_track(self,record,width=1,seconds=10,cold=True):
        value=old_track([record],seconds,cold);value['contract']=contract(width)
        return value

    def test_complete_denominator_keeps_failed_missing_and_all_errors(self):
        a,b=pair();b['windows']=[];b['decision']['state']='failed';b['errors'].append({'complete_error':'preserve'})
        r=A.analyze(self.native_track(a),self.native_track(b,4,5))
        self.assertEqual(r['source_count'],1);self.assertEqual(r['counts']['missing_windows'],1)
        self.assertEqual(r['rows'][0]['complete_source_observations']['candidate']['errors'],b['errors'])
        self.assertFalse(r['cold_speed_ratio_valid'])

    def test_reuse_cannot_be_cold_speed_and_cpu_never_gpu_success(self):
        a,b=pair();r=A.analyze(self.native_track(a),self.native_track(b,4,2,False))
        self.assertFalse(r['cold_speed_ratio_valid']);self.assertIsNone(r['cold_end_to_end_speed_ratio'])
        self.assertFalse(r['human_text_or_boundary_accuracy_certified'])

    def test_recovery_ignores_only_execution_cache_marker_not_packet_bytes(self):
        a,b=pair();a['windows'][0]['asr']['durable_native_batch']={'packet_sha256':'a'*64,'cache_hit':False}
        b['windows'][0]['asr']['durable_native_batch']={'packet_sha256':'a'*64,'cache_hit':True};b['windows'][0]['asr_cache_hit']=True
        r=A.analyze(self.native_track(a),self.native_track(b),mode='recovery');self.assertTrue(r['strict_native_results_equal'])
        b['windows'][0]['asr']['durable_native_batch']['packet_sha256']='b'*64
        r=A.analyze(self.native_track(a),self.native_track(b),mode='recovery');self.assertFalse(r['strict_native_results_equal'])


class NativeFullReadbackTests(unittest.TestCase):
    def setUp(self):
        import test_audit_speed_comparison as F
        from schema import build_record
        self.f=F.FullReadbackTests();self.addCleanup(lambda:self.f.tearDown())
        c=contract(1)
        if getattr(self,'FEATURE_PROTOCOL',False):add_feature_protocol(c)
        def build(*args):
            source=args[0];window=args[4][0];pcm=np.zeros(32000,dtype=np.float32)
            window['input_pcm_sha256']=hashlib.sha256(pcm.tobytes()).hexdigest()
            row=dict(row_id=source['source_id']+':'+window['window_id'],source_id=source['source_id'],window_id=window['window_id'],
                source_record_sha256=digest(source),input_pcm_sha256=window['input_pcm_sha256'],model_input_frames=32000,
                model_sample_rate=16000,audio=pcm,context={'source':source})
            model=digest(model_provenance(c)['asr']);producer=digest(dict(contract_sha256=self.f.csha,bundle_sha256='6'*64,
                model_sha256=model,execution_policy=S.POLICY,asr_batching=c['runtime']['asr_batching']))
            store=S.NativeTransformersBatchStore(self.f.root/'labels'/c['version']/'.native_torch_cohorts'/'claim'/'group',producer,
                synthetic_fixture=True,contract_sha256=self.f.csha,model_sha256=model,max_batch_rows=1)
            def infer(rows):
                native=dict(generated_token_ids=[11,2],effective_generated_token_ids=[11,2],effective_generated_token_count=2,
                    eos_observed=True,first_eos_index=1,eos_token_ids=[2],padding_token_id=2,raw_returned_token_count=2,
                    post_eos_token_ids=[],input_padded_tokens=2,input_attention_mask=[1,1],input_nonpad_tokens=2,
                    input_token_ids=[8,9],
                    actual_generate_rows=1,actual_generate_row_index=0,max_new_tokens=c['runtime']['max_new_tokens'],
                    model_training=False,inference_mode=True,grad_enabled=False)
                native['padding_policy']=dict(original_pad_token_id=None,pad_token_id=2,eos_token_ids=[2],padding_side='left',
                    policy='explicit_existing_pad_else_first_configured_EOS/v1')
                if c['runtime'].get('asr_preprocessing'):native['preprocessing']=feature_proof(native,row,c)
                return {row['row_id']:dict(status='completed',raw_text='哈哈',normalized_text='哈哈',language='Chinese',
                    normalization_audit=normalize_with_audit('哈哈'),native=native,batch_rows=[row['row_id']],batch_size=1,
                    inference_call=dict(stage='asr',rows=1,producer_row_ids=[row['row_id']],input_frames=[32000],execution_policy=S.POLICY))}
            window['asr']=store.run_batch([row],infer_batch=infer)[row['row_id']]
            window['alignment']['inference_call']=dict(stage='aligner',rows=1,producer_row_ids=[row['row_id']],input_frames=[32000],execution_policy=S.POLICY)
            return build_record(*args)
        with patch.object(F,'contract',return_value=c),patch.object(F,'build_record',side_effect=build):self.f.setUp()
        self.f.spec['schema']=A.SPEC_SCHEMA;self.f.specpath.write_bytes(canonical_bytes(self.f.spec))
        self.sha=hashlib.sha256(self.f.specpath.read_bytes()).hexdigest()

    def test_entire_real_finite_control_source_sidecar_batch_ready_ack_chain_read_only(self):
        before={str(p):p.read_bytes()for p in self.f.root.rglob('*')if p.is_file()}
        with patch('sqlite3.connect',side_effect=AssertionError('database forbidden')):value=A.load_track(self.f.specpath,self.sha)
        self.assertEqual(len(value['records']),1);self.assertEqual(value['native_batch_packets_read'],1)
        self.assertTrue(value['native_whole_batch_readback_verified']);self.assertFalse(value['performance']['actual_compute_execution_verified'])
        self.assertEqual(before,{str(p):p.read_bytes()for p in self.f.root.rglob('*')if p.is_file()})

    def test_full_native_cli_has_no_gpu_or_quality_admission(self):
        output=self.f.root/'native-report.json'
        with patch('sqlite3.connect',side_effect=AssertionError('database forbidden')):
            rc=A.main(['--baseline-spec',str(self.f.specpath),'--baseline-spec-sha',self.sha,
                '--candidate-spec',str(self.f.specpath),'--candidate-spec-sha',self.sha,'--output',str(output)])
        self.assertEqual(rc,0);value=json.loads(output.read_text());self.assertEqual(value['source_count'],1)
        self.assertFalse(value['cold_speed_ratio_valid']);self.assertFalse(value['human_text_or_boundary_accuracy_certified'])

    def test_readback_receipt_is_complete_pure_memory_and_cannot_hide_changed_records(self):
        value=A.load_track(self.f.specpath,self.sha)
        with patch.object(Path,'open',side_effect=AssertionError('receipt must not reopen files')):
            receipt=A.native_readback_receipt(value)
        self.assertEqual(receipt['schema'],'annotation-native-finite-readback/v1');self.assertEqual(receipt['source_count'],1)
        self.assertEqual(len(receipt['sources']),1);self.assertEqual(len(receipt['native_packets']),1)
        self.assertEqual(receipt['native_packets'][0]['packet']['sha256'],receipt['sources'][0]['windows'][0]['durable_native_batch']['packet_sha256'])
        next(iter(value['records'].values()))['errors'].append({'changed_after_read':True})
        with self.assertRaisesRegex(RuntimeError,'observations changed'):A.native_readback_receipt(value)

    def test_readback_receipt_size_cap_refuses_instead_of_truncating(self):
        value=A.load_track(self.f.specpath,self.sha)
        with patch.object(A,'MAX_READBACK_BYTES',1),self.assertRaisesRegex(RuntimeError,'64MiB'):A.native_readback_receipt(value)


class NativeB1FeatureProofTests(unittest.TestCase):
    def setUp(self):
        self.f=NativeAuditTests();self.f.FEATURE_PROTOCOL=True;self.f.setUp();self.addCleanup(self.f.doCleanups)

    def native(self):return copy.deepcopy(self.f.outputs[self.f.rows[0]['row_id']]['native'])

    def test_actual_whole_packet_feature_source_and_prompt_readback(self):
        self.assertEqual(self.f.verify()['row_id'],self.f.rows[0]['row_id'])

    def test_actual_tensor_witness_cannot_be_a_self_signed_boolean(self):
        changes=[('actual_generate_valid_feature_sha256','b'*64),('actual_generate_feature_mask',[1,1,0,1,0,0]),
                 ('actual_generate_feature_shape',[3,6]),('actual_generate_feature_dtype','float16'),
                 ('actual_generate_feature_padding_is_zero',False),('verified_from_actual_generate_tensors',False),
                 ('official_processor_rows',4),('actual_generate_nonpad_prompt_token_ids',[8,10]),
                 ('actual_generate_audio_placeholder_count',2),('actual_generate_feature_mask_dtype','int64'),
                 ('source_bindings_sha256','f'*64)]
        for key,value in changes:
            n=self.native();n['preprocessing'][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):A.validate_native_tokens(n,self.f.c,4,0)

    def test_feature_reference_must_bind_original_request_not_another_PCM(self):
        for key,value in [('row_id','another:w'),('frames',321),('original_input_pcm_sha256','f'*64)]:
            n=self.native();n['preprocessing'][key]=value
            if key=='original_input_pcm_sha256':n['preprocessing']['SDK_normalization_changed_PCM']=True
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'another source PCM'):
                A.validate_native_preprocessing(n,self.f.c,4,0,self.f.rows[0])

    def test_new_policy_needs_proof_and_old_policy_cannot_hide_preprocessing(self):
        n=self.native();n.pop('preprocessing')
        with self.assertRaisesRegex(RuntimeError,'proof missing'):A.validate_native_tokens(n,self.f.c,4,0)
        with self.assertRaisesRegex(RuntimeError,'undeclared'):A.validate_native_tokens(self.native(),contract(),4,0)
        with self.assertRaisesRegex(RuntimeError,'fixed producer field|only declared ASR batch'):
            A.compare_contracts(contract(),self.f.c,'speed')

    def tracks(self):
        a,b=pair();row=self.f.rows[0]
        for record in (a,b):
            record['source']=copy.deepcopy(self.f.records[0]['source'])
            w=record['windows'][0];w.update(window_id='w',input_pcm_sha256=row['input_pcm_sha256'],
                model_input_frames=row['model_input_frames'],model_sample_rate=16000)
            w['asr']=copy.deepcopy(self.f.outputs[row['row_id']]);n=w['asr']['native']
            n['actual_generate_rows']=1;n['actual_generate_row_index']=0
            n['preprocessing'].update(actual_generate_rows=1,actual_generate_row_index=0,position=0)
        tracks=[old_track([a],10),old_track([b],5)]
        for t,width in zip(tracks,(1,4)):
            t['contract']=contract(width);t['contract']['purpose']='pilot_validation'
            add_feature_protocol(t['contract'])
            t['performance']['native_whole_batch_readback_verified']=True
        return tracks

    def test_equal_effective_inputs_allow_controlled_cold_ratio_but_not_human_truth(self):
        a,b=self.tracks();result=A.analyze(a,b)
        self.assertTrue(result['cold_speed_ratio_valid']);self.assertEqual(result['cold_end_to_end_speed_ratio'],2)
        self.assertTrue(result['native_B1_effective_inputs_equal']);self.assertEqual(result['native_B1_preprocessing_verified_windows'],[1,1])
        self.assertFalse(result['human_text_or_boundary_accuracy_certified'])

    def test_same_PCM_but_different_valid_mel_cannot_qualify_cold_speed(self):
        a,b=self.tracks();p=next(iter(b['records'].values()))['windows'][0]['asr']['native']['preprocessing']
        p['feature_valid_sha256']=p['actual_generate_valid_feature_sha256']='b'*64
        result=A.analyze(a,b);self.assertEqual(result['counts']['input_mismatch_windows'],0)
        self.assertEqual(result['counts']['preprocessing_effective_input_mismatch_windows'],1)
        self.assertFalse(result['cold_speed_ratio_valid']);self.assertIsNone(result['cold_end_to_end_speed_ratio'])
        self.assertFalse(result['rows'][0]['windows'][0]['ASR']['preprocessing_valid_mel_equal'])

    def test_changed_nonpad_prompt_or_SDK_PCM_is_not_output_padding_metadata(self):
        for kind in ('prompt','SDK_PCM'):
            a,b=self.tracks();n=next(iter(b['records'].values()))['windows'][0]['asr']['native'];p=n['preprocessing']
            if kind=='prompt':
                n['input_token_ids'][-1]=10;p['baseline_input_token_ids'][-1]=10;p['actual_generate_nonpad_prompt_token_ids'][-1]=10
            else:p['sdk_input_pcm_sha256']='f'*64;p['SDK_normalization_changed_PCM']=True
            with self.subTest(kind=kind):
                r=A.analyze(a,b);self.assertFalse(r['cold_speed_ratio_valid']);self.assertFalse(r['native_B1_effective_inputs_equal'])

    def test_batch_padding_width_alone_can_differ_but_original_features_must_not(self):
        a,b=self.tracks();n=next(iter(b['records'].values()))['windows'][0]['asr']['native'];p=n['preprocessing']
        p.update(actual_generate_feature_padded_frames=9,actual_generate_feature_shape=[2,9],actual_generate_feature_mask=[1]*4+[0]*5)
        n['input_token_ids']=[0,0,8,9];n['input_attention_mask']=[0,0,1,1];n['input_padded_tokens']=4
        self.assertTrue(A.analyze(a,b)['native_B1_effective_inputs_equal'])

    def test_new_protocol_requires_exact_seven_sources_and_original_model_config_manifest(self):
        c=copy.deepcopy(self.f.c);c.pop('asr_preprocessing_sources')
        with self.assertRaisesRegex(RuntimeError,'seven preprocessing'):A.validate_contract(c)
        c=copy.deepcopy(self.f.c);c['asr_preprocessing_sources']['files'].pop('audio_utils')
        with self.assertRaisesRegex(RuntimeError,'seven preprocessing'):A.validate_contract(c)
        c=copy.deepcopy(self.f.c);c['asr_preprocessing_sources']['processor_config']['sha256']='f'*64
        with self.assertRaisesRegex(RuntimeError,'actual model manifest'):A.validate_contract(c)
        with patch.object(Path,'open',side_effect=AssertionError('offline structure check must not reread SDK source files')),\
             patch.object(Path,'stat',side_effect=AssertionError('per-row source getter must not issue filesystem metadata probes')):
            for _ in range(100):self.assertEqual(A.feature_source_bindings(self.f.c),self.f.c['asr_preprocessing_sources'])

    def test_same_model_but_another_feature_library_cannot_be_controlled_comparison(self):
        a,b=self.tracks();b['contract']['asr_preprocessing_sources']['files']['whisper_feature_extractor']['sha256']='f'*64
        with self.assertRaisesRegex(RuntimeError,'asr_preprocessing_sources'):A.analyze(a,b)

    def test_actual_worker_live_sources_and_configuration_cannot_be_just_a_boolean(self):
        c=copy.deepcopy(self.f.c);c['purpose']='pilot_validation';bindings=c['asr_preprocessing_sources']
        obs=dict(copy.deepcopy(bindings),verified_from_live_objects=True,
                 actual_feature_extractor_config=dict(sampling_rate=16000,feature_size=2,hop_length=160,n_fft=400))
        self.assertEqual(A.validate_preprocessing_sources_observation({'asr_preprocessing_sources_observed':obs},c),obs)
        for kind in ('missing','flag','source','config'):
            bad=copy.deepcopy(obs)
            if kind=='flag':bad['verified_from_live_objects']=False
            elif kind=='source':bad['files']['audio_utils']['sha256']='f'*64
            elif kind=='config':bad['actual_feature_extractor_config']={}
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):
                A.validate_preprocessing_sources_observation({}if kind=='missing'else{'asr_preprocessing_sources_observed':bad},c)


class NativeB1FeatureFullReadbackTests(NativeFullReadbackTests):
    FEATURE_PROTOCOL=True


if __name__=='__main__':unittest.main()
