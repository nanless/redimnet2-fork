"""新native正式协议的合成原件反例；不读取真实媒体、模型、GPU或平台。"""
from __future__ import annotations
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(HERE),str(HERE/'tests')]
import bounded_speed_runtime as M
import native_batch_admission as A
import native_batch_readback as B
import native_transformers_batch as S
import contracts as C
import test_bounded_speed_runtime as OldFixture


class NativeFixture:
    def __init__(self,case,full_code=False):
        self.old=OldFixture.RuntimeFixture();self.old.setUp();case.addCleanup(self.old.doCleanups);self.f=self.old.f;self.put=self.old.put
        self.code=self.old.code
        if full_code:
            import native_batch_queue_handoff as Queue
            for name in Queue.REQUIRED_SOURCE:
                target=self.code/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((HERE/name).read_bytes())
            members={name:C.sha_file(self.code/name)for name in Queue.REQUIRED_SOURCE}
        else:
            (self.code/'gpu_launch.py').write_text('# CPU format fixture, never execute');members={'gpu_launch.py':C.sha_file(self.code/'gpu_launch.py')}
        self.bundle=self.put(str((self.code/'bundle_files.json').relative_to(self.f.root)),members);self.bsha=self.bundle['sha256']
        c=copy.deepcopy(self.old.c);c['runtime'].update(execution_policy=C.NATIVE_BATCH_POLICY,asr_compute=dict(dtype='float32',attention='sdpa'),asr_batching=dict(policy='fixed_cohort_until_durable/v1',max_requests=4,ordering='frames_then_row_id_within_source_group/v1'))
        c['purpose']='cpu_fixture';c['version']='native-pilot';self.c=c;self.cp=self.put('native-pilot.json',c)
        self.new=copy.deepcopy(c);self.new['version']='native-formal-fixture';self.newpin=self.put('native-new.json',self.new)
        base=copy.deepcopy(c);base['version']='native-baseline';base['runtime']['asr_batching']['max_requests']=1;self.base=base;self.basepin=self.put('native-base.json',base)
        self.sources=self.f.sources[:4];self.index=self.index_for(self.sources,'native-main');self.ri=self.index_for(self.sources[:2],'native-recovery')
        self.tracks={};self.contracts={'baseline':base,'main':c,'recovery':c};self.contractpins={'baseline':self.basepin,'main':self.cp,'recovery':self.cp};self.resources={};self.readback={};self.outputs={};self.refmaps={};self.records={};self.performances={};self.step=[]
        for role in ('baseline','main'):
            self.make_track(role,self.sources,self.index,self.f.root/(role+'-native-labels'))
        self.make_track('recovery',self.sources[:2],self.ri,Path(self.tracks['main']['value']['output_root']),reuse='main')
        matrix=dict(schema='annotation-native-batch-prepared/v1',source_count=4,variants={
            'native-fp32-sdpa-b1':dict(track=self.tracks['baseline']['pin'],freeze=dict(bundle_sha256=self.bsha,frozen_copy=str(self.code))),
            'native-fp32-sdpa-b4':dict(track=self.tracks['main']['pin'],freeze=dict(bundle_sha256=self.bsha,frozen_copy=str(self.code)))})
        matrixpin=self.put('native-matrix.json',matrix);planpin=self.put('native-plan.json',dict(schema='annotation-native-batch-pilot/v1',main_prepared=matrixpin))
        result=self.put('native-result.json',dict(schema='annotation-native-batch-pilot-result/v1',purpose='cpu_fixture',all_finite_audits_completed=True,database_created=False,plan=planpin,steps=self.step))
        speed=self.comparison('speed');self.speed=self.put('native-speed.json',speed);recovery=self.put('native-recovery-comparison.json',self.comparison('recovery'))
        self.quality=dict(schema=A.QUALITY_SCHEMA,purpose='cpu_fixture',comparison=self.speed,candidate_contract=self.cp,review_complete=True,all_changed_rows_dispositioned=True,text_quality_accepted=True,complete_word_boundary_quality_accepted=True,candidate_labels_released=False,dispositions=[],review_evidence=[self.put('native-quality-source.json',dict(synthetic_CPU_only=True))])
        qpin=self.put('native-quality.json',self.quality)
        self.selected=dict(schema=A.SCHEMA,purpose='cpu_fixture',candidate_contract=self.cp,baseline_track=self.tracks['baseline']['pin'],main_track=self.tracks['main']['pin'],recovery_track=self.tracks['recovery']['pin'],selected_profile_name='native-fp32-sdpa-b4',selected_asr_batching=c['runtime']['asr_batching'],comparison=self.speed,recovery_comparison=recovery,pilot_result=result,quality_review=qpin,resource_evidence=self.resources,native_readbacks=self.readback)
        self.save()

    def index_for(self,sources,name):
        path=self.f.root/(name+'.jsonl');path.write_bytes(C.canonical_bytes(sources[0])if len(sources)==1 else b''.join(C.canonical_bytes(s)for s in sources))
        return self.put(name+'-index.json',dict(schema='redimnet2-shard-index/v1',source_count=len(sources),shards=[dict(id=name,path=path.name,count=len(sources),sha256=C.sha_file(path))]))
    def save(self):self.pin=self.put('native-selected.json',self.selected)
    def perf(self,role,count):return self.performances[role]
    def compute_metrics(self,rows,width):
        def loaded(dtype,attention):return dict(requested=dict(dtype=dtype,attention=attention),floating_parameter_elements_by_dtype={dtype:100},attention_configurations=[dict(module='config',attention_config=attention)])
        value=dict(model_API_calls=0,asr_rows=rows,aligner_rows=rows,asr_batches=0 if not rows else (rows+width-1)//width,aligner_batches=rows,
            asr_seconds=.2 if rows else 0.,aligner_seconds=.1 if rows else 0.,model_load_seconds=.1,
            asr_call_rows={}if not rows else {str(min(rows,width)):(rows+width-1)//width},aligner_call_rows={}if not rows else {'1':rows},native_ASR_stage_cache_hits=0,
            first_call_trace_completed=bool(rows),loaded_compute=dict(asr=loaded('float32','sdpa'),aligner=loaded('float32','eager')),
            first_call_compute_observation=dict(attention_forwards=dict(audio=1,text=2),functional_sdpa_calls=dict(audio=1,text=2),attention_module_configurations=dict(audio='sdpa',text='sdpa'),attention_configurations_stable=True,functional_sdpa_qkv_dtypes=dict(audio=['float32']*3,text=['float32']*3),functional_sdpa_qkv_dtypes_stable=True))
        return value
    def make_track(self,role,sources,index,labels,reuse=None):
        contract=self.contracts[role];pin=self.contractpins[role];resources=self.f.root/(role+'-native-resources')
        track=dict(schema=A.TRACK_SCHEMA,execution_policy=C.NATIVE_BATCH_POLICY,contract=pin,index=index,queue_root=str(self.f.root/(role+'-queue')),output_root=str(labels),resources_root=str(resources))
        tp=self.put(role+'-native-track.json',track);self.tracks[role]=dict(value=track,pin=tp)
        packets={};maps={};source_refs=[];outputs={}
        if reuse:
            packets=copy.deepcopy(self.refmaps[reuse]['packets']);outputs=self.outputs[reuse]
        else:
            model_sha=C.digest(C.model_provenance(contract)['asr']);producer=C.digest(dict(contract_sha256=pin['sha256'],bundle_sha256=self.bsha,model_sha256=model_sha,execution_policy=C.NATIVE_BATCH_POLICY,asr_batching=contract['runtime']['asr_batching']))
            store=S.NativeTransformersBatchStore(labels/contract['version']/'.native_torch_cohorts'/'shard'/'group',producer,synthetic_fixture=True,contract_sha256=pin['sha256'],model_sha256=model_sha,max_batch_rows=contract['runtime']['asr_batching']['max_requests'])
            rows=[]
            for i,source in enumerate(sources):
                pcm=np.full(160+i,i/10,dtype=np.float32)
                rows.append(dict(row_id=source['source_id']+':w',source_id=source['source_id'],window_id='w',source_record_sha256=C.digest(source),input_pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),model_input_frames=len(pcm),model_sample_rate=16000,audio=pcm))
            owner=SimpleNamespace(c=contract);outputs=store.run_batch(rows,infer_batch=lambda batch:NativeFixtureModule.NativeAuditTests.infer(owner,batch))
        records={}
        for source in sources:
            sid=source['source_id'];out=outputs[sid+':w'];ref=out['durable_native_batch'];packet=json.loads(Path(ref['packet_path']).read_text());content=packet['content'];request=content['cohort']['intent']['requests'][ref['position']]
            pp=dict(path=ref['packet_path'],sha256=ref['packet_sha256']);plan=dict(path=str(Path(pp['path']).parent.parent/'plan.json'),sha256=ref['plan_sha256'])
            packets[pp['path']]=dict(packet=pp,plan=plan,cohort_sha256=ref['cohort_sha256'],store_identity_sha256=C.digest(content['store_identity']),requests=content['cohort']['intent']['requests'])
            window=dict(window_id='w',preprocessing_status='completed',request=request,durable_native_batch=ref)
            sidecar=labels/contract['version']/source['sidecar_relpath']
            source_refs.append(dict(source_id=sid,source_record_sha256=C.digest(source),sidecar=dict(path=str(sidecar),sha256='1'*64),commit=dict(path=str(sidecar)+'.commit.json',sha256='2'*64),windows=[window],errors_sha256=C.digest([]),decision_state='candidate'))
            maps[sid]=source_refs[-1]
            if reuse:record=copy.deepcopy(self.records[reuse][sid])
            else:
                geometry=dict(window_id='w',context_start_sample=0,context_end_sample=request['model_input_frames'],owned_start_sample=0,owned_end_sample=request['model_input_frames'],
                    input_pcm_sha256=request['input_pcm_sha256'],model_input_frames=request['model_input_frames'],model_sample_rate=16000,resampling_policy='synthetic_no_resampling',time_mapping='source_offset_decimal_floor_start_ceil_end_v2',preprocessing_status='completed',asr_cache_hit=False,aligner_cache_hit=False)
                alignment=dict(status='completed_native_candidates',target_text=out['normalized_text'],encoded_target_words=['完整字'],native_timestamps_ms=[0,5],fixed_timestamps_ms=[0,5],repair_changed=False,author_final_items=[],items=[dict(text='完整字',start_time=0.,end_time=.005)])
                geometry.update(asr=copy.deepcopy(out),alignment=alignment)
                record=dict(source=source,source_identity=dict(decode_status='decoded_complete',file_sha256=C.sha_file(source['audio_path']),pcm_sha256=request['input_pcm_sha256'],pcm_hash_policy='decoded_float32le_native_interleaved/v1',sample_rate=16000,channels=1,frames=request['model_input_frames']),
                    units=[],decision=dict(state='candidate',usable_for_training=False,usable_for_evaluation=False),quality_axes={},errors=[],vad={},windows=[geometry],
                    contract=dict(sha256=pin['sha256'],version=contract['version']),provenance=dict(code_sha256=self.bsha),execution=dict(worker_id='worker-0',attempt_id=role+'-attempt',job_key=C.job_key(source,pin['sha256'])))
            records[sid]=record
            raw=C.canonical_bytes(record);sidecar.parent.mkdir(parents=True,exist_ok=True)
            if not reuse:sidecar.write_bytes(raw)
            source_refs[-1]['sidecar']['sha256']=C.sha_file(sidecar)
            marker=dict(schema='redimnet2-sidecar-commit/v1',source_id=sid,job_key=C.job_key(source,pin['sha256']),contract_sha256=pin['sha256'],source_record_sha256=C.digest(source),sidecar_sha256=C.sha_file(sidecar),bytes=sidecar.stat().st_size)
            marker_path=Path(str(sidecar)+'.commit.json')
            if not reuse:marker_path.write_bytes(C.canonical_bytes(marker))
            source_refs[-1]['commit']['sha256']=C.sha_file(marker_path)
        self.records[role]=records
        # 恢复只列它实际引用的包，包本身仍保留原main完整4成员。
        used={r['windows'][0]['durable_native_batch']['packet_path']for r in source_refs};packets={k:v for k,v in packets.items()if k in used}
        # 小chunk、两worker ready/ACK和父结果都是真实本机JSON，仍明确是CPU格式数据。
        chunk_sources=[dict(source_id=src['source_id'],job_key=C.job_key(src,pin['sha256']),sidecar_sha256=maps[src['source_id']]['sidecar']['sha256'],commit_sha256=maps[src['source_id']]['commit']['sha256'],sidecar_relpath=src['sidecar_relpath'],sidecar_bytes=Path(maps[src['source_id']]['sidecar']['path']).stat().st_size,decision='candidate')for src in sources]
        shard_id=json.loads(Path(index['path']).read_text())['shards'][0]['id']
        chunk=dict(schema='redimnet2-chunk-commit/v1',shard_id=shard_id,source_count=len(sources),contract_sha256=pin['sha256'],index_sha256=index['sha256'],attempt_id=role+'-attempt',sources=chunk_sources)
        chunkpin=self.put(str((Path(track['queue_root'])/'completed'/(shard_id+'.json')).relative_to(self.f.root)),chunk)
        receipt=dict(sha256=chunkpin['sha256'],value=chunk,fresh_source_ids=set()if reuse else set(records))
        workers=[]
        for number in (0,1):
            wid='worker-'+str(number);active=number==0;assigned=len(sources)if active else 0;newrows=0 if reuse else assigned
            summary=dict(worker_id=wid,contract_sha256=pin['sha256'],index_sha256=index['sha256'],device_uuid='CPU-noGPU-'+str(number),assigned_source_rows=assigned,sidecars_committed=newrows,sidecar_cache_hits=assigned if reuse else 0,model_windows=newrows,decoded_audio_seconds=.1*newrows,stage_cache_hits=dict(asr=0,aligner=0),elapsed_seconds=1.,model_API_calls=0,
                chunk_receipt_refs=[dict(id=shard_id,sha256=chunkpin['sha256'],source_count=assigned)]if active else [],backend_metrics=self.compute_metrics(newrows,contract['runtime']['asr_batching']['max_requests']))
            sp=self.put(str((resources/wid/'worker_summary.json').relative_to(self.f.root)),summary)
            ready=dict(summary_sha256=sp['sha256'],contract_sha256=pin['sha256'],index_sha256=index['sha256'],bundle_sha256=self.bsha,nonce='CPU-nonce-'+wid)
            rp=self.put(str((resources/wid/'terminal_ready.json').relative_to(self.f.root)),ready)
            self.put(str((resources/wid/'terminal_ack.json').relative_to(self.f.root)),dict(ready_sha256=rp['sha256'],nonce=ready['nonce']))
            workers.append(dict(worker_id=wid,uuid=summary['device_uuid'],terminal=dict(summary=summary,summary_sha256=sp['sha256'],ready_sha256=rp['sha256'])))
        parent=dict(schema='open-annotation-independent-gpu-run/v1',contract_sha256=pin['sha256'],index_sha256=index['sha256'],bundle_sha256=self.bsha,worker_count=2,workers=workers,resource_returned=True,full_sidecar_readback=True,parent_elapsed_seconds=10. if role=='baseline' else 5. if role=='main' else 2.,gpu_lease_wall_seconds=1.,verification=dict(passed=True,database_created=False,source_disjointness_storage='bounded_pilot_memory_set'))
        pp=self.put(str((resources/'parent_result.json').relative_to(self.f.root)),parent)
        idle=self.put(str((resources/'resource_return_double_probe.json').relative_to(self.f.root)),dict(interval_seconds=5,snapshots=[dict(compute_processes=[],devices=[dict(used_mib=0)])]*2));drain=self.put(str((resources/'terminal_drain_complete.json').relative_to(self.f.root)),dict(all_attested_contexts_absent=True));self.resources[role]=dict(parent_result=pp,resource_return=idle,terminal_drain=drain)
        actual_perf=NativeFixtureModule.A.summarize_workers(parent,resources,{shard_id:receipt},contract);actual_perf['native_whole_batch_readback_verified']=True;self.performances[role]=actual_perf
        proof=dict(schema='annotation-native-finite-readback/v1',purpose='cpu_fixture',track_spec=tp,track_spec_sha256=tp['sha256'],contract_sha256=pin['sha256'],index_sha256=index['sha256'],bundle_sha256=self.bsha,source_count=len(sources),sources=source_refs,native_packets=list(packets.values()),native_packet_count=len(packets),records_sha256=C.digest(records),parent_result=pp,performance=actual_perf,full_sidecar_commit_readback_verified=True,native_whole_batch_readback_verified=True,model_API_calls=0,database_created=False,training_labels_released=False)
        in_memory=dict(records=records,receipts={shard_id:receipt},spec=track,spec_sha256=tp['sha256'],contract_sha256=pin['sha256'],index_sha256=index['sha256'],native_whole_batch_readback_verified=True,_native_readback_receipt=proof)
        sealed=B.seal_admission_readback(in_memory)
        self.readback[role]=self.put(role+'-native-readback.json',sealed);self.outputs[role]=outputs;self.refmaps[role]=dict(sources=maps,packets=packets)
        self.step.append(dict(stage='recovery'if role=='recovery'else'main',profile='native-fp32-sdpa-b1'if role=='baseline'else'native-fp32-sdpa-b4',source_count=len(sources),sources=len(sources),resource_returned=True,parent_result_sha256=pp['sha256'],new_ASR_calls=0,new_FA_calls=0))
    def comparison(self,mode):
        left,right=('baseline','main')if mode=='speed'else('main','recovery')
        tracks={}
        for role in (left,right):
            records=self.records[role]
            tracks[role]=dict(contract=self.contracts[role],contract_sha256=self.contractpins[role]['sha256'],spec_sha256=self.tracks[role]['pin']['sha256'],
                records=records,performance=self.performances[role],decision_counts=dict(Counter(r['decision']['state']for r in records.values())))
        value=NativeFixtureModule.A.analyze(tracks[left],tracks[right],mode=mode)
        # 仅CPU结构反例模拟被声明的速度；production必先拒绝cpu_fixture合同。
        if mode=='speed':value.update(cold_speed_ratio_valid=True,cold_end_to_end_speed_ratio=2.)
        return value

# 模块引用避免unittest误将导入的TestCase再执行一次。
import test_audit_native_transformers_batch as NativeFixtureModule


class NativeAdmissionTests(unittest.TestCase):
    def setUp(self):self.f=NativeFixture(self)
    def check(self):return A.check_native_selected_pilot(self.f.pin,self.f.new,'cpu_fixture')
    def rewrite(self,key,changes):
        value=A.pinned(self.f.selected[key]);value.update(changes);self.f.selected[key]=self.f.put('changed-'+key+'.json',value);self.f.save()

    def test_complete_native_format_and_real_batch_json_no_db(self):
        with patch('sqlite3.connect',side_effect=AssertionError('no database')):value=self.check()
        self.assertEqual(value['schema'],A.SCHEMA);self.assertTrue(value['verified_evidence_pins'])

    def test_old_single_window_admission_cannot_be_borrowed(self):
        self.f.selected['schema']=M.ADMISSION_SCHEMA;self.f.save()
        with self.assertRaisesRegex(ValueError,'old single-window'):self.check()

    def test_cpu_contract_never_becomes_production_gpu_evidence(self):
        self.f.selected['purpose']='production';self.f.save()
        with self.assertRaisesRegex(ValueError,'actual native pilot'):A.check_native_selected_pilot(self.f.pin,self.f.new,'production')

    def test_slow_or_unexecuted_candidate_refused(self):
        self.rewrite('comparison',dict(cold_end_to_end_speed_ratio=.9))
        with self.assertRaisesRegex(ValueError,'speed must exceed'):self.check()

    def test_recovery_may_not_call_models_or_publish_again(self):
        r=A.pinned(self.f.selected['recovery_comparison']);r['candidate_performance']['totals']['asr_rows']=1;self.rewrite('recovery_comparison',r)
        with self.assertRaisesRegex(ValueError,'may not repeat'):self.check()

    def test_native_mathematics_source_cannot_change_after_pilot(self):
        value=copy.deepcopy(self.f.new);value['runtime']['max_new_tokens']+=1
        with self.assertRaisesRegex(ValueError,'exactly identical'):A.check_native_selected_pilot(self.f.pin,value,'cpu_fixture')

    def test_original_full_batch_members_remain_in_512_style_subset_recovery(self):
        p=self.f.readback['recovery'];proof=A.pinned(p);self.assertEqual(len(proof['sources']),2);self.assertEqual(len(proof['native_packets'][0]['requests']),4)
        self.check()

    def test_missing_packet_map_or_wrong_position_refuses(self):
        proof=A.pinned(self.f.readback['main']);proof['sources'][0]['windows'][0]['durable_native_batch']['position']=1
        self.f.selected['native_readbacks']['main']=self.f.put('bad-native-readback.json',proof);self.f.save()
        with self.assertRaisesRegex(RuntimeError,'source/PCM'):self.check()

    def test_comparison_text_cannot_differ_from_actual_packet(self):
        value=A.pinned(self.f.speed);value['rows'][0]['windows'][0]['observations']['candidate']['ASR']['raw_text']='伪造文字';self.rewrite('comparison',value)
        q=copy.deepcopy(self.f.quality);q['comparison']=self.f.selected['comparison'];self.f.selected['quality_review']=self.f.put('updated-quality.json',q);self.f.save()
        with self.assertRaisesRegex(ValueError,'not from sealed actual sidecars|difference flag|not from actual whole packet'):self.check()

    def test_changed_text_cannot_be_called_padding_metadata(self):
        value=A.pinned(self.f.speed);value['rows'][0]['windows'][0]['ASR']['raw_text_equal']=False;self.rewrite('comparison',value)
        q=copy.deepcopy(self.f.quality);q.update(comparison=self.f.selected['comparison'],dispositions=[dict(source_id=value['rows'][0]['source_id'],window_ids=['w'],classification='metadata_only_verified',review_evidence=q['review_evidence'])]);self.f.selected['quality_review']=self.f.put('bad-quality-metadata.json',q);self.f.save()
        with self.assertRaisesRegex(ValueError,'cannot be called padding'):self.check()

    def test_new_preprocessing_input_difference_never_counts_as_metadata_only(self):
        row=copy.deepcopy(A.pinned(self.f.speed)['rows'][0])
        # 旧协议不带这些新字段时保持原判断；新协议字段存在时必须明确为True。
        before=A._metadata_only(row)
        for flag in ('preprocessing_valid_mel_equal','preprocessing_feature_mask_equal','preprocessing_unpadded_prompt_equal'):
            r=copy.deepcopy(row);r['windows'][0]['ASR'][flag]=False;self.assertFalse(A._metadata_only(r))
            r['windows'][0]['ASR'][flag]=True;self.assertEqual(A._metadata_only(r),before)

    def test_numeric_or_null_equal_flag_cannot_hide_required_quality_review(self):
        original=A.pinned(self.f.speed)
        for bad in (0,1,None,'false'):
            value=copy.deepcopy(original);value['rows'][0]['windows'][0]['ASR']['preprocessing_valid_mel_equal']=bad
            with self.subTest(value=bad),self.assertRaisesRegex(ValueError,'actual booleans'):
                A.check_quality(self.f.selected['quality_review'],value,self.f.cp,'cpu_fixture')

    def feature_contract_pair(self):
        sources=dict(schema='annotation-official-b1-feature-sources/v1',files={},processor_config=self.f.put('feature-src/preprocessor_config.json',{'sampling_rate':16000}))
        for role in ('whisper_feature_extractor','sequence_feature_extractor','batch_feature','processor_mixin','audio_utils','qwen_processor','sdk_normalizer'):
            path=self.f.f.root/'feature-src'/(role+'.py');path.write_text('# synthetic CPU source only');sources['files'][role]=dict(path=str(path),sha256=C.sha_file(path))
        left=copy.deepcopy(self.f.base);right=copy.deepcopy(self.f.c)
        for c in (left,right):c['runtime']['asr_preprocessing']={'policy':'official_b1_features_then_pad/v1'};c['asr_preprocessing_sources']=copy.deepcopy(sources)
        return left,right

    def test_feature_protocol_requires_exact_seven_sources_and_configuration(self):
        left,right=self.feature_contract_pair();A.compare_native_profiles(left,right)
        right['asr_preprocessing_sources']['files']['audio_utils']['sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'preprocessing sources differ'):A.compare_native_profiles(left,right)
        with self.assertRaisesRegex(ValueError,'preprocessing sources differ'):A.formal_equivalent(left,right,'cpu_fixture')

    def test_feature_protocol_missing_sources_refused_old_absent_kept(self):
        left,right=self.feature_contract_pair();left.pop('asr_preprocessing_sources');right.pop('asr_preprocessing_sources')
        with self.assertRaisesRegex(ValueError,'seven preprocessing'):A.compare_native_profiles(left,right)
        oldleft=copy.deepcopy(self.f.base);oldright=copy.deepcopy(self.f.c);A.compare_native_profiles(oldleft,oldright)
        self.assertIsNone(A.feature_source_identity(oldleft))
        oldleft['runtime']['asr_preprocessing']=None
        with self.assertRaisesRegex((ValueError,RuntimeError),'explicit official B1'):A.feature_source_identity(oldleft)

    def test_control_gpu_parent_change_cannot_use_only_cpu_review(self):
        code=self.f.f.root/'different-code';code.mkdir();(code/'gpu_launch.py').write_text('# different GPU parent')
        pin=self.f.put(str((code/'bundle_files.json').relative_to(self.f.f.root)),dict({'gpu_launch.py':C.sha_file(code/'gpu_launch.py')}));self.f.selected['formal_execution_bundle']=pin;self.f.save()
        with self.assertRaisesRegex(ValueError,'unreviewable native formal source'):self.check()

    def test_ordinary_v1_index_cannot_start_native_silver(self):
        c=copy.deepcopy(self.f.c);c.update(purpose='silver_annotation',native_formal=copy.deepcopy(C.NATIVE_FORMAL_DECLARATION))
        with self.assertRaisesRegex(ValueError,'ordinary v1 index'):M.guard_native_formal_route(c,None)

    def test_single_runtime_v1_cannot_authenticate_native_silver(self):
        c=copy.deepcopy(self.f.c);c.update(purpose='silver_annotation',native_formal=copy.deepcopy(C.NATIVE_FORMAL_DECLARATION))
        pin=self.f.put('wrong-runtime.json',dict(schema=M.SCHEMA,purpose='production'))
        with self.assertRaisesRegex(ValueError,'cannot borrow single-window'):M.guard_native_formal_route(c,pin)

    def test_native_silver_worker_requires_actual_inherited_lease(self):
        c=copy.deepcopy(self.f.c);c.update(purpose='silver_annotation',native_formal=copy.deepcopy(C.NATIVE_FORMAL_DECLARATION))
        pin=self.f.put('new-runtime.json',dict(schema=M.NATIVE_SCHEMA,purpose='production'))
        with self.assertRaisesRegex(ValueError,'inherited verified'):M.guard_native_formal_route(c,pin,require_lease=True)

    def test_native_v2_runtime_authenticates_original_ancestor_locks(self):
        old=self.f.old;old.runtime.update(schema=M.NATIVE_SCHEMA,new_contract=self.f.newpin,selected_pilot_admission=self.f.pin,new_bundle=self.f.bundle);old.save()
        with M.RuntimeLease(old.pin)as lease:
            self.assertEqual(lease.compiled['selected']['schema'],A.SCHEMA)
            self.assertEqual(len(lease.pass_fds),4)
            self.assertTrue(lease.unique_claim_pin)

    def test_old_runtime_v1_refuses_native_even_if_explicit_contract_is_given(self):
        old=self.f.old;old.runtime.update(new_contract=self.f.newpin,selected_pilot_admission=self.f.pin,new_bundle=self.f.bundle);old.save()
        with self.assertRaisesRegex(ValueError,'old runtime v1 refuses native'):M.compile_runtime(old.pin)

    def test_native_parent_rejects_ordinary_index_before_any_model_or_gpu_read(self):
        import gpu_launch
        c=copy.deepcopy(self.f.c);c.update(purpose='silver_annotation',native_formal=copy.deepcopy(C.NATIVE_FORMAL_DECLARATION))
        args=SimpleNamespace(bundle_sha=self.f.bsha,contract=self.f.newpin['path'],contract_sha=self.f.newpin['sha256'],shard_index=self.f.index['path'],index_sha=self.f.index['sha256'],target_uuids='auto8')
        with patch.object(gpu_launch,'verify_bundle',return_value={}),patch.object(gpu_launch,'read_contract',return_value=c),patch.object(gpu_launch,'_run')as launch,patch.object(gpu_launch,'model_facts')as model,patch.object(gpu_launch.G,'capture_snapshot')as nvml:
            with self.assertRaisesRegex(ValueError,'ordinary v1 index'):gpu_launch.run(args)
            launch.assert_not_called();model.assert_not_called();nvml.assert_not_called()

    def test_real_worker_route_rejects_before_queue_store_and_backend(self):
        import worker
        c=copy.deepcopy(self.f.c);c.update(purpose='silver_annotation',native_formal=copy.deepcopy(C.NATIVE_FORMAL_DECLARATION))
        args=['--contract',self.f.newpin['path'],'--contract-sha',self.f.newpin['sha256'],'--shard-index',self.f.index['path'],'--index-sha',self.f.index['sha256'],
              '--queue-root',str(self.f.f.root/'never-queue'),'--output-root',str(self.f.f.root/'never-labels'),'--worker-id','worker-0','--worker-output',str(self.f.f.root/'refused-worker-evidence')]
        with patch.object(worker,'read_contract',return_value=c),patch.object(worker,'WorkQueue')as queue,patch.object(worker,'SidecarStore')as store,patch.object(worker,'verify_decoder')as decoder:
            with self.assertRaisesRegex(ValueError,'ordinary v1 index'):worker.main(args)
            queue.assert_not_called();store.assert_not_called();decoder.assert_not_called()

    def test_real_bootstrap_route_rejects_native_before_torch_or_model_stats(self):
        import gpu_bootstrap as bootstrap
        import os
        import builtins
        c=copy.deepcopy(self.f.c);c.update(purpose='silver_annotation',native_formal=copy.deepcopy(C.NATIVE_FORMAL_DECLARATION))
        cp=self.f.put('bootstrap-native-silver.json',c)
        ticket=self.f.put('bootstrap-synthetic-ticket.json',dict(schema='open-annotation-resource-ticket/v1',parent_pid=os.getppid(),parent_start_time='synthetic-start',boot_id='synthetic-boot',created_monotonic=0,
            worker_id='worker-0',bundle_sha256='a'*64,contract_sha256='a'*64,index_sha256=self.f.index['sha256'],nonce='synthetic-nonce',target_uuid='CPU-no-GPU',locks=[]))
        values=dict(ticket=ticket['path'],ticket_sha='a'*64,bundle_sha='a'*64,contract=cp['path'],contract_sha='a'*64,shard_index=self.f.index['path'],index_sha=self.f.index['sha256'],queue_root=str(self.f.f.root/'never-q'),output_root=str(self.f.f.root/'never-labels'),worker_id='worker-0',worker_output=str(self.f.f.root/'never-output'),resource_attestation=str(self.f.f.root/'never-attestation'))
        argv=['gpu_bootstrap.py']
        for key,value in values.items():argv.extend(['--'+key.replace('_','-'),value])
        read_text=Path.read_text;real_import=builtins.__import__
        def read(path,*args,**kwargs):
            return 'synthetic-boot'if str(path)=='/proc/sys/kernel/random/boot_id'else read_text(path,*args,**kwargs)
        def no_torch(name,*args,**kwargs):
            if name=='torch' or name.startswith('torch.'):raise AssertionError('native rejection must occur before torch import')
            return real_import(name,*args,**kwargs)
        with patch.object(sys,'argv',argv),patch.dict(os.environ,PILOT_NONCE='synthetic-nonce',CUDA_VISIBLE_DEVICES='CPU-no-GPU'),patch.object(bootstrap,'verify_bundle'),patch.object(bootstrap,'sha',side_effect=lambda path:self.f.index['sha256']if str(path)==self.f.index['path']else'a'*64),patch.object(bootstrap,'process_start',return_value='synthetic-start'),patch.object(bootstrap.time,'monotonic',return_value=5),patch.object(Path,'read_text',read),patch.object(bootstrap,'verify_model_stats')as model,patch('builtins.__import__',side_effect=no_torch):
            with self.assertRaisesRegex(ValueError,'ordinary v1 index'):bootstrap.main()
            model.assert_not_called()

    def test_comparison_flags_cannot_hide_a_real_text_difference(self):
        comparison=A.pinned(self.f.speed)
        comparison['rows'][0]['windows'][0]['observations']['candidate']['ASR']['normalized_text']='different real text'
        self.rewrite('comparison',comparison)
        q=copy.deepcopy(self.f.quality);q['comparison']=self.f.selected['comparison'];self.f.selected['quality_review']=self.f.put('changed-quality2.json',q);self.f.save()
        with self.assertRaisesRegex(ValueError,'not from sealed actual sidecars|flag hides actual transcript'):self.check()


    def rebind_speed_quality(self,value):
        self.f.selected['comparison']=self.f.put('changed-speed-real.json',value)
        q=copy.deepcopy(self.f.quality);q['comparison']=self.f.selected['comparison']
        self.f.selected['quality_review']=self.f.put('changed-speed-quality.json',q);self.f.save()

    def test_two_sides_same_fake_geometry_units_vad_still_refused(self):
        value=A.pinned(self.f.speed)
        for row in value['rows']:
            for label in ('baseline','candidate'):
                obs=row['complete_source_observations'][label]
                for key in ('units','decision','quality_axes','vad'):obs[key]={'fake_equal_value':True}
            for window in row['windows']:
                for label in ('baseline','candidate'):window['observations'][label]['alignment']={'items':[{'text':'fake','start_time':17,'end_time':19}]}
                window['alignment']={'items_equal':True}
        self.rebind_speed_quality(value)
        with self.assertRaisesRegex(ValueError,'not from sealed actual sidecars'):self.check()

    def test_self_reported_huge_speed_cannot_replace_actual_parent_elapsed(self):
        value=A.pinned(self.f.speed);value['cold_end_to_end_speed_ratio']=1e6;self.rebind_speed_quality(value)
        with self.assertRaisesRegex(ValueError,'actual parent elapsed'):self.check()

    def test_performance_and_readback_cannot_jointly_invent_faster_parent(self):
        value=A.pinned(self.f.speed);value['candidate_performance']['parent_wall_seconds']=.00001;value['cold_end_to_end_speed_ratio']=1e6
        proof=A.pinned(self.f.readback['main']);proof['performance']=value['candidate_performance'];self.f.selected['native_readbacks']['main']=self.f.put('fake-fast-readback.json',proof);self.rebind_speed_quality(value)
        with self.assertRaisesRegex(ValueError,'actual parent and worker terminal'):self.check()

    def test_dropped_boundary_flags_cannot_vacuously_look_equal(self):
        value=A.pinned(self.f.speed);value['rows'][0]['windows'][0]['alignment']={};self.rebind_speed_quality(value)
        with self.assertRaisesRegex(ValueError,'not from sealed actual sidecars'):self.check()

    def test_comparison_counts_must_be_recomputed_from_true_rows(self):
        value=A.pinned(self.f.speed);value['counts']['sources']=999;self.rebind_speed_quality(value)
        with self.assertRaisesRegex(ValueError,'aggregate differs'):self.check()

    def test_sealing_is_in_memory_and_startup_never_reopens_finite_sidecars(self):
        forbidden={ref['sidecar']['path']for role in self.f.readback.values()for ref in A.pinned(role)['sources']}
        original=Path.open
        def opened(path,*args,**kwargs):
            if str(path)in forbidden or str(path).endswith('.annotation.json.commit.json'):raise AssertionError('startup must not repeat finite sidecar readback')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',opened):self.check()

    def test_old_v1_unbound_readback_is_explicitly_refused(self):
        proof=A.pinned(self.f.readback['main']);proof['schema']='annotation-native-finite-readback/v1'
        self.f.selected['native_readbacks']['main']=self.f.put('old-unbound-readback.json',proof);self.f.save()
        with self.assertRaisesRegex(ValueError,'old unbound readback'):self.check()

    def test_real_window_source_clock_change_is_bound_to_view_digest(self):
        proof=A.pinned(self.f.readback['main']);proof['sources'][0]['record_view']['windows'][0]['owned_start_sample']=1
        self.f.selected['native_readbacks']['main']=self.f.put('bad-sourceclock-readback.json',proof);self.f.save()
        with self.assertRaisesRegex(ValueError,'sealed native source identity'):self.check()

    def code_preparation_fixture(self):
        root=self.f.f.root/'actual-pilot-code';root.mkdir()
        c=copy.deepcopy(self.f.c);c['purpose']='pilot_validation';c['version']='selected-native-pilot'
        members={}
        for name in set(C.engine_modules_for_contract(c))|{'gpu_launch.py','gpu_bootstrap.py','gpu_common.py','native_batch_admission.py'}:
            path=root/name;path.write_text('# CPU synthetic fixed source '+name);members[name]=C.sha_file(path)
        c['engine_source_files']={name:members[name]for name in C.engine_modules_for_contract(c)}
        bp=self.f.put(str((root/'bundle_files.json').relative_to(self.f.f.root)),members)
        cp=self.f.put('actual-pilot-contract-data.json',c)
        return root,bp,cp,c

    def test_formal_code_copy_only_pinned_pilot_and_four_helpers(self):
        root,bp,cp,c=self.code_preparation_fixture();before={p.name:p.read_bytes()for p in root.iterdir()}
        replacement=self.f.f.root/'new-admission.py';replacement.write_text('# CPU-only reviewed helper candidate')
        overlay=self.f.put('overlay.json',dict(schema='annotation-native-cpu-helper-overlay/v2',files={'native_batch_admission.py':dict(path=str(replacement),sha256=C.sha_file(replacement))}))
        output=self.f.f.root/'prepared-formal-code';value=A.prepare_formal_code(root,bp,cp,output,overlay_pin=overlay)
        self.assertEqual(set(value['changed_control_sources']),{'native_batch_admission.py'});self.assertFalse(value['formal_runtime_admitted']);self.assertFalse(value['independent_review_complete'])
        self.assertEqual({p.name:p.read_bytes()for p in root.iterdir()},before)
        self.assertTrue(all(C.sha_file(output/name)==digest for name,digest in c['engine_source_files'].items()))

    def test_gpu_parent_or_math_overlay_is_never_cpu_reviewable(self):
        root,bp,cp,c=self.code_preparation_fixture();replace=self.f.f.root/'foreign.py';replace.write_text('# forbidden GPU change')
        overlay=self.f.put('forbidden-overlay.json',dict(schema='annotation-native-cpu-helper-overlay/v2',files={'gpu_launch.py':dict(path=str(replace),sha256=C.sha_file(replace))}))
        with self.assertRaisesRegex(ValueError,'only four explicitly'):A.prepare_formal_code(root,bp,cp,self.f.f.root/'wrong-formal-code',overlay_pin=overlay)


    def test_resigned_sealed_view_and_aggregate_cannot_change_committed_geometry(self):
        for role in ('baseline','main','recovery'):
            pin=self.f.selected['native_readbacks'][role];proof=A.pinned(pin)
            for source in proof['sources']:
                for window in source['record_view']['windows']:
                    window['alignment']['native_timestamps_ms']=[999000,1000000];window['alignment']['fixed_timestamps_ms']=[999000,1000000]
                source['record_view_sha256']=C.digest(source['record_view'])
            proof['records_sha256']='f'*64
            self.f.selected['native_readbacks'][role]=self.f.put('resigned-'+role+'.json',proof)
        self.f.save()
        with self.assertRaisesRegex(ValueError,'original committed sidecar bytes'):self.check()

    def test_real_current_auditor_load_track_then_seal_adaptor(self):
        case=NativeFixtureModule.NativeFullReadbackTests();case.setUp();self.addCleanup(case.doCleanups)
        track=NativeFixtureModule.A.load_track(case.f.specpath,case.sha)
        self.assertNotIn('receipts',track)
        sealed=B.seal_admission_readback(track)
        self.assertEqual(sealed['additional_small_control_reads_for_seal'],1)
        self.assertEqual(sealed['schema'],B.SEALED_SCHEMA)
        pin=self.f.put('actual-auditor-sealed.json',sealed)
        value=B.verify_readback_receipt(pin,track['spec_pin'],track['spec'],track['contract'],sealed['bundle_sha256'],1,'cpu_fixture')
        self.assertEqual(value['records'],track['records'])
        # 真冻结auditor用Counter，只出现实际状态；不能让手造fixture的零键掩盖差异。
        self.assertEqual(value['track']['decision_counts'],track['decision_counts'])
        self.assertTrue(all(type(count)is int and count>0 for count in value['track']['decision_counts'].values()))

    def test_seal_cli_with_real_finite_auditor_and_no_models(self):
        import subprocess,shutil
        case=NativeFixtureModule.NativeFullReadbackTests();case.setUp();self.addCleanup(case.doCleanups)
        audit=self.f.f.root/'audit-source';audit.mkdir()
        names={'audit_native_transformers_batch.py','audit_speed_comparison.py','native_transformers_batch.py','contracts.py','schema.py','normalization.py','source_clock.py','quality_axes.py','readonly_progress.py','work_queue.py','source_registry.py','gpu_launch.py'}
        members={}
        for name in names:shutil.copyfile(HERE/name,audit/name);members[name]=C.sha_file(audit/name)
        bundle=self.f.put(str((audit/'bundle_files.json').relative_to(self.f.f.root)),members)
        output=self.f.f.root/'cli-sealed.json'
        done=subprocess.run([sys.executable,'-B',str(HERE/'native_batch_readback.py'),'--track',str(case.f.specpath),'--track-sha',case.sha,'--audit-code-root',str(audit),'--audit-bundle-sha',bundle['sha256'],'--output',str(output)],capture_output=True,text=True)
        self.assertEqual(done.returncode,0,done.stderr);self.assertEqual(A.pinned(dict(path=str(output),sha256=C.sha_file(output)))['schema'],B.SEALED_SCHEMA)



if __name__=='__main__':unittest.main()
