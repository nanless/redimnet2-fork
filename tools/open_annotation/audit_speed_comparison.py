"""只读比较固定输入的转写提速试验和断点恢复，不能授予人工质量资格。

输入是已结束的有限验证任务。只写明确指定的新报告，不创建队列、不加载模型。
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import sys

# 与既有生产脚本相同，明确从本文件同目录加载只读校验模块。
# 不从旧服务器目录或当前工作目录寻找唯一依赖。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contracts import atomic_new, digest, job_key, need, safe_join, utcnow, model_provenance
from readonly_progress import inspect_progress, read_small, strict_json, SLOT_BYTES
from work_queue import decode_completion_slot
from schema import assert_valid_record

SPEC_SCHEMA = 'redimnet2-annotation-speed-track/v1'
MAX_SOURCES = 20000
DECISIONS = ('candidate', 'unknown', 'failed')
WINDOW_INPUTS = ('window_id', 'context_start_sample', 'context_end_sample', 'owned_start_sample',
                 'owned_end_sample', 'input_pcm_sha256', 'model_input_frames', 'model_sample_rate',
                 'resampling_policy', 'time_mapping')
SOURCE_INPUTS = ('decode_status', 'file_sha256', 'pcm_sha256', 'pcm_hash_policy', 'sample_rate', 'channels', 'frames')


def bound(path, expected=None, maximum=64 * 1024**2):
    path = Path(path).absolute()
    need(all(not p.is_symlink() for p in (path, *path.parents)), 'input ancestor symlink refused: ' + str(path))
    value, raw = read_small(path, maximum)
    sha = hashlib.sha256(raw).hexdigest()
    need(expected is None or sha == expected, 'input SHA differs: ' + str(path))
    return value, sha, len(raw)


def pin(spec, key):
    value = spec[key]
    need(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'explicit path/SHA pin required: ' + key)
    need(Path(value['path']).is_absolute(), 'absolute pinned path required')
    return bound(value['path'], value['sha256'])


def compare_contracts(a, b, mode):
    """模型、输入/时钟和字界设置不变；新数值策略只允许显式转写配置变化。"""
    for c in (a, b):
        need(c.get('schema') == 'redimnet2-production-contract/v2', 'unknown producer contract')
        need(c.get('model_API_calls_allowed') == 0 and not c.get('release_training_labels', False), 'model/API admission differs')
    if mode == 'recovery':
        need(a == b, 'recovery requires the exact same contract')
        return
    for key in ('models', 'source_files', 'alignment', 'decoder', 'quality_policy', 'engine_source_files'):
        need(key in a and a[key] == b.get(key), 'non-ASR producer identity differs: ' + key)
    ar, br = dict(a['runtime']), dict(b['runtime'])
    ar.pop('asr_compute', None); br.pop('asr_compute', None)
    need(ar == br, 'window, normalization, alignment or non-ASR runtime differs')
    need(ar.get('max_batch') == 1 and ar.get('execution_policy') == 'single_window_independent', 'this auditor requires single-window calls')


def _finite_seconds(value, label):
    need(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'invalid measured seconds: ' + label)
    return value


def compute_execution(metrics, contract):
    """配置读回与真实首窗dispatch分别核验；缺少观测就不提供已兑现结论。"""
    profile = contract['runtime'].get('asr_compute', {'dtype': 'float32', 'attention': 'eager'})
    if metrics.get('asr_rows', 0) == 0:
        return {'verified': False, 'reason': 'no_new_ASR_call_in_this_worker', 'requested': profile}
    loaded = metrics.get('loaded_compute'); observation = metrics.get('first_call_compute_observation')
    if not isinstance(loaded, dict) or not isinstance(observation, dict) or metrics.get('first_call_trace_completed') is not True:
        return {'verified': False, 'reason': 'actual_compute_observation_missing', 'requested': profile}
    for stage, expected in (('asr', profile), ('aligner', {'dtype': 'float32', 'attention': 'eager'})):
        value = loaded.get(stage, {})
        need(value.get('requested') == expected and set(value.get('floating_parameter_elements_by_dtype', {})) == {expected['dtype']},
             'actual loaded dtype/profile differs: ' + stage)
        need(all(type(v) is int and v > 0 for v in value['floating_parameter_elements_by_dtype'].values()), 'invalid parameter element count')
        if stage == 'asr' and 'asr_compute' in contract['runtime']:
            configs = value.get('attention_configurations', [])
            need(configs and all(v['attention_config'] == expected['attention'] for v in configs), 'actual nested ASR attention config differs')
    need(observation.get('attention_configurations_stable') is True, 'attention configurations changed during actual forwards')
    forward = observation.get('attention_forwards', {}); sdpa = observation.get('functional_sdpa_calls', {})
    need(forward and all(type(v) is int and v > 0 for v in forward.values()), 'actual ASR attention forwards missing')
    need(all(type(v) is int and v >= 0 for v in sdpa.values()), 'invalid actual SDPA call counts')
    explicit = 'asr_compute' in contract['runtime']
    if explicit:
        if profile['attention'] == 'sdpa':
            need(all(sdpa.get(k, 0) == v for k, v in forward.items()), 'requested SDPA not used by actual ASR attention forwards')
            need(observation.get('functional_sdpa_qkv_dtypes_stable') is True and all(
                observation.get('functional_sdpa_qkv_dtypes', {}).get(k) == [profile['dtype']] * 3 for k in forward),
                'actual SDPA QKV tensor dtype differs from explicit ASR profile')
        else: need(sum(sdpa.values()) == 0, 'requested eager observed actual SDPA calls')
    else:
        # Legacy只说明实际发生的逐模块配置，不能按顶层eager推断子模块。
        actual = observation.get('attention_module_configurations', {})
        if not all(actual.get(k) in {'eager', 'sdpa'} for k in forward):
            return {'verified': False, 'reason': 'legacy_attention_module_config_missing_or_unsupported',
                    'profile_declared': False, 'observation': observation}
        for key, count in forward.items():
            expected_calls = count if actual[key] == 'sdpa' else 0
            need(sdpa.get(key, 0) == expected_calls, 'legacy actual attention config/dispatch differs: ' + key)
        need(not any(k not in forward and v for k,v in sdpa.items()), 'legacy SDPA calls lack an observed module')
    return {'verified': True, 'reason': 'explicit_profile_actual_parameter_config_and_forward_observed'
            if explicit else 'legacy_actual_per_module_config_and_dispatch_observed',
            'profile_declared': explicit, 'requested': profile if explicit else None,
            'observed_parameter_dtype': profile['dtype'],
            'observed_nonuniform_attention_dispatch': len(set(observation.get('attention_module_configurations', {}).values())) > 1,
            'observation': observation, 'CUDA_kernel_trace_performed': False}


def summarize_workers(parent, resource_root, expected_receipts, contract):
    """核实际worker终态摘要与发布集合；资源归还不代表此刻仍空闲。"""
    need(parent.get('resource_returned') is True and parent.get('full_sidecar_readback') is True,
         'run has no successful terminal publication/resource return')
    parts = parent.get('workers', [])
    need(parts and len(parts) == parent.get('worker_count'), 'worker terminal denominator differs')
    seen_workers, seen_sources = set(), set(); sums = Counter(); workers = []
    for part in parts:
        wid = part['worker_id']; need(wid not in seen_workers, 'duplicate worker'); seen_workers.add(wid)
        summary, summary_sha, _ = bound(safe_join(resource_root, wid + '/worker_summary.json'))
        terminal = part['terminal']
        need(summary == terminal['summary'] and summary_sha == terminal['summary_sha256'], 'parent/worker summary differs')
        ready, ready_sha, _ = bound(safe_join(resource_root, wid + '/terminal_ready.json'))
        ack, _, _ = bound(safe_join(resource_root, wid + '/terminal_ack.json'))
        need(ready['summary_sha256'] == summary_sha and terminal['ready_sha256'] == ready_sha == ack['ready_sha256'],
             'worker ready/ACK digest differs')
        need(ready['nonce'] == ack['nonce'] and summary['worker_id'] == wid, 'worker nonce/ID differs')
        need(ready['bundle_sha256'] == parent['bundle_sha256'], 'worker bundle identity differs')
        for key in ('contract_sha256', 'index_sha256'):
            need(summary[key] == parent[key] == ready[key], 'worker input identity differs: ' + key)
        need(summary['device_uuid'] == part['uuid'], 'actual worker device differs')
        if contract.get('purpose') != 'cpu_fixture':
            att, att_sha, _ = bound(safe_join(resource_root, wid + '/resource_attestation.json'))
            challenge, challenge_sha, _ = bound(safe_join(resource_root, wid + '/context_challenge/attestation.json'))
            need(att_sha == ready['resource_attestation_sha256'] and att['challenge_attestation_sha'] == challenge_sha,
                 'resource challenge/ready digest differs')
            need(att['worker_id'] == wid and att['nonce'] == ready['nonce'] and att['UUIDs'] == [part['uuid']]
                 and att['contract_sha256'] == parent['contract_sha256'] and att['index_sha256'] == parent['index_sha256']
                 and att['bundle_sha256'] == parent['bundle_sha256'], 'active resource identity differs')
            need(challenge['complete'] is True and challenge['all_free_phases_returned_to_baseline'] is True
                 and len(challenge['phases']) == 7 and challenge['UUIDs'] == [part['uuid']]
                 and challenge['known_container_worker_pid'] == att['known_container_worker_pid'] == ready['worker_pid'],
                 'saved seven-phase resource context proof differs')
        metrics = summary['backend_metrics']
        need(summary['model_API_calls'] == metrics['model_API_calls'] == 0, 'nonlocal model calls observed')
        source_ids = set(); fresh_count = 0
        for ref in summary['chunk_receipt_refs']:
            item = expected_receipts[ref['id']]
            need(ref['sha256'] == item['sha256'] and ref['source_count'] == item['value']['source_count'], 'worker chunk binding differs')
            ids = {r['source_id'] for r in item['value']['sources']}
            need(not source_ids & ids and not seen_sources & ids, 'worker source ownership overlap')
            source_ids.update(ids); fresh_count += len(item['fresh_source_ids'])
        seen_sources.update(source_ids)
        need(len(source_ids) == summary['assigned_source_rows'] == summary['sidecars_committed'] + summary['sidecar_cache_hits'],
             'new/whole-sidecar reuse denominator differs')
        need(fresh_count == summary['sidecars_committed'], 'new sidecar count differs from actual publication attempts')
        for key in ('assigned_source_rows', 'sidecars_committed', 'sidecar_cache_hits', 'model_windows', 'decoded_audio_seconds'):
            sums[key] += summary[key]
        for key in ('asr_rows', 'aligner_rows', 'asr_batches', 'aligner_batches', 'asr_seconds', 'aligner_seconds', 'model_load_seconds'):
            need(key in metrics, 'missing actual backend metric: ' + key)
            if key.endswith('seconds'): _finite_seconds(metrics[key], key)
            else: need(type(metrics[key]) is int and metrics[key] >= 0, 'invalid actual invocation count')
            sums[key] += metrics[key]
        for stage in ('asr', 'aligner'):
            need(set(metrics[stage + '_call_rows']) <= {'1'} and metrics[stage + '_rows'] == metrics[stage + '_batches']
                 == metrics[stage + '_call_rows'].get('1', 0), 'actual calls are not independent single rows')
            sums[stage + '_stage_cache_hits'] += summary['stage_cache_hits'][stage]
        sums['worker_elapsed_seconds'] += _finite_seconds(summary['elapsed_seconds'], 'worker elapsed')
        workers.append({'worker_id': wid, 'summary_sha256': summary_sha, 'assigned_sources': len(source_ids),
                        'new_sidecars': summary['sidecars_committed'], 'whole_sidecar_reuse': summary['sidecar_cache_hits'],
                        'backend_metrics': metrics, 'compute_execution': compute_execution(metrics, contract),
                        'worker_elapsed_seconds': summary['elapsed_seconds']})
    all_expected = {r['source_id'] for item in expected_receipts.values() for r in item['value']['sources']}
    need(seen_sources == all_expected, 'worker publication union is incomplete')
    need(len({p['uuid'] for p in parts}) == len(parts), 'worker GPU identities repeat')
    if contract.get('purpose') != 'cpu_fixture':
        returned, _, _ = bound(Path(resource_root) / 'resource_return_double_probe.json')
        need(len(returned['snapshots']) == 2 and returned['interval_seconds'] >= 5, 'saved resource return double probe missing')
        for snapshot in returned['snapshots']:
            need(not snapshot['compute_processes'] and {d['uuid'] for d in snapshot['devices']} == {p['uuid'] for p in parts}
                 and all(d['used_mib'] <= 256 for d in snapshot['devices']), 'saved resource return was not idle')
    wall = _finite_seconds(parent['parent_elapsed_seconds'], 'parent elapsed')
    need(wall > 0, 'zero elapsed cannot establish throughput')
    return {'workers': workers, 'totals': dict(sums), 'parent_wall_seconds': wall,
            'gpu_lease_wall_seconds': _finite_seconds(parent['gpu_lease_wall_seconds'], 'GPU lease wall'),
            'all_assigned_sources_per_parent_second': len(all_expected) / wall,
            'new_sidecars_per_parent_second': sums['sidecars_committed'] / wall,
            'all_sources_recomputed_cold': sums['sidecar_cache_hits'] == sums['asr_stage_cache_hits'] == sums['aligner_stage_cache_hits'] == 0,
            'resource_returned_at_saved_run_end': True, 'current_GPU_availability_checked': False,
            'historical_resource_digest_chain_verified': contract.get('purpose') != 'cpu_fixture',
            'actual_compute_execution_verified': bool(sums['asr_rows'] > 0 and all(w['compute_execution']['verified']
                for w in workers if w['backend_metrics']['asr_rows'] > 0))}


def load_track(spec_path, expected_sha):
    spec, spec_sha, _ = bound(spec_path, expected_sha)
    need(spec.get('schema') == SPEC_SCHEMA, 'wrong track spec schema')
    contract, contract_sha, _ = pin(spec, 'contract'); index, index_sha, _ = pin(spec, 'index')
    need(type(index.get('source_count')) is int and 0 < index['source_count'] <= MAX_SOURCES,
         'finite validation source limit exceeded; this is not a full-library auditor')
    for key in ('queue_root', 'output_root', 'resources_root'):
        need(Path(spec[key]).is_absolute(), 'absolute track directory required: ' + key)
    queue, output, resource = (Path(spec[k]) for k in ('queue_root', 'output_root', 'resources_root'))
    progress = inspect_progress(spec['index']['path'], index_sha, queue, contract_sha)
    need(progress['committed_source_lower_bound'] == index['source_count'] and not progress['snapshot_leased_chunks_excluded'],
         'track is incomplete; running chunks cannot be silently omitted')
    tree = safe_join(output, contract['version'])
    identity, _, _ = bound(tree / 'identity.json')
    need(identity == {'schema': 'redimnet2-sidecar-tree/v1', 'version': contract['version'], 'contract_sha256': contract_sha},
         'versioned sidecar identity differs')
    records, receipts, decisions, audit_bytes = {}, {}, Counter(), 0
    with (queue / 'completion_map.bin').open('rb') as completion:
        for shard in index['shards']:
            slot = decode_completion_slot(completion.read(SLOT_BYTES))
            chunk, chunk_sha, size = bound(queue / 'completed' / (shard['id'] + '.json'), slot['commit_sha256']); audit_bytes += size
            need(chunk['source_count'] == shard['count'] and chunk['shard_id'] == shard['id']
                 and chunk['shard_sha256'] == shard['sha256'] and chunk['index_sha256'] == index_sha
                 and chunk['contract_sha256'] == contract_sha, 'committed chunk input identity differs')
            need(digest(sorted(r['source_id'] for r in chunk['sources'])) == slot['source_ids_sha256'], 'slot source IDs differ')
            receipt_by_id = {r['source_id']: r for r in chunk['sources']}
            need(len(receipt_by_id) == len(chunk['sources']) == shard['count'], 'receipt source denominator differs')
            shard_path = safe_join(Path(spec['index']['path']).parent, shard['path'])
            raw = shard_path.read_bytes(); need(hashlib.sha256(raw).hexdigest() == shard['sha256'], 'input shard changed')
            sources = [strict_json(line) for line in raw.splitlines() if line]
            need(len(sources) == shard['count'] and {s['source_id'] for s in sources} == set(receipt_by_id), 'source shard/receipt union differs')
            local_counts = Counter(); fresh_ids = set()
            for source in sources:
                sid = source['source_id']; need(sid not in records, 'duplicate source in validation set')
                receipt = receipt_by_id[sid]; path = safe_join(tree, source['sidecar_relpath'])
                record, record_sha, size = bound(path, receipt['sidecar_sha256']); audit_bytes += size
                assert_valid_record(record)
                need(record['source'] == source and record['contract']['sha256'] == contract_sha
                     and record['execution']['job_key'] == receipt['job_key'] == job_key(source, contract_sha), 'sidecar source/contract/job differs')
                need(record['provenance'].get('runtime') == contract['runtime'], 'saved effective runtime differs from track contract')
                need(record['provenance'].get('models') == model_provenance(contract), 'saved model identity differs from producer contract')
                if record['execution']['attempt_id'] == chunk['attempt_id']: fresh_ids.add(sid)
                need(not record['decision']['usable_for_training'] and not record['decision']['usable_for_evaluation'], 'automatic eligibility release refused')
                commit, commit_sha, _ = bound(path.with_name(path.name + '.commit.json'), receipt['commit_sha256'])
                need(commit == {'schema': 'redimnet2-sidecar-commit/v1', 'source_id': sid, 'job_key': receipt['job_key'],
                     'contract_sha256': contract_sha, 'source_record_sha256': digest(source), 'sidecar_sha256': record_sha, 'bytes': size}, 'per-source commit differs')
                need(receipt['sidecar_bytes'] == size and receipt['decision'] == record['decision']['state'], 'receipt byte/decision differs')
                local_counts[record['decision']['state']] += 1
                records[sid] = record
            need({k: local_counts[k] for k in DECISIONS} == slot['decision_counts'], 'sidecar/completion decision counts differ')
            decisions.update(local_counts); receipts[shard['id']] = {'sha256': chunk_sha, 'value': chunk, 'fresh_source_ids': fresh_ids}
    need(len(records) == index['source_count'], 'complete source denominator differs')
    parent, parent_sha, _ = bound(resource / 'parent_result.json')
    need(parent['contract_sha256'] == contract_sha and parent['index_sha256'] == index_sha
         and parent.get('model_API_calls') == 0, 'parent terminal identity differs')
    for record in records.values():
        need(record['provenance']['code_sha256'] == parent['bundle_sha256'], 'source producer bundle differs from this track; separate reused producer identity explicitly')
    performance = summarize_workers(parent, resource, receipts, contract)
    need(performance['totals']['assigned_source_rows'] == len(records), 'worker source denominator differs')
    return {'spec': spec, 'spec_sha256': spec_sha, 'contract': contract, 'contract_sha256': contract_sha,
            'index_sha256': index_sha, 'records': records, 'decision_counts': dict(decisions),
            'performance': performance, 'parent_result_sha256': parent_sha, 'sidecar_bytes_read': audit_bytes}


def compare_record(a, b, mode='speed'):
    """差异是模型观察变化，不是相对于人审的错误；失败样本也返回完整记录。"""
    need(a['source'] == b['source'], 'source record changed between tracks')
    sid = a['source']['source_id']; result = {'source_id': sid, 'changed_fields': [], 'windows': [],
        'source_identity_equal': all(a['source_identity'].get(k) == b['source_identity'].get(k) for k in SOURCE_INPUTS),
        'decoded_audio_identity_comparable': all(r['source_identity'].get('decode_status') == 'decoded_complete'
            and r['source_identity'].get('file_sha256') and r['source_identity'].get('pcm_sha256') for r in (a,b)),
        'baseline_state': a['decision']['state'], 'candidate_state': b['decision']['state']}
    for field in ('units', 'decision', 'quality_axes', 'errors', 'vad'):
        if a.get(field) != b.get(field): result['changed_fields'].append(field)
    if not result['source_identity_equal']: result['changed_fields'].append('source_identity')
    if mode == 'recovery':
        for field in ('source_identity', 'provenance', 'contract'):
            if a.get(field) != b.get(field): result['changed_fields'].append('complete_' + field)
    aw = {w['window_id']: w for w in a.get('windows', [])}; bw = {w['window_id']: w for w in b.get('windows', [])}
    need(len(aw) == len(a.get('windows', [])) and len(bw) == len(b.get('windows', [])), 'duplicate window ID')
    if set(aw) != set(bw): result['changed_fields'].append('window_set')
    result['baseline_window_count'], result['candidate_window_count'] = len(aw), len(bw)
    for wid in sorted(set(aw) | set(bw)):
        left, right = aw.get(wid), bw.get(wid)
        row = {'window_id': wid, 'both_present': left is not None and right is not None}
        if not row['both_present']:
            row['comparison_status'] = 'missing_window_retained_in_denominator'; result['windows'].append(row); continue
        row['input_identity_equal'] = all(left.get(k) == right.get(k) for k in WINDOW_INPUTS)
        if not row['input_identity_equal']: result['changed_fields'].append('window_input:' + wid)
        row['ASR'] = {}; aa, bb = left.get('asr', {}), right.get('asr', {})
        for key in ('status', 'raw_text', 'normalized_text', 'language', 'normalization_audit'):
            row['ASR'][key + '_equal'] = aa.get(key) == bb.get(key)
        for key in ('generated_token_ids', 'eos_observed', 'first_eos_index', 'max_new_tokens'):
            row['ASR'][key + '_equal'] = aa.get('native', {}).get(key) == bb.get('native', {}).get(key)
        row['ASR']['baseline_status'], row['ASR']['candidate_status'] = aa.get('status'), bb.get('status')
        row['observations'] = {}
        for label, original in (('baseline', left), ('candidate', right)):
            av, lv = original.get('asr', {}), original.get('alignment', {})
            row['observations'][label] = {'ASR': {k: av.get(k) for k in ('status', 'raw_text', 'normalized_text', 'language')},
                'ASR_native': {k: av.get('native', {}).get(k) for k in ('generated_token_ids', 'eos_observed', 'first_eos_index', 'max_new_tokens')},
                'alignment': {k: lv.get(k) for k in ('status', 'target_text', 'encoded_target_words', 'native_timestamps_ms',
                    'fixed_timestamps_ms', 'repair_changed', 'author_final_items', 'items')}}
        al, bl = left.get('alignment', {}), right.get('alignment', {})
        row['alignment'] = {key + '_equal': al.get(key) == bl.get(key) for key in
                           ('status', 'target_text', 'encoded_target_words', 'native_timestamps_ms',
                            'fixed_timestamps_ms', 'repair_changed', 'author_final_items', 'items')}
        words_a = al.get('encoded_target_words', [v.get('text') for v in al.get('items', [])])
        words_b = bl.get('encoded_target_words', [v.get('text') for v in bl.get('items', [])])
        same_target = bool(words_a) and words_a == words_b and al.get('target_text') == bl.get('target_text')
        row['alignment']['endpoint_comparison_status'] = 'unmeasured_target_or_geometry_differs'
        if same_target:
            for key in ('native_timestamps_ms', 'fixed_timestamps_ms'):
                v, w = al.get(key), bl.get(key)
                if isinstance(v,list) and isinstance(w,list) and len(v)==len(w) and v and all(type(x) in (int,float) and math.isfinite(x) for x in v+w):
                    row['alignment'][key + '_max_absolute_delta_ms'] = max(abs(x-y) for x,y in zip(v,w))
                    row['alignment']['endpoint_comparison_status'] = 'same_native_target_items'
        if mode == 'recovery':
            # Per-call execution timing is operational telemetry, not model truth.
            omit = {'asr_cache_hit', 'aligner_cache_hit'}
            row['complete_native_window_equal_except_cache_flags'] = ({k:v for k,v in left.items() if k not in omit}
                                                                      == {k:v for k,v in right.items() if k not in omit})
        row['baseline_cache_hits'] = {k: left.get(k) for k in ('asr_cache_hit', 'aligner_cache_hit')}
        row['candidate_cache_hits'] = {k: right.get(k) for k in ('asr_cache_hit', 'aligner_cache_hit')}
        row['comparison_status'] = 'same_window_inputs' if row['input_identity_equal'] else 'input_mismatch_not_a_model_difference'
        result['windows'].append(row)
    return result


def analyze(baseline, candidate, mode='speed'):
    need(mode in {'speed','recovery'}, 'unknown comparison mode')
    compare_contracts(baseline['contract'], candidate['contract'], mode)
    left, right = baseline['records'], candidate['records']
    need(set(right) <= set(left) and (mode == 'recovery' or set(left) == set(right)), 'fixed input source union differs')
    rows = [compare_record(left[sid], right[sid], mode) for sid in sorted(right)]
    counts = Counter(); max_delta = {'native_timestamps_ms': None, 'fixed_timestamps_ms': None}; comparable = 0
    recovery_equal = True
    for row in rows:
        counts['sources'] += 1; counts['source_identity_changed'] += not row['source_identity_equal']
        comparable += bool(row['source_identity_equal'] and row['decoded_audio_identity_comparable'])
        counts['decision_changed_sources'] += row['baseline_state'] != row['candidate_state']
        recovery_equal &= not row['changed_fields'] and row['source_identity_equal'] and row['baseline_window_count'] == row['candidate_window_count']
        for w in row['windows']:
            counts['planned_window_union'] += 1
            if not w['both_present']: counts['missing_windows'] += 1; recovery_equal = False; continue
            counts['input_mismatch_windows'] += not w['input_identity_equal']
            for key,value in w['ASR'].items():
                if key.endswith('_equal'): counts['ASR_' + key.removesuffix('_equal') + '_changed_windows'] += not value
            for key,value in w['alignment'].items():
                if key.endswith('_equal'): counts['alignment_' + key.removesuffix('_equal') + '_changed_windows'] += not value
            for key in max_delta:
                value = w['alignment'].get(key + '_max_absolute_delta_ms')
                if value is not None:
                    max_delta[key] = value if max_delta[key] is None else max(max_delta[key], value)
                    counts[key + '_comparable_windows'] += 1
            if mode == 'recovery': recovery_equal &= w['complete_native_window_equal_except_cache_flags']
    pa, pb = baseline['performance'], candidate['performance']
    cold_speed_valid = bool(mode == 'speed' and baseline['contract'].get('purpose') != 'cpu_fixture' and candidate['contract'].get('purpose') != 'cpu_fixture' and pa['all_sources_recomputed_cold'] and pb['all_sources_recomputed_cold']
                            and pa.get('actual_compute_execution_verified') and pb.get('actual_compute_execution_verified')
                            and pa['totals']['asr_rows'] > 0 and pb['totals']['asr_rows'] > 0
                            and comparable == len(rows) and not counts['missing_windows'] and not counts['input_mismatch_windows'])
    return {'schema': 'redimnet2-annotation-speed-comparison/v1', 'recorded_utc': utcnow(), 'mode': mode,
            'source_count': len(rows), 'baseline_total_sources': len(left), 'candidate_total_sources': len(right),
            'subset_of_baseline_explicit_in_recovery_mode': mode == 'recovery' and set(left) != set(right),
            'counts': dict(counts), 'decoded_audio_identity_comparable_sources': comparable,
            'maximum_same_target_endpoint_delta_ms': max_delta, 'baseline_performance': pa, 'candidate_performance': pb,
            'cold_end_to_end_speed_ratio': pa['parent_wall_seconds'] / pb['parent_wall_seconds'] if cold_speed_valid else None,
            'cold_speed_ratio_valid': cold_speed_valid,
            'strict_recovery_native_results_equal': bool(recovery_equal) if mode == 'recovery' else None,
            'baseline_decision_counts': baseline['decision_counts'], 'candidate_decision_counts': candidate['decision_counts'],
            'baseline_spec_sha256': baseline['spec_sha256'], 'candidate_spec_sha256': candidate['spec_sha256'],
            'baseline_contract_sha256': baseline['contract_sha256'], 'candidate_contract_sha256': candidate['contract_sha256'],
            'all_planned_sources_accounted': True, 'full_sidecar_commit_readback_verified': True,
            'source_audio_rehashed_in_this_audit': False, 'human_text_or_boundary_accuracy_certified': False,
            'training_or_evaluation_eligibility_released': False, 'formal_admission': False, 'changed_rows_are_accuracy_errors': False,
            'rows': rows, 'report_contains_private_source_ids_and_model_observations': True}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for prefix in ('baseline','candidate'):
        p.add_argument('--' + prefix + '-spec', required=True)
        p.add_argument('--' + prefix + '-spec-sha', required=True)
    p.add_argument('--mode', choices=('speed','recovery'), default='speed'); p.add_argument('--output', required=True)
    args = p.parse_args(argv); out = Path(args.output).absolute()
    need(out.parent.is_dir() and not out.exists() and not out.is_symlink(), 'report must be a new file in an existing directory')
    need(all(not v.is_symlink() for v in (out.parent,*out.parent.parents)), 'report ancestor symlink refused')
    a = load_track(args.baseline_spec, args.baseline_spec_sha); b = load_track(args.candidate_spec, args.candidate_spec_sha)
    for track in (a,b):
        for key in ('queue_root','output_root','resources_root'):
            root = Path(track['spec'][key]).resolve(); need(root != out and root not in out.parents, 'report must be outside producer trees')
    result = analyze(a,b,args.mode); atomic_new(out,result)
    return 2 if args.mode == 'recovery' and not result['strict_recovery_native_results_equal'] else 0


if __name__ == '__main__': raise SystemExit(main())
