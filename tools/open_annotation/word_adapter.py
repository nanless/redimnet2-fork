"""Read-only Ultra/v2 -> pending word-unit proposals; no training permission.

Default output is not word-complete-units/v1. Optional engineering export
requires complete physical-media binding and always keeps a non-train split.
"""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import traceback
import wave

BASE = Path(__file__).resolve().parent
DOMAINS = {'cnceleb2', 'voxblink2', 'voxceleb2'}
ENGINEERING_SPLIT = 'engineering_candidate_pending'
COUNTS = (1, 2, 3, 6)
REPOSITORY = next((p for p in BASE.parents if (p/'AGENTS.md').is_file() and (p/'redimnet2').is_dir()), None)


def need(ok, why):
    if not ok: raise ValueError(why)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path, expected=None):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink(), 'real absolute evidence file required')
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(identity(before) == identity(after), 'evidence changed during read')
    actual = hashlib.sha256(raw).hexdigest()
    if expected is not None: need(actual == expected, 'evidence SHA differs: ' + str(path))
    def pairs(rows):
        result = {}
        for key, value in rows:
            need(key not in result, 'duplicate JSON key'); result[key] = value
        return result
    data = json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    return data, raw, actual


def load_file_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_context(args):
    need(getattr(args, 'diagnostic_protection', None) and getattr(args, 'diagnostic_protection_sha', None), 'explicit external diagnostic protection path and SHA required')
    code = Path(args.producer_code).resolve()
    for name in ('contracts', 'normalization', 'quality_axes'):
        old = sys.modules.get(name)
        if old is not None:
            need(Path(old.__file__).resolve() == code / (name + '.py'), 'producer module namespace collision: ' + name)
    sys.path.insert(0, str(code))
    producer = load_file_module(code / 'schema.py', '_adapter_producer_schema')
    import contracts
    need(Path(contracts.__file__).resolve() == code / 'contracts.py', 'wrong actual producer contracts module')
    contract = contracts.read_contract(args.contract, args.contract_sha)
    need(contract['purpose'] in {'pilot_validation','silver_annotation'}, 'CPU fixtures cannot enter real-source adapter')
    bundle, _, _ = read_json(str(code / 'bundle_files.json'), args.producer_bundle_sha)
    for name, pin in bundle.items():
        rel = Path(name); need(not rel.is_absolute() and '..' not in rel.parts, 'unsafe bundle member')
        need(file_sha(code / rel) == pin, 'producer bundle member differs')
    need(set(contract['engine_source_files']) == set(contracts.ENGINE_MODULES), 'incomplete producer semantic pins')
    need(getattr(args, 'diagnostic_protection', None) and getattr(args, 'diagnostic_protection_sha', None), 'explicit external diagnostic protection path and SHA required')
    need(len(args.diagnostic_protection_sha) == 64 and all(c in '0123456789abcdef' for c in args.diagnostic_protection_sha), 'invalid protection SHA')
    protection_path = Path(args.diagnostic_protection).resolve()
    protection, _, protection_sha = read_json(str(protection_path), args.diagnostic_protection_sha)
    need(protection['schema'] == 'ultra-diagnostic-protection-reference/v1' and type(protection['case_count']) is int and protection['case_count'] > 0
         and len(protection['cases']) == protection['case_count']
         and sum(x['in_three_training_registry'] for x in protection['cases']) == protection['three_training_source_count'], 'incomplete diagnostic protection reference')
    need(all(x['benchmark_diagnostic_only_not_training_release'] is True for x in protection['cases']), 'invalid diagnostic protection policy')
    word_path = Path(args.word_schema).resolve()
    need(file_sha(word_path) == args.word_schema_sha, 'word loader source pin differs')
    word = load_file_module(word_path, '_adapter_word_schema')
    need(word.SCHEMA == 'word-complete-units/v1', 'wrong word consumer schema')
    return {'producer': producer, 'contracts': contracts, 'contract': contract,
            'contract_sha': args.contract_sha, 'bundle_sha': args.producer_bundle_sha,
            'word': word, 'word_schema_sha': args.word_schema_sha,
            'diagnostic_protection': protection, 'diagnostic_protection_sha256': protection_sha,
            'diagnostic_protection_content_sha256': digest(protection)}


def verify_publication(row, context):
    record, raw, actual = read_json(row['sidecar_path'], row['sidecar_sha256'])
    commit, commit_raw, commit_sha = read_json(row['commit_path'], row['commit_sha256'])
    context['producer'].assert_valid_record(record)
    c = context['contracts']; source = record['source']; producer_sha = context['contract_sha']
    need(record['schema'] == 'redimnet2-open-annotation/v2', 'only Ultra sidecar/v2 accepted')
    need(record['contract']['sha256'] == producer_sha, 'producer contract SHA differs')
    actual_contract = context['contract']
    need(record['contract']['purpose'] == actual_contract['purpose'] and record['contract']['version'] == actual_contract['version'], 'embedded producer purpose/version differs')
    prov = record['provenance']
    need(prov == record['contract']['provenance'], 'duplicated producer provenance differs')
    need(prov.get('models') == c.model_provenance(actual_contract) and prov.get('runtime') == actual_contract['runtime']
         and prov.get('alignment_identity') == actual_contract['alignment'], 'embedded producer models/runtime/alignment differ from actual pinned contract')
    need(record['execution']['backend'] == 'qwen3-asr-plus-ultra-no-head' and record['execution'].get('synthetic_fixture') is not True,
         'actual Ultra producer backend required, no CPU fixture')
    need(record['provenance']['alignment_method'] == 'ultra_no_head', 'not Ultra no-head')
    need(record['contract']['provenance']['code_sha256'] == context['bundle_sha'], 'producer code bundle differs')
    expected = {'schema': 'redimnet2-sidecar-commit/v1', 'source_id': source['source_id'],
                'job_key': c.job_key(source, producer_sha), 'contract_sha256': producer_sha,
                'source_record_sha256': c.digest(source), 'sidecar_sha256': actual, 'bytes': len(raw)}
    need(commit == expected and record['execution']['job_key'] == expected['job_key'], 'sidecar/commit/source/job binding differs')
    need(source['dataset'] in DOMAINS, 'outside current three-training-domain scope')
    need(source['original_splits'] == ['provider_train'], 'not canonical original provider_train')
    need(bool(source['roles']) and set(source['roles']) <= {'train_ptn', 'train_lm'}, 'foreign consumer role')
    need(source['protection']['training_forbidden'] is False and not source.get('split_conflicts'), 'protected or conflicted source')
    need(source['source_id'] == 'src_' + hashlib.sha256((source['dataset'] + '\0' + source['relative_path']).encode()).hexdigest(), 'source path identity differs')
    decision = record['decision']
    need(decision['usable_for_training'] is False and decision['usable_for_evaluation'] is False
         and decision['human_review'] is None, 'producer eligibility must remain false/pending')
    need(record.get('human_review') is None and record['quality_axes']['human_review'] is None, 'human review cannot be invented or promoted')
    return record, raw, commit_raw, actual, commit_sha


def word_template(record, selected, language, speaker_id, family_id, context):
    source, identity = record['source'], record['source_identity']; word = context['word']
    source_window = next(w for w in record['windows'] if w['window_id'] == selected[0]['window_id'])
    text = ''.join(unit['text'] for unit in selected) if language == 'zh' else ' '.join(unit['text'] for unit in selected)
    normalized = word.normalize_text(text, language); tokens = word.tokenize_units(normalized, language)
    start, end = selected[0]['start_sample'], selected[-1]['end_sample']
    result = {'audio_path': source['audio_path'], 'audio_realpath': None, 'audio_sha256': identity['file_sha256'],
              'audio_stat': None, 'sample_rate': identity['sample_rate'], 'source_num_samples': identity['frames'],
              'source_channels': identity['channels'], 'start_sample': start, 'end_sample': end,
              'speaker_id': speaker_id, 'speaker_verified_identity': False, 'text_original': text,
              'spoken_text': normalized, 'language': language, 'n_units': len(tokens), 'unit_texts': tokens,
              'count_bin': word.count_bin(len(tokens)),
              'counting_policy': 'han_characters_no_punctuation_v1' if language == 'zh' else 'english_words_v1',
              'evidence': 'forced_alignment_candidate', 'transcript_origin': 'open_model_ASR_then_Ultra_no_head',
              'boundary_review': 'pending', 'source_utterance_id': source['source_id'], 'source_session_id': None,
              'corpus': source['dataset'], 'split': ENGINEERING_SPLIT,
              'duration_sec': (end-start)/identity['sample_rate'],
              'n_samples_16k': math.ceil((end-start)*16000/identity['sample_rate']), 'family_id': family_id,
              'usable_for_training': False, 'usable_for_evaluation': False, 'human_review': None,
              'provenance': {'source_original': copy.deepcopy(source), 'producer_contract_sha256': context['contract_sha'],
                  'producer_bundle_sha256': context['bundle_sha'],
                  'producer_sidecar_sha256': context['sidecar_sha'], 'producer_commit_sha256': context['commit_sha'], 'speaker_evidence': copy.deepcopy(source['speaker_evidence']),
                  'source_identity_original': copy.deepcopy(identity), 'source_session_evidence': 'not_available_never_inferred_from_parent_path',
                  'no_automatic_eligibility_release': True, 'original_clock_mapping': 'source_offset_floor_start_ceil_end_v1'},
              'alignment': {'method': 'ultra_no_head_pending_candidate', 'native_unit_refs': copy.deepcopy(selected),
                  'source_window_id': source_window['window_id'], 'source_window_input_PCM_sha256': source_window['input_pcm_sha256'],
                  'window_ASR_raw_text': source_window['asr']['raw_text'], 'window_ASR_normalized_text': source_window['asr']['normalized_text'],
                  'window_native_raw_ms': copy.deepcopy(source_window['alignment']['native_timestamps_ms']),
                  'window_author_FIX_ms': copy.deepcopy(source_window['alignment']['fixed_timestamps_ms'])},
              'vad': {'not_completeness_proof': True}}
    result['unit_id'] = word.digest(word.unit_identity(result))
    return result


def validate_family(records):
    need([r['n_units'] for r in records] == list(COUNTS), 'family must contain exactly ordered1/2/3/6')
    long = records[-1]; previous = long['start_sample']
    fields = ('speaker_id','audio_sha256','audio_realpath','source_utterance_id','source_session_id',
              'start_sample','sample_rate','source_num_samples','source_channels','language','counting_policy','family_id')
    for r in records:
        need(all(r.get(k) == long.get(k) for k in fields), 'family source/speaker/anchor/clock/language differs')
        need(r['unit_texts'] == long['unit_texts'][:r['n_units']] and r['end_sample'] > previous, 'family nonprefix or nonincreasing end')
        need(r['alignment']['native_unit_refs'] == long['alignment']['native_unit_refs'][:r['n_units']], 'family native occurrence prefix differs')
        previous = r['end_sample']


def diagnostic_matches(source, context):
    """Exact registered ID or explicitly recorded original/alias path; no suffix guesses."""
    reference = context.get('diagnostic_protection')
    need(reference is not None and len(context.get('diagnostic_protection_sha256', '')) == 64
         and digest(reference) == context.get('diagnostic_protection_content_sha256'),
         'SHA-pinned diagnostic protection reference required')
    return [x['case_id'] for x in reference['cases']
            if (x.get('source_id') is not None and source.get('source_id') == x['source_id'])
            or source.get('audio_path') in (x['path'], x['normalized_alias_path'])]


def propose(record, context):
    """Pure transformation AFTER verify_publication; no source/model I/O."""
    source = record['source']; result = {'source_id': source['source_id'], 'source_original': copy.deepcopy(source),
        'producer_decision': copy.deepcopy(record['decision']), 'human_review': None, 'usable_for_training': False,
        'usable_for_evaluation': False, 'status': 'diagnostic_only', 'reasons': [], 'families': []}
    protected = diagnostic_matches(source, context)
    result['diagnostic_protection_sha256'] = context['diagnostic_protection_sha256']
    if protected:
        result['reasons'] = ['frozen_benchmark_diagnostic_only']
        result['protected_diagnostic_case_ids'] = protected
        return result
    if record['decision']['state'] != 'candidate':
        result['reasons'] = ['producer_not_candidate'] + record['decision']['reasons']; return result
    speaker = source.get('speaker_evidence') or {}; label = speaker.get('provider_label')
    if not isinstance(label, str) or not label.strip() or speaker.get('verified_identity') is not False:
        result['reasons'] = ['no_unambiguous_unverified_provider_speaker_label']; return result
    speaker_id = source['dataset'] + '::provider_label::' + label
    word = context['word']
    for window in record['windows']:
        observed = window['asr'].get('language'); language = {'Chinese': 'zh', 'English': 'en'}.get(observed)
        if language is None:
            result['reasons'].append(window['window_id'] + ':word_protocol_language_not_supported'); continue
        units = [copy.deepcopy(u) for u in record['units'] if u['window_id'] == window['window_id']]
        if len(units) < 6:
            result['reasons'].append(window['window_id'] + ':fewer_than_six_owned_native_units'); continue
        selected = units[:6]; indices = [u['native_index'] for u in selected]
        if indices != list(range(indices[0], indices[0]+6)):
            result['reasons'].append(window['window_id'] + ':nonconsecutive_native_occurrences'); continue
        try:
            for u in selected:
                need(len(word.tokenize_units(u['text'], language)) == 1, 'coarse_or_ambiguous_native_token_no_artificial_split')
            start = selected[0]['start_sample']; previous = start
            for u in selected:
                need(u['sample_rate'] == record['source_identity']['sample_rate'] and u['start_sample'] >= previous
                     and u['end_sample'] > u['start_sample'], 'invalid_or_overlapping_original_clock_unit')
                previous = u['end_sample']
            family_id = 'family_' + digest({'source_id': source['source_id'], 'audio_sha256': record['source_identity']['file_sha256'],
                'sample_rate': record['source_identity']['sample_rate'], 'start_sample': start,
                'language': language, 'speaker_id': speaker_id, 'window_id': window['window_id'],
                'producer_contract_sha256': context['contract_sha']})
            records = [word_template(record, selected[:n], language, speaker_id, family_id, context) for n in COUNTS]
            validate_family(records)
            result['families'].append({'family_id': family_id, 'counts': list(COUNTS), 'window_id': window['window_id'],
                'ASR_language_prediction_not_truth': observed, 'speaker_label_not_verified_human_identity': True,
                'media_binding_state': 'required_before_old_word_schema_export',
                'raw_and_FIX_preserved_in_source_sidecar': True, 'records_pending_media_binding': records})
        except ValueError as exc: result['reasons'].append(window['window_id'] + ':' + str(exc))
    if result['families']: result['status'] = 'engineering_candidate_proposals_pending_media_and_eligibility'
    return result


def verify_media(path, expected):
    """Exact media copy may have a different inode, never different bytes/PCM."""
    path = Path(path); need(path.is_absolute() and path.is_file() and not path.is_symlink(), 'mapped media must be an absolute regular file')
    before = path.stat(); need(file_sha(path) == expected['file_sha256'], 'mapped encoded media SHA differs')
    h = hashlib.sha256(); frames = 0; decoder = None
    try:
        import soundfile as sf
    except ImportError:
        sf = None
    if sf is not None:
        with sf.SoundFile(path) as stream:
            sr, channels, total = stream.samplerate, stream.channels, len(stream)
            while True:
                data = stream.read(65536, dtype='float32', always_2d=True)
                if not len(data): break
                need(bool(__import__('numpy').isfinite(data).all()), 'nonfinite mapped PCM')
                h.update(data.astype('<f4', copy=False).tobytes(order='C')); frames += len(data)
        decoder = 'soundfile/' + sf.__version__
    else:
        # Exact stdlib PCM-WAV audit, not a lossy conversion or guessed header.
        with wave.open(str(path), 'rb') as stream:
            sr, channels, total, width = stream.getframerate(), stream.getnchannels(), stream.getnframes(), stream.getsampwidth()
            need(stream.getcomptype() == 'NONE' and width in (1, 2, 3, 4), 'soundfile required for non-PCM-WAV media')
            while True:
                raw = stream.readframes(65536)
                if not raw: break
                need(len(raw) % (channels*width) == 0, 'partial PCM frame')
                frames += len(raw)//(channels*width)
                for start in range(0, len(raw), width):
                    value = (raw[start]-128) if width == 1 else int.from_bytes(raw[start:start+width], 'little', signed=True)
                    h.update(struct.pack('<f', value/(2**(8*width-1))))
        decoder = 'stdlib-wave/integer_PCM_exact_float32le'
    after = path.stat()
    same = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(same(before) == same(after), 'mapped media changed during verification')
    need((sr, channels, total, frames) == (expected['sample_rate'], expected['channels'], expected['frames'], expected['frames']), 'mapped source clock differs')
    need(h.hexdigest() == expected['pcm_sha256'], 'mapped native multichannel PCM SHA differs')
    return {'audio_path': str(path), 'audio_realpath': str(path.resolve(strict=True)),
            'audio_stat': {'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns},
            'verification': {'encoded_sha256': expected['file_sha256'], 'native_PCM_sha256': h.hexdigest(),
                             'clock_matched': True, 'decoder': decoder, 'copy_path_mapping_disclosed': True}}


def engineer_export(proposals, media_map, context):
    records = []; refused = []
    for proposal in proposals:
        for family in proposal['families']:
            candidates = copy.deepcopy(family['records_pending_media_binding']); sid = proposal['source_id']
            try:
                need(proposal['source_id'] == proposal['source_original']['source_id'], 'proposal source identity differs')
                need(not diagnostic_matches(proposal['source_original'], context), 'frozen_benchmark_diagnostic_only')
                validate_family(candidates)
                need(sid in media_map, 'source_media_binding_not_provided')
                expected = candidates[0]['provenance']['source_identity_original']
                need(all(r['audio_sha256'] == expected['file_sha256'] and r['sample_rate'] == expected['sample_rate']
                         and r['source_num_samples'] == expected['frames'] and r['source_channels'] == expected['channels'] for r in candidates),
                     'old word header/audio identity differs from producer source identity')
                bound = verify_media(media_map[sid], expected)
                for record in candidates:
                    record.update({k: bound[k] for k in ('audio_path', 'audio_realpath', 'audio_stat')})
                    record['provenance']['current_media_binding'] = bound['verification']
                    context['word'].validate_unit(record, allow_candidates=True, verify_audio=False)
                    need(record['evidence'] == 'forced_alignment_candidate' and record['boundary_review'] == 'pending'
                         and record['transcript_origin'] == 'open_model_ASR_then_Ultra_no_head', 'reference or human evidence promotion refused')
                    need(record['split'] == ENGINEERING_SPLIT and record['usable_for_training'] is False
                         and record['usable_for_evaluation'] is False and record['human_review'] is None, 'eligibility upgrade refused')
                records.extend(candidates)
            except (ValueError, OSError, wave.Error, RuntimeError, KeyError, TypeError) as exc:
                refused.append({'source_id': sid, 'family_id': family['family_id'], 'reason': str(exc), 'traceback': traceback.format_exc()})
    return records, refused


def run(args):
    context = load_context(args); selection, _, selection_sha = read_json(str(Path(args.selection).resolve()), args.selection_sha)
    rows = selection['records']; need(0 < len(rows) <= args.max_sources, 'explicit source cap exceeded or empty selection')
    out = Path(args.output).resolve(); need(not out.exists() and out != BASE and BASE not in out.parents
         and (REPOSITORY is None or (out != REPOSITORY and REPOSITORY not in out.parents)), 'output must be fresh and outside source repository/code')
    out.mkdir(); results = []; integrity_errors = []; proposals = []; seen_sources = set(); seen_anchors = set()
    for ordinal, row in enumerate(rows):
        try:
            record, raw, commit, sidecar_sha, commit_sha = verify_publication(row, context)
            need(record['source']['source_id'] not in seen_sources, 'duplicate source ID in selection')
            seen_sources.add(record['source']['source_id'])
            per_source_context = dict(context, sidecar_sha=sidecar_sha, commit_sha=commit_sha)
            result = propose(record, per_source_context)
            for family in result['families']:
                long = family['records_pending_media_binding'][-1]
                anchor = (long['audio_sha256'], long['start_sample'], long['sample_rate'])
                need(anchor not in seen_anchors, 'duplicate physical audio anchor across families'); seen_anchors.add(anchor)
            result['input_identity'] = {'sidecar_sha256': sidecar_sha, 'commit_sha256': commit_sha,
                'producer_contract_sha256': context['contract_sha'], 'producer_bundle_sha256': context['bundle_sha']}
            folder = out / ('input_%04d' % ordinal); folder.mkdir()
            (folder/'sidecar_original.json').write_bytes(raw); (folder/'commit_original.json').write_bytes(commit)
            proposals.append(result); results.append({'ordinal': ordinal, 'source_id': result['source_id'], 'status': result['status'], 'reasons': result['reasons'], 'families': len(result['families'])})
        except Exception:
            error = {'ordinal': ordinal, 'status': 'refused_integrity', 'input_original': row, 'traceback': traceback.format_exc()}
            results.append(error); integrity_errors.append(error)
    body = {'schema': 'ultra-word-consumption-proposals/v1', 'selection_sha256': selection_sha,
        'producer_contract_sha256': context['contract_sha'], 'producer_bundle_sha256': context['bundle_sha'],
        'word_schema_source_sha256': context['word_schema_sha'], 'source_count': len(rows), 'results': results, 'proposals': proposals,
        'usable_for_training': False, 'usable_for_evaluation': False, 'human_review': None,
        'direct_word_training_manifest': False, 'GPU_used': False, 'model_API_calls': 0,
        'diagnostic_protection_sha256': context['diagnostic_protection_sha256'],
        'adapter_source_sha256': file_sha(__file__), 'adapter_implementation': Path(__file__).name}
    (out/'proposals.json').write_text(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    engineering_export_failed = False
    if args.emit_engineering_candidates:
        maps, _, _ = read_json(str(Path(args.audio_bindings).resolve()), args.audio_bindings_sha)
        records, refusals = engineer_export(proposals, maps['source_to_audio_path'], context)
        engineering_export_failed = bool(refusals) or not records
        export = {'conversion_succeeded': not engineering_export_failed, 'schema': 'engineering-word-candidate-export-status/v1', 'candidate_records': len(records), 'refusals': refusals,
                  'training_release': False, 'evaluation_release': False, 'source_count': len(rows)}
        if records:
            dataset = context['word'].seal_dataset(records, {'split': ENGINEERING_SPLIT, 'source_proposals_sha256': file_sha(out/'proposals.json'),
                'usable_for_training': False, 'usable_for_evaluation': False, 'human_review': None, 'purpose': 'CPU candidate shape inspection only; old train split gate rejects'})
            (out/'engineering_candidates.json').write_text(json.dumps(dataset, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        (out/'engineering_export_status.json').write_text(json.dumps(export, ensure_ascii=False, indent=2)+'\n')
    return body, bool(integrity_errors) or engineering_export_failed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('producer-code','producer-bundle-sha','contract','contract-sha','word-schema','word-schema-sha','selection','selection-sha','output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--diagnostic-protection', required=True, help='external diagnostic reference, never bundled with public code')
    p.add_argument('--diagnostic-protection-sha', required=True)
    p.add_argument('--max-sources', type=int, required=True)
    p.add_argument('--emit-engineering-candidates', action='store_true')
    p.add_argument('--audio-bindings'); p.add_argument('--audio-bindings-sha')
    args = p.parse_args()
    if args.emit_engineering_candidates and not (args.audio_bindings and args.audio_bindings_sha): p.error('explicit SHA-pinned physical audio bindings required')
    body, failed = run(args)
    print(json.dumps({'sources': body['source_count'], 'families': sum(len(r['families']) for r in body['proposals']),
                      'integrity_errors': sum(r['status']=='refused_integrity' for r in body['results']), 'training_release': False}))
    raise SystemExit(2 if failed else 0)


if __name__ == '__main__': main()
