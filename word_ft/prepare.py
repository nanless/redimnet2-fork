"""Build whole-reference-utterance Chinese protocols; never manufacture short units."""
from __future__ import annotations
import collections
import hashlib
import json
import re
from pathlib import Path

AXIS = ('1', '2-3', '4-6', '7+')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def chinese_reference(text):
    """Only unambiguous Han characters plus punctuation/space; numbers need reviewed expansion."""
    from word_ft.schema import normalize_text
    return normalize_text(text, 'zh')


def count_bin(count):
    return '1' if count == 1 else '2-3' if 2 <= count <= 3 else '4-6' if 4 <= count <= 6 else '7+' if count >= 7 else None


def read_kaldi(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in result:
            raise ValueError(f'bad or duplicate metadata row: {path}')
        result[fields[0]] = fields[1]
    return result


def index_childmandarin(root):
    """Read existing text/speaker/audio mapping, not any inferred ASR transcription."""
    root = Path(root).resolve()
    prepared = root / 'kaldi_files/prepared_data'
    tables = {k: read_kaldi(prepared / k) for k in ('text', 'utt2spk', 'wav.scp')}
    if any(set(v) != set(tables['text']) for v in tables.values()):
        raise ValueError('metadata key sets differ')
    candidates, rejected = [], []
    for uid, text in sorted(tables['text'].items()):
        try:
            norm = chinese_reference(text)
            path = Path(tables['wav.scp'][uid])
            if not path.is_absolute() or not path.resolve().is_relative_to(root) or not path.is_file():
                raise ValueError('audio outside corpus or missing; pipelines are forbidden')
            candidates.append({'unit_id': 'childmandarin:' + uid, 'source_utterance_id': 'childmandarin:' + uid,
                'speaker_id': 'childmandarin:' + tables['utt2spk'][uid], 'path': str(path),
                'text': text, 'spoken_text': norm, 'n_units': len(norm), 'count_bin': count_bin(len(norm)),
                'language': 'zh', 'dataset': 'BAAI-ChildMandarin41.25H', 'original_split': path.resolve().relative_to(root).parts[0],
                'session_id': None, 'session_evidence': 'not supplied; filename suffix is not assumed to be a session',
                'source_metadata_path': str(path.with_suffix('.json'))})
        except ValueError as error:
            rejected.append({'utterance_id': uid, 'text': text, 'reason': str(error)})
    per_speaker = collections.defaultdict(collections.Counter)
    for unit in candidates:
        per_speaker[unit['speaker_id']][unit['count_bin']] += 1
    return candidates, {'total_rows': len(tables['text']), 'accepted_text_candidates': len(candidates),
        'rejected': rejected, 'count_bins': dict(collections.Counter(x['count_bin'] for x in candidates)),
        'per_speaker': {k: dict(v) for k, v in per_speaker.items()},
        'metadata_sha256': {k: hashlib.sha256((prepared / k).read_bytes()).hexdigest() for k in tables}}


def select_speaker_splits(candidates, *, dev_speakers=30, test_speakers=30, seed=20261023, required_bins=('2-3', '4-6', '7+')):
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for unit in candidates:
        grouped[unit['speaker_id']][unit['count_bin']].append(unit)
    eligible = [s for s, bins in grouped.items() if all(len(bins.get(b, [])) >= 2 for b in required_bins)]
    eligible.sort(key=lambda s: digest([seed, s]))
    if dev_speakers < 2 or test_speakers < 2 or len(eligible) < dev_speakers + test_speakers:
        raise ValueError('insufficient disjoint speakers with two natural files per required count bin')
    splits = {'dev-cal': eligible[:dev_speakers], 'test': eligible[dev_speakers:dev_speakers + test_speakers]}
    selected = []
    for split, speakers in splits.items():
        for speaker in speakers:
            for b in AXIS:
                pool = sorted(grouped[speaker].get(b, []), key=lambda u: digest([seed, u['unit_id']]))
                for slot, unit in enumerate(pool[:2]):
                    selected.append(dict(unit, split=split, template_slot=slot))
    return selected, {'seed': seed, 'required_bins': list(required_bins), 'eligible_speakers': len(eligible),
                      'splits': splits, 'speaker_disjoint': True,
                      'limitation': 'cross-file only; no evidenced cross-session guarantee'}


def build_count_matrix(units, split='test'):
    """Same speaker/slot templates across counts; different recordings, never same-crop paired."""
    lookup = {}
    for unit in units:
        if unit['split'] != split:
            continue
        key = (unit['speaker_id'], unit['count_bin'], unit['template_slot'])
        if key in lookup:
            raise ValueError('duplicate speaker/count/template slot')
        lookup[key] = unit
    speakers = sorted({k[0] for k in lookup})
    covered_bins = {b for b in AXIS if len(speakers)>=2 and all((s,b,0) in lookup and (s,b,1) in lookup for s in speakers)}
    cells = []
    for left in AXIS:
        for right in AXIS:
            trials = []
            for a in speakers if left in covered_bins and right in covered_bins else []:
                for b in speakers:
                    e, t = lookup.get((a, left, 0)), lookup.get((b, right, 1))
                    if e is None or t is None:
                        continue
                    if e['source_utterance_id'] == t['source_utterance_id'] or (e.get('audio_sha256') is not None and e.get('audio_sha256') == t.get('audio_sha256')):
                        raise ValueError('registration/test source utterance leakage')
                    condition = 'matched' if e['spoken_text'] == t['spoken_text'] else 'mismatched'
                    trials.append({'trial_id': digest([split, left, right, a, b])[:24], 'label': int(a == b),
                        'enroll': e['unit_id'], 'verify': t['unit_id'], 'enroll_speaker': a, 'verify_speaker': b,
                        'content_condition': condition, 'template_id': digest([split, a, b])[:24]})
            groups = {}
            for condition in ('all', 'matched', 'mismatched'):
                group = trials if condition == 'all' else [t for t in trials if t['content_condition'] == condition]
                labels = {t['label'] for t in group}
                groups[condition] = {'trials': group, 'trialset_sha256': digest(group),
                    'available': labels == {0, 1}, 'positive': sum(t['label'] for t in group),
                    'negative': sum(1-t['label'] for t in group),
                    'unavailable_reason': None if labels == {0, 1} else ('fixed speaker cohort lacks two whole files in this count bin' if left not in covered_bins or right not in covered_bins else 'natural whole files do not supply both trial classes')}
            cells.append({'name': left+'x'+right, 'enroll_count_bin': left, 'test_count_bin': right, 'groups': groups})
    return {'schema_version': 1, 'split': split, 'axis': list(AXIS), 'cells': cells,
        'comparison_policy': 'fixed speaker/template strata; different whole files across count bins; not same-crop pairing'}


def materialize_reference_units(selected, metadata_sha256):
    from word_ft.schema import make_unit, file_sha
    result = []
    for candidate in selected:
        meta_path = Path(candidate['source_metadata_path'])
        raw = meta_path.read_bytes()
        source = json.loads(raw)
        if (source.get('text') != candidate['text'] or str(source.get('speaker_id')) != candidate['speaker_id'].split(':')[-1]
                or source.get('id') != Path(candidate['path']).stem):
            raise ValueError('original reference JSON and prepared metadata differ: '+str(meta_path))
        unit = make_unit(candidate['path'], candidate['speaker_id'], source['text'], language='zh',
            source_utterance_id=candidate['source_utterance_id'], corpus=candidate['dataset'], split=candidate['split'],
            provenance={'original_metadata_path': str(meta_path), 'original_metadata_sha256': hashlib.sha256(raw).hexdigest(),
                'original_metadata': source, 'prepared_metadata_sha256': metadata_sha256,
                'original_split': candidate['original_split'],
                'completeness_scope': 'provider whole-utterance reference; no independent listening/word-boundary audit claimed',
                'session_evidence': candidate['session_evidence']})
        unit['template_slot'] = candidate['template_slot']
        result.append(unit)
    # Identical audio bytes can be a duplicate even when paths/utterance IDs differ.
    by_hash = collections.defaultdict(list)
    for unit in result: by_hash[unit['audio_sha256']].append(unit['unit_id'])
    if any(len(ids) > 1 for ids in by_hash.values()):
        raise ValueError('duplicate audio bytes among selected units; review selection explicitly')
    return result


def duration_statistics(units):
    import statistics
    result = {}
    for split in ('dev-cal', 'test'):
        result[split] = {}
        for b in AXIS:
            rows = [r for r in units if r['split'] == split and r['count_bin'] == b]
            values = [r['duration_sec'] for r in rows]
            duration_bins = collections.Counter('<=0.5' if x<=.5 else '0.5-1' if x<=1 else '1-2' if x<=2 else '2-3' if x<=3 else '3-5' if x<=5 else '>5' for x in values)
            result[split][b] = {'clips':len(rows),'speakers':len({r['speaker_id'] for r in rows}),
                'seconds_min':min(values) if values else None,'seconds_median':statistics.median(values) if values else None,
                'seconds_max':max(values) if values else None,'natural_elapsed_duration_bins':dict(duration_bins)}
    return result


def prepare_childmandarin(root, output, *, dev_speakers=30, test_speakers=30, seed=20261023):
    from word_ft.schema import protect_output, write_dataset
    output = protect_output(output)
    if output.exists(): raise FileExistsError('preparation requires a new immutable output: '+str(output))
    candidates, inventory = index_childmandarin(root)
    selected, selection = select_speaker_splits(candidates, dev_speakers=dev_speakers, test_speakers=test_speakers, seed=seed)
    units = materialize_reference_units(selected, inventory['metadata_sha256'])
    output.mkdir(parents=True, exist_ok=False)
    dataset = write_dataset(output/'units.json', units, {'selection':selection,'inventory':inventory,
        'duration_statistics':duration_statistics(units), 'scope':'whole natural reference utterances; no arbitrary cut, VAD trimming, padding, ASR or alignment inference'})
    protocol = {'schema':'word-count-matrix/v1','dataset_id':dataset['dataset_id'],
        'dataset_file':'units.json','selection':selection,'matrices':{s:build_count_matrix(units,s) for s in ('dev-cal','test')}}
    protocol['protocol_id']=digest(protocol)
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return {'output':str(output),'dataset_id':dataset['dataset_id'],'protocol_id':protocol['protocol_id'],
        'selected_units':len(units),'selection':selection,'inventory_counts':inventory['count_bins'],
        'duration_statistics':duration_statistics(units)}
