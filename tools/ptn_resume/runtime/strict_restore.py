"""47→48接续的实际运行公共门；只在显式configure后绑定新包。"""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
import ptn_contracts as C
import state_gate

PYTHON = '/root/miniforge3/envs/redimnet2/bin/python'
DECODER = Path(C.BASE + '/lm-continuation-20261022/event-review-20261023-v4/runtime_audio/lib/libsndfile.so.1')


def need(value, message):
    C.need(value, message)


def sha(path):
    return C.hash_file(Path(path))[0]


def snapshot(path):
    return C.snapshot(Path(path).stat())


def configure(path, pin, package_sha):
    global CONTRACT_PATH, CONTRACT_SHA, PACKAGE_SHA, PACKAGE, BUNDLE, CONFIG, SOURCE, RUN, OLD, PREDECESSOR, CK, CK_SHA, SPK_SHA, CONTROL
    CONTRACT_PATH = C.physical(path)
    CONTRACT_SHA, PACKAGE_SHA = pin, package_sha
    PACKAGE, BUNDLE = CONTRACT_PATH.parent, CONTRACT_PATH.parent / 'runtime'
    need(C.is_sha(PACKAGE_SHA) and sha(PACKAGE / 'bundle_files.json') == PACKAGE_SHA,
         'externally pinned new package manifest differs')
    CONFIG = state_gate.verify_package(CONTRACT_PATH, CONTRACT_SHA)
    SOURCE, RUN = Path(CONFIG['source_root']), Path(CONFIG['new_run_root'])
    OLD, PREDECESSOR = Path(C.ORIGINAL), Path(C.PREDECESSOR)
    CK, CK_SHA, SPK_SHA = Path(C.CHECKPOINT), C.CK_SHA, C.SPK_SHA
    CONTROL = RUN.with_name(RUN.name + '-control')
    need(SOURCE == PACKAGE / 'source' and RUN != PACKAGE and PACKAGE not in RUN.parents,
         'source/run/package boundaries differ')
    return CONFIG


def verify_bundle():
    need(sha(PACKAGE / 'bundle_files.json') == PACKAGE_SHA, 'new package manifest drift; never reseal')
    return state_gate.verify_package(CONTRACT_PATH, CONTRACT_SHA)


def verify_source():
    verify_bundle()
    return {'source': str(SOURCE), 'source_members': 190,
            'source_manifest_sha256': CONFIG['source_manifest_sha256'],
            'source_delta_sha256': CONFIG['source_delta_sha256'],
            'original_source189_sha256': C.SOURCE189_SHA}


def validate_config(config):
    import yaml
    need(sha(CONFIG['resume_config_path']) == CONFIG['resume_config_sha256'], 'effective config bytes differ')
    expected = yaml.safe_load(Path(CONFIG['resume_config_path']).read_bytes())
    need(C.canonical(config) == C.canonical(expected), 'only reviewed new resume configuration allowed')
    return expected


def checkpoint():
    import torch
    before = snapshot(CK)
    need(before[2] == 650777251 and sha(CK) == CK_SHA, 'model47 identity differs')
    payload = torch.load(CK, map_location='cpu', weights_only=True, mmap=True)
    need(set(payload) == {'model', 'optimizer', 'scaler', 'epoch'} and payload['epoch'] == 47,
         'complete same-stage model47 required')
    need(snapshot(CK) == before, 'model47 changed during load')
    complete = sorted(int(p.stem.split('_')[-1]) for p in CK.parent.glob('model_*.pt')
                      if p.stem.split('_')[-1].isdigit())
    need(complete and complete[-1] == 47, 'model47 no longer latest complete predecessor checkpoint')
    need(not any('tmp' in p.name for p in CK.parent.iterdir()), 'predecessor has unfinished checkpoint files')
    return payload


def verify_dependencies():
    legacy = Path(CONFIG['legacy']['bundle_root'])
    manifest = C.json_bytes((legacy / 'bundle_files.json').read_bytes())
    need(sha(legacy / 'bundle_files.json') == C.LEGACY_BUNDLE_SHA, 'legacy bundle manifest changed')
    need(sha(legacy / 'dependencies.json') == manifest['dependencies.json'], 'dependency records changed')
    value = C.json_bytes((legacy / 'dependencies.json').read_bytes())
    for row in value['small_full_sha_files']:
        need(sha(row['path']) == row['sha256'], 'fixed dependency differs: ' + row['path'])
    for row in value['tables']:
        p = Path(row['path']); before = snapshot(p)
        need(before[2] == row['bytes'] and before[3] == row['mtime_ns'], 'training table identity changed')
        with p.open('rb') as f:
            first = f.read(4096); f.seek(max(0, before[2] - 4096)); last = f.read(4096)
        need(hashlib.sha256(first).hexdigest() == row['first4096_sha256'] and
             hashlib.sha256(last).hexdigest() == row['last4096_sha256'], 'training table bounded bytes changed')
        need(snapshot(p) == before, 'training table changed during bounded read')
    return {'fixed_dependencies': len(value['small_full_sha_files']), 'tables': len(value['tables']),
            'full_tables_rehashed': False, 'dependency_record_sha256': manifest['dependencies.json']}


def verify_imports():
    records = {}
    for name, module in list(sys.modules.items()):
        if name in ('redimnet2', 'wespeaker_lite') or name.startswith(('redimnet2.', 'wespeaker_lite.')):
            origin = getattr(module, '__file__', None)
            if origin is None:
                paths = [Path(p).resolve() for p in getattr(module, '__path__', [])]
                need(len(paths) == 1 and SOURCE in paths[0].parents, 'namespace outside new source: ' + name)
                records[name] = {'namespace_directory': str(paths[0])}
            else:
                p = Path(origin).resolve(); need(SOURCE in p.parents, 'module outside new source: ' + name)
                records[name] = {'path': str(p), 'sha256': sha(p)}
    return records


def fresh_terminals():
    refs = {}
    C.physical(CONTROL)
    info = CONTROL.stat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o022,
         'unsafe fresh terminal control directory')
    for name in ('original', 'predecessor'):
        p = CONTROL / (name + '-terminal.json'); st = p.lstat()
        need(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_nlink == 1 and not st.st_mode & 0o022,
             'unsafe terminal control file')
        refs[name] = {'path': str(p), 'sha256': sha(p)}
    result = {'original': C.terminal(refs['original'], 't-20261019232824-8zqrl', C.ORIGINAL, {'Terminated'}),
              'predecessor': C.terminal(refs['predecessor'], 't-20261024154424-vggvm', C.PREDECESSOR, {'Failed'})}
    return refs, result


def verify_restored(model, optimizer, scaler, start_epoch, epoch_iter, world, config, spk_path,
                    *, scheduler, margin):
    need(start_epoch == 48 and epoch_iter == 15001 and world == 8, 'exact PTN48/W8/15001 required')
    need(sha(spk_path) == SPK_SHA, 'actual newly computed speaker map differs')
    # state_gate independently reloads the pinned file; caller payload is not a proof.
    refs, _ = fresh_terminals()
    return state_gate.verify_restored(CONTRACT_PATH, CONTRACT_SHA, model=model, optimizer=optimizer,
        scaler=scaler, scheduler=scheduler, margin=margin, config=config,
        speaker_map_path=spk_path, terminal_evidence=refs)
