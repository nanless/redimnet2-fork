"""把当前维护源码复制为独立冻结副本，并从显式外部合同生成新身份。"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def helpers(code):
    spec = importlib.util.spec_from_file_location('_annotation_freeze_contract', code/'contracts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze(base_path, base_sha, output, version_prefix):
    h = helpers(HERE)
    h.need(h.sha_file(base_path) == base_sha, 'external base contract SHA differs')
    c = json.loads(Path(base_path).read_bytes())
    h.need(c['purpose'] in {'pilot_validation', 'silver_annotation'}, 'real offline contract required')
    h.need(c.get('model_API_calls_allowed') == 0 and c.get('release_training_labels') is False,
           'automatic API or label release is forbidden')
    h.relative_path(version_prefix)
    h.need('/' not in version_prefix, 'version prefix must be one directory name')
    output = Path(output).resolve()
    repository = next((p for p in HERE.parents if (p/'AGENTS.md').is_file() and (p/'redimnet2').is_dir()), None)
    h.need(output != HERE and HERE not in output.parents and not output.exists()
           and (repository is None or (output != repository and repository not in output.parents)),
           'fresh frozen copy must be outside source repository')
    output.mkdir(parents=True)
    for path in HERE.rglob('*'):
        rel = path.relative_to(HERE)
        if path.is_file() and not path.is_symlink() and 'tests' not in rel.parts and '__pycache__' not in rel.parts:
            if path.suffix == '.py' or path.name == 'sidecar.schema.json':
                target = output/rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    c['parent_contract_sha256'] = base_sha
    # New source pins are intentional; historical contract bytes are never rewritten.
    c['engine_source_files'] = {name: h.sha_file(output/name) for name in h.ENGINE_MODULES}
    identity = h.digest({key: c[key] for key in ('decoder','models','source_files','runtime',
                                                'alignment','engine_source_files','quality_policy')})[:16]
    records = {}
    for purpose in ('pilot_validation', 'silver_annotation'):
        current = copy.deepcopy(c)
        current.update(purpose=purpose, version=version_prefix+'-'+purpose+'-'+identity)
        path = output/'contracts'/(current['version']+'.json')
        pin = h.atomic_new(path, current)
        h.read_contract(path, pin)
        records[purpose] = {'path': str(path), 'sha256': pin}
    bundle = {str(path.relative_to(output)):h.sha_file(path) for path in sorted(output.rglob('*'))
              if path.is_file() and 'contracts' not in path.relative_to(output).parts}
    bundle_sha = h.atomic_new(output/'bundle_files.json', bundle)
    receipt = {'schema':'annotation-maintenance-freeze/v1','parent_contract_sha256':base_sha,
               'bundle_sha256':bundle_sha,'contracts':records,'frozen_copy':str(output),
               'GPU_used':False,'models_executed':False,'historical_acceptance_inherited':False}
    h.atomic_new(output/'freeze_receipt.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-contract', type=Path, required=True)
    p.add_argument('--base-contract-sha', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--version-prefix', required=True)
    a = p.parse_args()
    print(json.dumps(freeze(a.base_contract,a.base_contract_sha,a.output,a.version_prefix), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
