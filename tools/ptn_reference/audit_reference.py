"""只核参考源码的字节身份与语法；不导入归档模块，不授予运行资格。"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
INDEX_SHA256 = '933d19945485c9de71e243e06f5db5cc1c47c936535f013f49edee49fe71dfc7'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stable_read(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'regular non-symlink reference file required')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(identity(before) == identity(after), 'reference changed during read')
    return raw


def audit_tree(root, files):
    """纯文件检查；传入合成清单也不能成为历史身份或运行证明。"""
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), 'physical reference directory required')
    need(isinstance(files, dict) and files, 'nonempty expected file set required')
    need({p.name for p in root.iterdir()} == set(files), 'missing or unexpected reference entry')
    checked = []
    for name, pin in sorted(files.items()):
        rel = PurePosixPath(name)
        need(not rel.is_absolute() and len(rel.parts) == 1 and str(rel) == name
             and name not in {'.', '..'}, 'reference filenames cannot escape directory')
        path = root/name
        raw = stable_read(path)
        need(type(pin['bytes']) is int and len(raw) == pin['bytes'] and sha(raw) == pin['sha256'],
             'reference bytes/SHA differ: '+name)
        if path.suffix == '.py':
            # compile produces a code object only; never eval/exec/import it.
            compile(raw, name, 'exec')
            kind = 'Python compile only, not executed'
        elif path.suffix == '.sh':
            bash = shutil.which('bash')
            need(bash is not None, 'bash is required for shell syntax checking')
            result = subprocess.run([bash, '-n', str(path)], capture_output=True, timeout=5)
            need(result.returncode == 0 and not result.stderr, 'shell syntax check failed')
            # Bind the path read by bash to the same pinned bytes before/after.
            need(stable_read(path) == raw, 'shell reference changed during syntax check')
            kind = 'Bash -n only, not executed'
        else:
            raise ValueError('only Python/shell source belongs in reference archive')
        checked.append({'name':name, 'bytes':len(raw), 'sha256':sha(raw), 'syntax':kind})
    return {'file_set_and_syntax_verified':True, 'files':checked,
            'archived_module_imported':False, 'GPU_or_training_executed':False,
            'runtime_admission':False, 'historical_execution_proven':False}


def verify_reference(root=HERE/'archive'):
    raw = stable_read(HERE/'archive_index.json')
    need(sha(raw) == INDEX_SHA256, 'historical reference index SHA differs')
    index = json.loads(raw)
    need(index['schema'] == 'ptn-read-only-reference/v1'
         and index['historical_reference_only'] is True and index['runtime_admission'] is False,
         'wrong reference policy')
    need(len(index['files']) == 10 and sum(n.endswith('.py') for n in index['files']) == 9,
         'historical source file denominator differs')
    exact=sum(x.get('identity')=='historical_byte_exact' for x in index['files'].values())
    derived=sum(x.get('identity')=='public_derived_without_fixed_sample_keys' for x in index['files'].values())
    need(exact==index['historical_exact_files']==9 and derived==index['public_derived_files']==1, 'historical/public derivative identity differs')
    need(all(x['sha256']==x['historical_sha256'] for x in index['files'].values() if x['identity']=='historical_byte_exact'), 'historical exact-file identity differs')
    need(index['files']['audio_review.py']['identity']=='public_derived_without_fixed_sample_keys', 'expected public derivative missing')
    result = audit_tree(root, index['files'])
    result.update(reference_source_integrity_verified=True, historical_exact_files=9, public_derived_files=1, archive_index_sha256=INDEX_SHA256,
                  external_inputs_not_included=index['external_inputs_not_included'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, default=HERE/'archive')
    parser.add_argument('--require-runtime-admission', action='store_true',
                        help='always refuses: this archive is not a runnable deployment')
    args = parser.parse_args(argv)
    try:
        result = verify_reference(args.reference_root)
        if args.require_runtime_admission:
            result['refusal'] = 'reference source cannot grant runtime admission; external originals and new execution review are absent'
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if args.require_runtime_admission else 0
    except (ValueError, OSError, SyntaxError, subprocess.SubprocessError) as exc:
        print(json.dumps({'file_set_and_syntax_verified':False, 'runtime_admission':False,
                          'error':str(exc)}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
