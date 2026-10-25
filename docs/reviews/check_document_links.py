#!/usr/bin/env python3
"""清点说明文件，并只检查本地链接目标；不访问网络或执行文档命令。"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit


LOCAL_ONLY = {'.git', '.venv', 'venv', '.audit-local', '__pycache__',
              'node_modules', '.mypy_cache', '.pytest_cache', '.ruff_cache'}


def public_paths(root):
    """优先使用Git公开工作树集合；无独立Git的合成夹具才用剪枝遍历。"""
    try:
        probe = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
                               capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        probe = None
    if probe is not None and probe.returncode == 0 and Path(probe.stdout.strip()).resolve() == root:
        result = subprocess.run(['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                                capture_output=True, check=True, timeout=30)
        paths = {root / os.fsdecode(raw) for raw in result.stdout.split(b'\0') if raw}
        mode = 'git ls-files --cached --others --exclude-standard；额外排除本地环境与私有审计目录'
    else:
        paths = set()
        for folder, directories, names in os.walk(root, followlinks=False):
            directories[:] = [name for name in directories if name not in LOCAL_ONLY
                              and not (Path(folder) / name).is_symlink()]
            paths.update(Path(folder) / name for name in names)
        mode = '非独立Git目录的合成夹具剪枝遍历；不跟随软链'
    selected = []
    for path in sorted(paths):
        parts = path.relative_to(root).parts
        if LOCAL_ONLY.intersection(parts):
            continue
        current = root
        aliased = current.is_symlink()
        for part in parts:
            if aliased:
                break
            current = current / part
            aliased = current.is_symlink()
        if not aliased and path.is_file():
            selected.append(path)
    return selected, mode


def inspect(root):
    root = Path(root).resolve()
    documents, links, failures = [], [], []
    paths, selection = public_paths(root)
    for path in paths:
        if path.suffix.lower() != '.md' and not path.name.startswith('README'):
            continue
        raw = path.read_bytes()
        documents.append({'path': path.relative_to(root).as_posix(),
                          'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
        text = raw.decode('utf-8', errors='strict')
        # 用相同换行数替代代码块，保持原文件行号；代码示例里的链接不是说明导航。
        text = re.sub(r'(?ms)^```.*?^```[^\n]*',
                      lambda match: '\n' * match.group().count('\n'), text)
        targets = [(match.start(), match.group(1)) for match in
                   re.finditer(r'!?\[[^\]]*\]\(([^\n]+?)\)', text)]
        targets += [(match.start(), match.group(1)) for match in
                    re.finditer(r'<(?:img|a)\b[^>]*?\b(?:src|href)=["\']([^"\']+)["\']', text)]
        for offset, target in sorted(targets):
            target = target.strip()
            if not target:
                continue
            target = target[1:target.index('>')] if target.startswith('<') and '>' in target else target.split()[0]
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            item = {'document': path.relative_to(root).as_posix(),
                    'line': text[:offset].count('\n') + 1, 'target': target}
            target_path = path.parent / unquote(parsed.path)
            item['exists'] = target_path.exists()
            item['external_to_repository'] = not target_path.resolve().is_relative_to(root)
            links.append(item)
            if not item['exists']:
                failures.append(item)
    return {'schema': 'redimnet2-document-link-check/v1',
            'file_selection': selection,
            'scope': 'UTF-8说明文件与Markdown/HTML内联本地路径；不检查网络、标题锚点、代码示例和正文语义',
            'documents': documents, 'document_count': len(documents),
            'local_links': links, 'missing_target_count': len(failures), 'missing_targets': failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = inspect(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result['missing_targets'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
