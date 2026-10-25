#!/usr/bin/env python3
"""检查本仓库代码语法、Shell语法、配置和文档链接，不启动训练或访问网络。

默认检查所有受控文件以及尚未提交但未被忽略的新增文件。生成的JSON报告保存在
调用者指定位置；退出0表示检查通过，退出1表示有确定错误。依赖模型、GPU和
实际语料的语义检查应另外执行，不能由本脚本的静态结果推断通过。
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def repository_files(root: Path) -> list[Path]:
    """仅使用目标仓库的Git索引/忽略规则，禁止向父目录寻找另一仓库。"""
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--show-toplevel'], text=True).strip()
    if Path(actual).resolve() != root.resolve():
        raise ValueError('目标Git根目录身份不一致')
    raw = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'])
    return sorted({root / path.decode() for path in raw.split(b'\0') if path})


def local_links(path: Path, root: Path) -> list[dict]:
    """检查仓库内相对文件链接；网络地址、页内锚点和明确外部数据路径不在此范围。"""
    text = path.read_text(encoding='utf-8')
    # fenced代码段的示例不是文档链接；去掉后再解析普通Markdown链接。
    text = re.sub(r'(?ms)^```.*?^```[^\n]*$', '', text)
    result = []
    for match in re.finditer(r'!?\[[^\]\n]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)', text):
        target = match.group(1).strip('<>')
        if not target or target.startswith(('#', '/', '~')) or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
            continue
        target = unquote(target.split('#', 1)[0].split('?', 1)[0])
        if not target:
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.is_relative_to(root.resolve()):
            continue
        if not resolved.exists():
            line = text[:match.start()].count('\n') + 1
            result.append({'check': 'markdown_link', 'file': str(path.relative_to(root)), 'line': line, 'target': target})
    return result


def check(root: Path) -> dict:
    files = repository_files(root)
    errors = []
    counts = {'files': len(files), 'python': 0, 'shell': 0, 'json': 0, 'markdown': 0}
    for path in files:
        if not path.is_file():
            continue  # Git已记录的删除由diff/审阅处理，不把缺文件当成语法检查。
        kind = path.suffix.lower()
        try:
            if kind == '.py':
                counts['python'] += 1
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            elif kind == '.sh':
                counts['shell'] += 1
                run = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True, timeout=30)
                if run.returncode:
                    errors.append({'check': 'bash_syntax', 'file': str(path.relative_to(root)), 'detail': run.stderr})
            elif kind == '.json':
                counts['json'] += 1
                json.loads(path.read_text(encoding='utf-8-sig'))
            elif kind == '.md':
                counts['markdown'] += 1
                errors.extend(local_links(path, root))
        except (ValueError, SyntaxError, UnicodeError, OSError, subprocess.TimeoutExpired) as exc:
            errors.append({'check': 'parse', 'file': str(path.relative_to(root)), 'detail': str(exc)})
    return {'schema': 'repository-static-check/v1', 'counts': counts, 'errors': errors, 'passed': not errors,
            'scope': 'Python AST/Bash -n/JSON/Markdown local file links; no GPU, model, audio, network or runtime-quality validation'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='检查结果JSON路径；建议本仓库.audit-local内')
    args = parser.parse_args(argv)
    result = check(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'counts': result['counts'], 'error_count': len(result['errors']), 'passed': result['passed']}, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
