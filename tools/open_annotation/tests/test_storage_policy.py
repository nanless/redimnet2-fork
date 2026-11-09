"""防回归：本目录不得把大数据默认写到旧数据根。

2026-11-09 存储规定（见 storage_policy.py）：大数据默认只落
/root/group-shared/voiceprint 下的两个基址——整轮 run/侧车放
open_annotation/redimnet2-silver-<日期>，准入材料/账本/证据包放
artifacts/redimnet2/<用途>；/root 与 /tmp 只留代码仓库与临时日志。

扫描范围是本目录的代码、builtin 配置与计划模板。出现
storage_policy.FORBIDDEN_DATA_ROOTS 里的前缀即失败，只有带
`storage-policy-allow: <原因>` 标记的行例外。

已声明的历史例外（不改写，见 reviews/storage-policy-open-annotation.md）：
- legacy_snapshots/ 是冻结的生产源码原字节，改写即伪造冻结件。
- 四份 2026-10-28 实际运行记录文档保留当时真实路径。
- storage_policy.py 本身是旧根常量的定义处。
"""
from pathlib import Path
import json
import re
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import storage_policy as POLICY

MARKER = 'storage-policy-allow:'
EXPECTED_LEAVES = ('experiments', 'codex_artifacts', 'annotations')
EXPECTED_PREFIX = '/root/'
EXPECTED_OPEN = '/root/group-shared/voiceprint/open_annotation'
EXPECTED_ARTIFACTS = '/root/group-shared/voiceprint/artifacts/redimnet2'
FROZEN_DIRS = {'legacy_snapshots', '__pycache__'}
POLICY_MODULE = 'storage_policy.py'
CHECKED_SUFFIXES = {'.py', '.json', '.sh'}
# 冻结/历史例外只允许这些文档带旧根字样；新增文档必须用新基址。
HISTORY_DOCS = {
    '原生八卡提速接续执行清单_20261028.md',
    '新特征任务只读取证与故障隔离.md',
    '原生转写中途恢复探针.md',
    '有界提速独立证据与审阅_20261028.md',
}
# 全目录唯一允许出现带标记旧根常量的代码文件（原 Ultra 根身份校验）。
MARKED_CODE_FILES = {'verify_portable_successor_mounts.py'}

FORBIDDEN = re.compile(
    '(?:' + '|'.join(re.escape(root) for root in POLICY.FORBIDDEN_DATA_ROOTS) + r')(?=[/"\'\s:]|$)')


def scope_files():
    """本目录下除冻结快照与字节码缓存外的受检文件。"""
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or path.suffix not in CHECKED_SUFFIXES:
            continue
        if FROZEN_DIRS & set(path.relative_to(HERE).parts):
            continue
        if path.name == POLICY_MODULE:
            continue
        yield path


def hits(path):
    """返回该文件中命中旧数据根的行号与行文本。"""
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return []
    return [(number, line) for number, line in enumerate(text.splitlines(), 1) if FORBIDDEN.search(line)]


class StoragePolicyBases(unittest.TestCase):
    def test_policy_bases_are_the_given_group_shared_roots(self):
        self.assertEqual(str(POLICY.OPEN_ANNOTATION), EXPECTED_OPEN)
        self.assertEqual(str(POLICY.ARTIFACTS), EXPECTED_ARTIFACTS)
        self.assertEqual(str(POLICY.run_root('20261109')), EXPECTED_OPEN + '/redimnet2-silver-20261109')
        self.assertEqual(str(POLICY.artifact_root('native-x')), EXPECTED_ARTIFACTS + '/native-x')

    def test_forbidden_roots_are_exactly_the_three_legacy_trees(self):
        roots = POLICY.FORBIDDEN_DATA_ROOTS
        self.assertEqual(tuple(Path(root).name for root in roots), EXPECTED_LEAVES)
        self.assertTrue(all(root.startswith(EXPECTED_PREFIX) for root in roots))

    def test_forbidden_and_physical_helpers(self):
        for root in POLICY.FORBIDDEN_DATA_ROOTS:
            self.assertTrue(POLICY.forbidden(root))
            self.assertTrue(POLICY.forbidden(root + '/nested/file.json'))
        self.assertFalse(POLICY.forbidden(EXPECTED_OPEN))
        self.assertFalse(POLICY.forbidden('/tmp/scratch'))
        self.assertEqual(str(POLICY.physical(EXPECTED_OPEN + '/a/b')),
                         '/shared/group-shared/voiceprint/open_annotation/a/b')
        self.assertEqual(str(POLICY.physical('/tmp/x')), '/tmp/x')

    def test_run_root_rejects_non_date_and_artifact_rejects_traversal(self):
        for bad in ('2026100', 'abcdefgh', ''):
            with self.assertRaises(ValueError):
                POLICY.run_root(bad)
        for bad in ('', '../escape', '/absolute'):
            with self.assertRaises(ValueError):
                POLICY.artifact_root(bad)


class NoLegacyDataRoots(unittest.TestCase):
    def test_code_and_templates_do_not_default_to_legacy_data_roots(self):
        violations = {}
        for path in scope_files():
            for number, line in hits(path):
                if MARKER in line or (number > 1 and self.previous_line_has_marker(path, number)):
                    continue
                violations.setdefault(str(path.relative_to(HERE)), []).append(number)
        self.assertEqual(violations, {},
                         'legacy data root without storage-policy-allow marker: %r' % violations)

    def previous_line_has_marker(self, path, number):
        lines = path.read_text(encoding='utf-8').splitlines()
        return MARKER in lines[number - 2]

    def test_only_the_declared_constant_carries_a_marker(self):
        marked = {}
        for path in scope_files():
            for number, line in hits(path):
                if MARKER in line or (number > 1 and self.previous_line_has_marker(path, number)):
                    marked.setdefault(path.name, []).append(number)
        self.assertEqual(set(marked), MARKED_CODE_FILES)
        self.assertEqual(len(marked['verify_portable_successor_mounts.py']), 1)

    def test_history_documents_are_the_only_markdown_exception(self):
        found = {}
        for path in sorted(HERE.glob('*.md')):
            matched = hits(path)
            if matched:
                found[path.name] = matched
        self.assertEqual(set(found), HISTORY_DOCS)
        for name, matched in found.items():
            text = (HERE / name).read_text(encoding='utf-8')
            self.assertIn(MARKER, text, 'history doc needs an explicit storage-policy-allow note: ' + name)

    def test_templates_only_point_into_new_base_or_tool_prefixes(self):
        offenders = []
        for path in sorted((HERE / 'templates').glob('*.json')):
            payload = json.loads(path.read_text(encoding='utf-8'))
            for pointer in absolute_strings(payload):
                if not pointer.startswith(EXPECTED_PREFIX):
                    continue
                if pointer.startswith(POLICY.NON_DATA_ROOT_PREFIXES):
                    continue
                offenders.append((path.name, pointer))
        self.assertEqual(offenders, [], 'template path outside new base: %r' % offenders)


def absolute_strings(value):
    """递归取出 JSON 中所有 /root 或 /shared 开头的字符串值。"""
    if isinstance(value, str):
        if value.startswith('/root/') or value.startswith('/shared/'):
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from absolute_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from absolute_strings(item)


if __name__ == '__main__':
    unittest.main()
