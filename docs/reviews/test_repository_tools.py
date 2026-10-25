"""仓库检查与CPU汇总的独立合成回归，不访问网络、模型或GPU。"""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepositoryToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='repository-tool-review-')
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.static = load('independent_static', 'scripts/check_repository.py')
        self.runner = load('independent_runner', 'scripts/run_cpu_checks.py')
        self.runner.ROOT = self.path

    def execute_case(self, prelude='', skipped=False, empty=False):
        code = "import unittest\n" + prelude + '\n'
        if not empty:
            code += "class Example(unittest.TestCase):\n"
            if skipped:
                code += "    @unittest.skip('synthetic unavailable dependency')\n"
            code += "    def test_one(self): self.assertEqual(1, 1)\n"
        code += "unittest.main()\n"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        return self.runner.execute('case', ['-B', '-c', code], self.path, 10, env)

    def init_git(self):
        subprocess.run(['git', 'init', '-q', str(self.path)], check=True, capture_output=True)

    def test_runner_real_success(self):
        actual = self.execute_case()
        self.assertEqual(actual['reported_unittest_count'], 1)
        self.assertTrue(actual['passed_without_skip'])

    def test_runner_real_skip_is_not_full_acceptance(self):
        actual = self.execute_case(skipped=True)
        self.assertEqual(actual['reported_skipped'], 1)
        self.assertFalse(actual['passed_without_skip'])

    def test_runner_stdout_cannot_hide_real_skip(self):
        actual = self.execute_case("print('skipped=0')", skipped=True)
        self.assertEqual(actual['reported_skipped'], 1)
        self.assertFalse(actual['passed_without_skip'])

    def test_runner_stdout_cannot_inflate_real_test_count(self):
        actual = self.execute_case("print('Ran 999 tests in imaginary seconds')")
        self.assertEqual(actual['reported_unittest_count'], 1)

    def test_runner_zero_discovered_tests_is_not_acceptance(self):
        actual = self.execute_case(empty=True)
        self.assertEqual(actual['reported_unittest_count'], 0)
        self.assertFalse(actual['passed_without_skip'])

    def test_static_sees_untracked_invalid_files(self):
        self.init_git()
        (self.path/'broken.py').write_text('def broken(:\n', encoding='utf-8')
        (self.path/'broken.json').write_text('{broken}', encoding='utf-8')
        (self.path/'broken.sh').write_text('if then\n', encoding='utf-8')
        (self.path/'README.md').write_text('[missing](not-present.md)\n', encoding='utf-8')
        actual = self.static.check(self.path)
        self.assertFalse(actual['passed'])
        self.assertEqual({x['file'] for x in actual['errors']},
                         {'broken.py', 'broken.json', 'broken.sh', 'README.md'})

    def test_static_honors_owned_gitignore(self):
        self.init_git()
        (self.path/'.gitignore').write_bytes((ROOT/'.gitignore').read_bytes())
        for name in ['audio.wav', 'weights.pt', 'tensor.npy']:
            (self.path/name).write_bytes(b'synthetic, not actual media')
        (self.path/'.audit-local').mkdir()
        (self.path/'.audit-local'/'invalid.py').write_text('def broken(:\n', encoding='utf-8')
        (self.path/'maintained.py').write_text('value = 1\n', encoding='utf-8')
        files = {x.relative_to(self.path).as_posix() for x in self.static.repository_files(self.path)}
        self.assertEqual(files, {'.gitignore', 'maintained.py'})
        self.assertTrue(self.static.check(self.path)['passed'])

    def test_static_refuses_parent_repository_identity(self):
        self.init_git()
        nested = self.path/'nested'
        nested.mkdir()
        with self.assertRaisesRegex(ValueError, 'Git根目录身份'):
            self.static.repository_files(nested)


if __name__ == '__main__':
    unittest.main()
