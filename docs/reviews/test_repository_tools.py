"""仓库检查与CPU汇总的独立合成回归，不访问网络、模型或GPU。"""
import importlib.util
import errno
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

    def execute_case(self, prelude='', skipped=False, empty=False, expected_failure=False):
        code = "import unittest\n" + prelude + '\n'
        if not empty:
            code += "class Example(unittest.TestCase):\n"
            if skipped:
                code += "    @unittest.skip('synthetic unavailable dependency')\n"
            if expected_failure:
                code += "    @unittest.expectedFailure\n"
            code += "    def test_one(self): self.assertEqual(1, %d)\n" % (2 if expected_failure else 1)
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

    def test_runner_expected_failure_is_not_full_acceptance(self):
        actual = self.execute_case(expected_failure=True)
        self.assertEqual(actual['exit_code'], 0)
        self.assertEqual(actual['reported_expected_failures'], 1)
        self.assertFalse(actual['passed_without_skip'])

    def test_timeout_reaps_own_ignoring_child_after_parent_exits(self):
        child_code = ("import os,signal,sys,time;from pathlib import Path;"
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                      "Path(sys.argv[1]).write_text(str(os.getpid()));"
                      "time.sleep(20);Path(sys.argv[2]).write_text('survived')")
        pid_file=self.path/'child.pid';marker=self.path/'survived'
        parent_code = ("import subprocess,sys,time;from pathlib import Path;"
                       "subprocess.Popen([sys.executable,'-c',"+repr(child_code)+","+repr(str(pid_file))+","+repr(str(marker))+"]);"
                       "time.sleep(30)")
        actual=self.runner.execute('timeout_child',['-B','-c',parent_code],self.path,1,
                                   dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        self.assertTrue(pid_file.exists())
        self.assertTrue(actual['timeout'])
        self.assertFalse(actual['passed_without_skip'])
        self.assertTrue(actual['timeout_cleanup']['complete'],actual)
        self.assertTrue(any(a['signal']==9 for a in actual['timeout_cleanup']['actions']))
        self.assertFalse(marker.exists())

    def teardown_probe_case(self, *, persistent=False, deny_signal=False):
        process=SimpleNamespace(pid=424242,poll=lambda: -15)
        state={'signal':None,'kill_probes':0,'clock':0.}
        def clock():
            state['clock']+=.001
            return state['clock']
        def send(group,sig):
            state['signal']=int(sig)
            if deny_signal and int(sig)==9:
                raise PermissionError(errno.EPERM,'synthetic signal denial')
        def exists(proc):
            if state['signal']!=9:return True
            state['kill_probes']+=1
            if persistent or state['kill_probes']==1:
                raise PermissionError(errno.EPERM,'synthetic post-KILL teardown probe')
            return False  # A fresh ESRCH interpretation from own_group_exists.
        with patch.object(self.runner,'own_group_exists',side_effect=exists),\
             patch.object(self.runner.os,'killpg',side_effect=send),\
             patch.object(self.runner.time,'monotonic',side_effect=clock),\
             patch.object(self.runner.time,'sleep'):
            return self.runner.terminate_own_session(process,grace_seconds=.02)

    def test_transient_teardown_permission_requires_fresh_absence(self):
        result=self.teardown_probe_case()
        self.assertTrue(result['complete'])
        self.assertFalse(result['remaining_group_present'])
        self.assertEqual(result['probe_errors'][0]['phase'],'after_SIGKILL')
        self.assertEqual(result['probe_errors'][0]['errno'],errno.EPERM)

    def test_persistent_permission_remains_unknown_not_clean(self):
        result=self.teardown_probe_case(persistent=True)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['remaining_group_present'])
        self.assertTrue(result['probe_errors'])

    def test_denied_signal_is_not_treated_as_transient_probe(self):
        result=self.teardown_probe_case(deny_signal=True)
        self.assertFalse(result['complete'])
        self.assertEqual(result['actions'][-1]['result'],'denied')
        self.assertEqual(result['actions'][-1]['phase'],'send_SIGKILL')
        self.assertEqual(result['actions'][-1]['errno'],errno.EPERM)

    def test_static_result_never_overwrites_existing_source(self):
        self.init_git()
        source = self.path/'maintained.py'
        source.write_text('answer = 42\n')
        before = source.read_bytes()
        self.static.ROOT = self.path
        with self.assertRaises(SystemExit) as rejected:
            self.static.main(['--output', str(source)])
        self.assertEqual(rejected.exception.code, 2)
        self.assertEqual(source.read_bytes(), before)

    def test_static_rejects_external_and_dangling_symlink_without_reading(self):
        self.init_git()
        outside = self.path.parent/(self.path.name + '-outside.py')
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        outside.write_bytes(b'\xff-private-input-must-not-be-parsed')
        (self.path/'alias.py').symlink_to(outside)
        (self.path/'missing.py').symlink_to(self.path/'missing-target')
        actual = self.static.check(self.path)
        self.assertEqual({(x['file'], x['check']) for x in actual['errors']},
                         {('alias.py', 'symlink'), ('missing.py', 'symlink')})

    def test_static_never_reads_cached_file_through_replaced_directory(self):
        self.init_git()
        nested=self.path/'nested';nested.mkdir()
        source=nested/'source.py';source.write_text('value=1\n')
        subprocess.run(['git','-C',str(self.path),'add','nested/source.py'],check=True,capture_output=True)
        source.unlink();nested.rmdir()
        with tempfile.TemporaryDirectory() as outside:
            (Path(outside)/'source.py').write_bytes(b'\xff-must-not-be-read')
            nested.symlink_to(outside,target_is_directory=True)
            actual=self.static.check(self.path)
        self.assertFalse(actual['passed'])
        self.assertEqual({(x['file'],x['check']) for x in actual['errors']},
                         {('nested','symlink'),('nested/source.py','symlink')})

    def test_static_rejects_nonfinite_and_duplicate_JSON(self):
        self.init_git()
        for name, text in [('nan.json', '{"lr": NaN}'), ('inf.json', '{"x": Infinity}'),
                           ('overflow.json', '{"lr":1e999}'),
                           ('duplicate.json', '{"scope":1,"scope":2}')]:
            (self.path/name).write_text(text)
        actual = self.static.check(self.path)
        self.assertFalse(actual['passed'])
        self.assertEqual({x['file'] for x in actual['errors']}, {'nan.json', 'inf.json', 'overflow.json', 'duplicate.json'})

    def test_static_missing_link_keeps_line_number_after_code_block(self):
        self.init_git()
        doc = self.path/'README.md'
        doc.write_text('# 文档\n```\n[示例](not-checked.md)\n```\n[缺失](missing.md)\n')
        errors = self.static.local_links(doc, self.path)
        self.assertEqual(len(errors), 1)
        self.assertEqual((errors[0]['line'], errors[0]['target']), (5, 'missing.md'))

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
