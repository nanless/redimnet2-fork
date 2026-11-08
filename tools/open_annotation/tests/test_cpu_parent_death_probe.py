"""探针控制器的失败收尾回归：真实自有 CPU child，不是 Linux 内核成功证据。

只把探针的平台/启动时钟读取替换为失败控制桩；不会执行合成 Linux
pool、生产音频、模型或 GPU。每个实际 Popen 都由测试外层保底回收。
"""
from contextlib import ExitStack, contextmanager, redirect_stderr
import ctypes
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import probe_retained_process_parent_death as P

REAL_POPEN = subprocess.Popen


@contextmanager
def without_attribute(module, name):
    """显式模拟老 Conda 缺失接口；结束时恢复原模块，不触发任何系统调用。"""
    absent = object()
    original = getattr(module, name, absent)
    if original is not absent:
        delattr(module, name)
    try:
        yield
    finally:
        if original is not absent:
            setattr(module, name, original)
        elif hasattr(module, name):
            delattr(module, name)


class CPUParentDeathProbeFailures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / 'failure-evidence'
        self.children = []
        self.addCleanup(self.temporary.cleanup)
        # unittest cleanup 后进先出：要先回收实际自有进程，再清临时证据。
        self.addCleanup(self.cleanup_owned_children)

    def cleanup_owned_children(self):
        for process in self.children:
            try:
                if process.poll() is None:
                    REAL_POPEN.terminate(process)
                try:
                    REAL_POPEN.wait(process, timeout=5)
                except subprocess.TimeoutExpired:
                    REAL_POPEN.kill(process)
                    REAL_POPEN.wait(process, timeout=5)
            finally:
                if process.stdin is not None:
                    process.stdin.close()

    def spawn_owned_cpu_child(self, argv, **kwargs):
        self.assertEqual(argv[-2], '--parent')
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(kwargs['stdin'], subprocess.PIPE)
        self.assertEqual({k: kwargs['env'].get(k) for k in P.U._CPU_ENV}, P.U._CPU_ENV)
        # 仅等待 stdin：TERM 或关闭管道都能结束，不运行探针 Linux parent。
        process = REAL_POPEN(
            [sys.executable, '-B', '-c', 'import sys; sys.stdin.buffer.read(1)'],
            **kwargs,
        )
        self.children.append(process)
        return process

    def controlled_failure_environment(self, *, start_error=None, writer=None, spawn=None):
        stack = ExitStack()
        stack.enter_context(patch.object(P.sys, 'platform', 'linux'))
        # Mac 不具备 pidfd；这里只让失败分支可达，且所有测试都零调用。
        pidfd = stack.enter_context(patch.object(P.os, 'pidfd_open', create=True))
        sender = stack.enter_context(patch.object(P.signal, 'pidfd_send_signal', create=True))
        stack.enter_context(patch.object(P.subprocess, 'Popen', side_effect=spawn or self.spawn_owned_cpu_child))
        stack.enter_context(patch.object(P.U, '_linux_start_ticks',
                                        side_effect=start_error, return_value='fixture-start-not-Linux-proof'))
        if writer is not None:
            stack.enter_context(patch.object(P, 'write_new', side_effect=writer))
        return stack, pidfd, sender

    def assert_owned_drained(self):
        self.assertEqual(len(self.children), 1)
        self.assertIsNotNone(self.children[0].poll(), 'run must reap its actual owned child before returning')
        self.assertTrue(self.children[0].stdin.closed)
        self.assertFalse((self.output / 'result.json').exists())
        self.assertFalse((self.output / 'parent-ready.json').exists())

    def test_created_receipt_enospc_reaps_real_owned_child_and_preserves_failure(self):
        original = P.write_new
        def writer(path, value):
            if Path(path).name == 'controller-created-parent.json':
                raise OSError('injected evidence ENOSPC')
            return original(path, value)
        context, pidfd, sender = self.controlled_failure_environment(writer=writer)
        with context:
            with self.assertRaisesRegex(RuntimeError, 'preserved'):
                P.run(self.output)
        self.assert_owned_drained()
        failure = json.loads((self.output / 'failure.json').read_text())
        self.assertIn('injected evidence ENOSPC', failure['traceback'])
        self.assertEqual(failure['parent']['pid'], self.children[0].pid)
        self.assertEqual(failure['cleanup_errors'], [])
        pidfd.assert_not_called(); sender.assert_not_called()

    def test_start_identity_read_failure_still_reaps_actual_owned_child(self):
        context, pidfd, sender = self.controlled_failure_environment(start_error=OSError('injected start identity read failure'))
        with context:
            with self.assertRaisesRegex(RuntimeError, 'preserved'):
                P.run(self.output)
        self.assert_owned_drained()
        failure = json.loads((self.output / 'failure.json').read_text())
        self.assertIsNone(failure['parent'])
        self.assertIn('injected start identity read failure', failure['traceback'])
        pidfd.assert_not_called(); sender.assert_not_called()

    def test_terminate_error_does_not_skip_pipe_close_final_wait_or_failure_receipt(self):
        original = P.write_new
        def writer(path, value):
            if Path(path).name == 'controller-created-parent.json':
                raise OSError('injected primary evidence error')
            return original(path, value)
        def spawn(argv, **kwargs):
            process = self.spawn_owned_cpu_child(argv, **kwargs)
            # 真 child 继续等待；本次 terminate 报错后要靠关闭 stdin 和最终 wait 回收。
            process.terminate = lambda: (_ for _ in ()).throw(OSError('injected terminate error'))
            return process
        context, pidfd, sender = self.controlled_failure_environment(writer=writer, spawn=spawn)
        with context:
            with self.assertRaisesRegex(RuntimeError, 'preserved'):
                P.run(self.output)
        self.assert_owned_drained()
        failure = json.loads((self.output / 'failure.json').read_text())
        self.assertIn('injected primary evidence error', failure['traceback'])
        self.assertEqual(len(failure['cleanup_errors']), 1)
        self.assertIn('injected terminate error', failure['cleanup_errors'][0])
        pidfd.assert_not_called(); sender.assert_not_called()

    def test_failure_receipt_error_reports_stderr_after_actual_child_cleanup(self):
        context, pidfd, sender = self.controlled_failure_environment(
            writer=lambda *args: (_ for _ in ()).throw(OSError('injected all evidence writes failed')))
        stderr = io.StringIO()
        with context, redirect_stderr(stderr):
            with self.assertRaisesRegex(RuntimeError, 'receipt could not be saved; details emitted to stderr'):
                P.run(self.output)
        self.assert_owned_drained()
        self.assertFalse((self.output / 'failure.json').exists())
        report = json.loads(stderr.getvalue())
        self.assertIn('injected all evidence writes failed', report['traceback'])
        self.assertIn('injected all evidence writes failed', report['failure_receipt_write_error'])
        pidfd.assert_not_called(); sender.assert_not_called()

    def test_nonlinux_refuses_before_popen_and_no_fake_kernel_success(self):
        with patch.object(P.sys, 'platform', 'darwin'), patch.object(P.subprocess, 'Popen') as create:
            with self.assertRaisesRegex(ValueError, 'requires actual Linux; no mock success'):
                P.run(self.output)
        create.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_fallback_header_wrong_number_or_name_refuses_before_libc(self):
        for header in ('#define __NR_pidfd_open 435\n',
                       '#define __NR_other 434\n',
                       '#define __NR_pidfd_open 4340\n'):
            with self.subTest(header=header), patch.object(P.sys, 'platform', 'linux'), \
                    patch.object(P.platform, 'machine', return_value='x86_64'), \
                    patch.object(P.Path, 'read_text', return_value=header), \
                    patch.object(P.ctypes, 'CDLL') as library:
                with self.assertRaisesRegex(ValueError, 'number not verified'):
                    P.verified_syscall(434, 'pidfd_open')
                library.assert_not_called()

    def test_fallback_unknown_architecture_refuses_before_header_or_libc(self):
        with patch.object(P.sys, 'platform', 'linux'), \
                patch.object(P.platform, 'machine', return_value='aarch64'), \
                patch.object(P.Path, 'read_text') as read, patch.object(P.ctypes, 'CDLL') as library:
            with self.assertRaisesRegex(ValueError, 'no verified pidfd fallback'):
                P.verified_syscall(434, 'pidfd_open')
            read.assert_not_called(); library.assert_not_called()

    def test_fallback_exact_header_selects_only_verified_local_number(self):
        for name, number in (('pidfd_open', 434), ('pidfd_send_signal', 424)):
            with self.subTest(name=name), patch.object(P.sys, 'platform', 'linux'), \
                    patch.object(P.platform, 'machine', return_value='x86_64'), \
                    patch.object(P.Path, 'read_text', return_value=f'#define __NR_{name} {number}\n') as read, \
                    patch.object(P.ctypes, 'CDLL') as library:
                self.assertIs(P.verified_syscall(number, name), library.return_value)
                read.assert_called_once_with()
                library.assert_called_once_with(None, use_errno=True)
                self.assertIs(library.return_value.syscall.restype, ctypes.c_long)

    def test_fallback_open_wrong_pid_or_fd_kind_closes_actual_owned_descriptor(self):
        for info, kind in (('Pid:\t124\n', 'anon_inode:[pidfd]'),
                           ('Pid:\t123\n', '/dev/null')):
            with self.subTest(info=info, kind=kind):
                descriptor = os.open(os.devnull, os.O_RDONLY)
                library = Mock(); library.syscall.return_value = descriptor
                try:
                    with without_attribute(P.os, 'pidfd_open'), \
                            patch.object(P, 'verified_syscall', return_value=library) as verify, \
                            patch.object(P.Path, 'read_text', return_value=info), \
                            patch.object(P.os, 'readlink', return_value=kind):
                        with self.assertRaisesRegex(ValueError, 'does not bind exact'):
                            P.open_exact_pidfd(123)
                    verify.assert_called_once_with(434, 'pidfd_open')
                    self.assertEqual([value.value for value in library.syscall.call_args.args], [434, 123, 0])
                    with self.assertRaises(OSError):
                        os.fstat(descriptor)
                finally:
                    try: os.close(descriptor)
                    except OSError: pass

    def test_fallback_open_verified_identity_keeps_noninheritable_owned_fd(self):
        descriptor = os.open(os.devnull, os.O_RDONLY)
        library = Mock(); library.syscall.return_value = descriptor
        try:
            with without_attribute(P.os, 'pidfd_open'), \
                    patch.object(P, 'verified_syscall', return_value=library), \
                    patch.object(P.Path, 'read_text', return_value='Pid:\t123\n'), \
                    patch.object(P.os, 'readlink', return_value='anon_inode:[pidfd]'):
                self.assertEqual(P.open_exact_pidfd(123), descriptor)
            self.assertFalse(os.get_inheritable(descriptor))
            os.fstat(descriptor)
        finally:
            os.close(descriptor)

    def test_fallback_kill_uses_descriptor_and_no_bare_pid_signal(self):
        library = Mock(); library.syscall.return_value = 0
        with without_attribute(P.signal, 'pidfd_send_signal'), \
                patch.object(P, 'verified_syscall', return_value=library) as verify, \
                patch.object(P.os, 'kill') as bare_pid_kill:
            P.kill_exact_pidfd(37)
        verify.assert_called_once_with(424, 'pidfd_send_signal')
        values = [value.value for value in library.syscall.call_args.args]
        self.assertEqual(values, [424, 37, signal.SIGKILL, None, 0])
        bare_pid_kill.assert_not_called()

    def test_fallback_negative_syscalls_raise_instead_of_claiming_success(self):
        library = Mock(); library.syscall.return_value = -1
        with without_attribute(P.os, 'pidfd_open'), without_attribute(P.signal, 'pidfd_send_signal'), \
                patch.object(P, 'verified_syscall', return_value=library), \
                patch.object(P.ctypes, 'get_errno', return_value=38), patch.object(P.os, 'kill') as bare_pid_kill:
            with self.assertRaisesRegex(OSError, 'pidfd_open failed'):
                P.open_exact_pidfd(123)
            with self.assertRaisesRegex(OSError, 'pidfd_send_signal failed'):
                P.kill_exact_pidfd(37)
        bare_pid_kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
