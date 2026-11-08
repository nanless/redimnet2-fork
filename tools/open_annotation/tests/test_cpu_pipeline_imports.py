"""在仓库外cwd导入新模块，防止意外导入真实GPU推理库。"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
MODULES=('union_coverage','union_consumer','engine_protocol','audit_engine_comparison',
         'asr_stage_store','vllm_backend','cloud_gate','vllm_live_probe','prepare_vllm_live_inputs',
         'portable_source_identity','freeze_legacy_successor','run_portable_successor',
         'finalize_remaining_roles','verify_portable_successor_mounts','benchmark_retained_reads',
         'probe_retained_process_parent_death')


class CPUImportBoundary(unittest.TestCase):
    def child(self, body):
        guard='''import builtins, importlib, sys, runpy
sys.dont_write_bytecode=True
sys.path.insert(0, %r)
original=builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','qwen_asr','vllm'}:
        raise AssertionError('heavy inference framework imported by CPU entry: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=checked
''' % str(HERE)
        with tempfile.TemporaryDirectory() as d:
            p=subprocess.run([sys.executable,'-B','-c',guard+body],cwd=Path(d).resolve(),
                             capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_new_modules_are_lazy_and_independent_of_current_directory(self):
        self.child('for name in '+repr(MODULES)+': importlib.import_module(name)\n')

    def test_CLI_help_does_not_construct_or_import_real_engine(self):
        # 每个help在独立子进程退出，不携带上一入口的sys.argv或模块状态。
        for name in ('union_coverage','union_consumer','audit_engine_comparison','vllm_backend','cloud_gate',
                     'vllm_live_probe','prepare_vllm_live_inputs','freeze_legacy_successor','run_portable_successor',
                     'finalize_remaining_roles','verify_portable_successor_mounts','benchmark_retained_reads',
                     'probe_retained_process_parent_death'):
            with self.subTest(entry=name):
                self.child('sys.argv=['+repr(name+'.py')+',"--help"]\nrunpy.run_path('+repr(str(HERE/(name+'.py')))+',run_name="__main__")\n')


if __name__=='__main__':unittest.main()
