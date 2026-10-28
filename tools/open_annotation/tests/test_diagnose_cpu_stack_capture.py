"""只测合成输入和默认不执行守卫；不运行真实封存或模型。"""
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import diagnose_cpu_stack_capture as D


class StackDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.source=self.root/'source';self.source.mkdir();self.schema=self.source/'schema.py'
        self.schema.write_text("raise RuntimeError('whole module must never import')\ndef _finite_tree(value,path='record'):\n return [] if isinstance(value,(dict,list)) else ['wrong']\n")
        self.sha=hashlib.sha256(self.schema.read_bytes()).hexdigest();self.out=self.root/'evidence'
    def test_only_pinned_function_compiles_without_running_module_body(self):
        f=D.source_function(self.schema,self.sha);self.assertEqual(f({'synthetic':[1,2]}),[])
        with self.assertRaisesRegex(ValueError,'SHA'):D.source_function(self.schema,'0'*64)
    def test_default_check_cannot_spawn_or_create_output(self):
        with patch.object(D.subprocess,'run',side_effect=AssertionError('check must not spawn')):
            result=D.run(self.schema,self.sha,self.out)
        self.assertTrue(result['checked']);self.assertFalse(result['executed']);self.assertFalse(self.out.exists())
    def test_duration_interval_and_existing_output_are_bounded(self):
        for seconds,interval in ((16,.01),(True,.01),(0,.01),(10,0),(10,1),(float('nan'),.01)):
            with self.subTest(seconds=seconds,interval=interval),self.assertRaises(ValueError):D.run(self.schema,self.sha,self.out,seconds=seconds,interval=interval)
        self.out.mkdir()
        with self.assertRaisesRegex(ValueError,'fresh'):D.run(self.schema,self.sha,self.out)


if __name__=='__main__':unittest.main()
