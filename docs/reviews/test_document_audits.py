"""说明检查器的拒绝分支，只使用临时目录和公开仓库表格。"""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DocumentAudits(unittest.TestCase):
    def test_private_environment_pruned_without_git(self):
        check = load('check_document_links')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'README.md').write_text('# public')
            for name in ('.venv', '.audit-local', 'node_modules'):
                (root / name).mkdir()
                (root / name / 'README.md').write_bytes(b'\xff-invalid-private-content')
            self.assertEqual(check.inspect(root)['document_count'], 1)

    def test_git_ignored_private_files_are_not_read(self):
        check = load('check_document_links')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(['git', 'init', str(root)], check=True, capture_output=True)
            (root / '.gitignore').write_text('private-receipts/\n')
            (root / 'README.md').write_text('# public')
            (root / 'private-receipts').mkdir()
            (root / 'private-receipts/README.md').write_bytes(b'\xff-invalid-private-content')
            result = check.inspect(root)
            self.assertEqual(result['document_count'], 1)
            self.assertTrue(result['file_selection'].startswith('git ls-files'))

    def test_missing_link_and_excluded_example(self):
        check = load('check_document_links')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'README.md').write_text('[空]( )\n[现存](a%20b.md)\n[缺少](missing.md)\n```\n[例子](example.md)\n```\n<img src="pic.png">')
            (root / 'a b.md').write_text('# 文件\n')
            (root / 'pic.png').write_bytes(b'synthetic-existence-only')
            result = check.inspect(root)
            self.assertEqual([item['target'] for item in result['missing_targets']], ['missing.md'])
            self.assertEqual(len(result['local_links']), 3)

    def test_changed_report_number_is_detected(self):
        check = load('check_report_tables')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('eval_vox1_O_all.csv', 'eval_vox1_EH_all.csv', 'eval_ood_all.csv',
                         'reports/02-VoxCeleb1结果.md', 'reports/03-域外试验结果.md'):
                target = root / 'checkpoints' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(check.ROOT / 'checkpoints' / name, target)
            check.ROOT = root
            original = check.check()
            self.assertEqual(original['checked_numeric_cells'], 192)
            self.assertEqual(original['errors'], [])
            report = root / 'checkpoints/reports/02-VoxCeleb1结果.md'
            report.write_text(report.read_text().replace('| b0-vox2-lm | 1.042 |', '| b0-vox2-lm | 9.042 |', 1))
            errors = check.check()['errors']
            self.assertEqual(len(errors), 1)
            self.assertEqual((errors[0]['checkpoint'], errors[0]['field']), ('b0-vox2-lm', 'Vox1-O'))


if __name__ == '__main__':
    unittest.main()
