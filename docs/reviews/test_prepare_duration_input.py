"""根样本准备入口的坏记录计账和边界测试，不读取原音或运行模型。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('prepared_duration_input', ROOT/'scripts/prepare_duration_finetune.py')
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class PrepareInputTests(unittest.TestCase):
    def test_bad_shape_records_are_counted_before_any_media_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input.jsonl'
            records=[None,[],7,{'key':3,'spk':'A_s','wav':'/synthetic.wav'},
                     {'key':'A_item','spk':None,'wav':'/synthetic.wav'}]
            source.write_text(''.join(json.dumps(row)+'\n' for row in records))
            with patch.object(M,'inspect_record',side_effect=AssertionError('坏结构不得读取媒体')):
                with self.assertRaisesRegex(ValueError,'bounded selection incomplete'):
                    M.prepare(source,root/'rttm',root/'new-output',train_speakers=1,heldout_speakers=1,
                              utterances=1,candidate_speakers_per_corpus=2)
            report=json.loads((root/'new-output/selection_report.json').read_text())
            self.assertFalse(report['selection_complete'])
            self.assertEqual(report['excluded']['bad_record'],len(records))
            self.assertEqual(report['scanned_lines'],len(records))

    def test_invalid_limits_refused_before_output_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'not-created'
            for key in ('train_speakers','heldout_speakers','utterances','max_lines',
                        'max_checked_per_corpus','candidate_speakers_per_corpus'):
                for value in (True,0,1.5):
                    with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                        M.prepare(root/'missing-input',root/'rttm',output,**{key:value})
                    self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
