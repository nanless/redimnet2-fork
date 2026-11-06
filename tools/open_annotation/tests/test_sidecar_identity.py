"""P2「内容身份等式排除 st_dev」的最小回归（纯逻辑，无 torch / 无真实文件）。

覆盖三条：只有 device 不同 → 接受且打标；inode / bytes / mtime / ctime 任一不同 → 拒绝；
字段缺失 → 拒绝（不能因为 recorded 缺字段就默认相等）。
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import sidecar  # noqa: E402

BASE = {'device': 2097190, 'inode': 26453836791, 'bytes': 367438,
        'mtime_ns': 1706434133000000000, 'ctime_ns': 1791867955838457274}


def rec(**over):
    value = dict(BASE)
    value.update(over)
    return value


class IdentityMatchesTests(unittest.TestCase):
    def test_device_only_change_is_accepted_and_flagged(self):
        now = rec(device=57)
        relaxed = sidecar.identity_matches(now, BASE)
        self.assertTrue(relaxed, 'device 漂移必须被标记为 device_relaxed')

    def test_same_device_is_not_flagged(self):
        self.assertFalse(sidecar.identity_matches(rec(), BASE))

    def test_each_content_field_mismatch_is_refused(self):
        for key, bad in (('inode', 1), ('bytes', 1), ('mtime_ns', 1), ('ctime_ns', 1)):
            with self.subTest(field=key):
                with self.assertRaises(Exception) as caught:
                    sidecar.identity_matches(rec(**{key: bad}), BASE)
                self.assertIn(key, str(caught.exception),
                              '拒绝理由必须点名出错字段，便于定位')

    def test_missing_recorded_field_is_refused(self):
        partial = dict(BASE)
        partial.pop('inode')
        with self.assertRaises(Exception):
            sidecar.identity_matches(rec(), partial)


if __name__ == '__main__':
    unittest.main(verbosity=2)
