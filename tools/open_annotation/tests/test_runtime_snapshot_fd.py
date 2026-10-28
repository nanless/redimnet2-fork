"""只用临时小文件核CPU证明复用；不读取模型、原音或正式材料。"""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bounded_speed_runtime as M


class SnapshotFDTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.folder=self.root/'inputs';self.folder.mkdir()
        self.path=self.folder/'proof.json';self.path.write_bytes(b'original-proof')
        self.pin={'path':str(self.path),'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()}
        self.snapshot=M._snapshot_verified_pins([self.pin])

    def test_actual_complete_SHA_and_unique_ancestor_checks_without_per_file_physical_walk(self):
        second=self.folder/'another.json';second.write_bytes(b'another-proof')
        snap=M._snapshot_verified_pins([self.pin,{'path':str(second),'sha256':hashlib.sha256(second.read_bytes()).hexdigest()}])
        read0=os.read;reads=[]
        def observed(fd,size):
            block=read0(fd,size);reads.append(len(block));return block
        with patch.object(M,'_directory_stamp',wraps=M._directory_stamp)as directories,patch.object(os,'read',side_effect=observed),patch.object(M.R,'pinned_file',side_effect=AssertionError('repeated physical ancestor walk forbidden')):
            M._verify_snapshot(snap)
        self.assertEqual(directories.call_count,2*len(snap['directories']))
        self.assertEqual(sum(reads),len(b'original-proof')+len(b'another-proof'))
        self.assertEqual(len(reads),4)

    def test_symlink_final_component_is_not_followed(self):
        target=self.folder/'target.json';self.path.rename(target);self.path.symlink_to(target)
        with patch.object(os,'read',side_effect=AssertionError('must not read symlink target')):
            with self.assertRaises((ValueError,OSError)):M._verify_snapshot(self.snapshot)

    @unittest.skipUnless(hasattr(os,'mkfifo'),'requires POSIX FIFO')
    def test_FIFO_is_nonblocking_opened_then_rejected_before_any_read(self):
        open0=os.open;flags=[];stamp0=M._file_stamp;replaced=False
        def stamp_then_replace(path):
            nonlocal replaced
            result=stamp0(path)
            if str(path)==str(self.path)and not replaced:
                replaced=True;self.path.unlink();os.mkfifo(self.path)
            return result
        def opened(path,value,*args,**kwargs):
            if str(path)==str(self.path):flags.append(value)
            return open0(path,value,*args,**kwargs)
        with patch.object(M,'_file_stamp',side_effect=stamp_then_replace),patch.object(os,'open',side_effect=opened),patch.object(os,'read',side_effect=AssertionError('must not read FIFO')):
            with self.assertRaisesRegex(ValueError,'identity/content timestamp'):M._verify_snapshot(self.snapshot)
        self.assertEqual(len(flags),1);self.assertTrue(flags[0]&os.O_NONBLOCK);self.assertTrue(flags[0]&os.O_NOFOLLOW)

    @unittest.skipUnless(hasattr(os,'mkfifo'),'requires POSIX FIFO')
    def test_preexisting_FIFO_is_refused_without_even_opening_it(self):
        self.path.unlink();os.mkfifo(self.path)
        with patch.object(os,'open',side_effect=AssertionError('FIFO should reject at path precheck')):
            with self.assertRaisesRegex(ValueError,'regular nonlinked'):M._verify_snapshot(self.snapshot)

    def test_wrong_opened_FD_cannot_substitute_identical_bytes_from_another_inode(self):
        other=self.folder/'same-bytes.json';other.write_bytes(self.path.read_bytes());open0=os.open
        def wrong(path,*args,**kwargs):return open0(other if str(path)==str(self.path)else path,*args,**kwargs)
        with patch.object(os,'open',side_effect=wrong),patch.object(os,'read',side_effect=AssertionError('inode must be checked first')):
            with self.assertRaisesRegex(ValueError,'identity/content timestamp'):M._verify_snapshot(self.snapshot)

    def test_read_midstream_change_is_rejected_and_own_FD_closed(self):
        self.path.write_bytes(b'a'*(8*1024*1024+19));self.pin['sha256']=hashlib.sha256(self.path.read_bytes()).hexdigest();snap=M._snapshot_verified_pins([self.pin])
        read0=os.read;open0=os.open;opened=[];changed=False
        def opening(path,*args,**kwargs):
            fd=open0(path,*args,**kwargs);opened.append(fd);return fd
        def reading(fd,size):
            nonlocal changed
            block=read0(fd,size)
            if block and not changed:
                changed=True
                with self.path.open('r+b')as f:f.seek(4*1024*1024+1);f.write(b'b')
            return block
        with patch.object(os,'open',side_effect=opening),patch.object(os,'read',side_effect=reading):
            with self.assertRaisesRegex(ValueError,'during SHA read|pinned file differs'):M._verify_snapshot(snap)
        self.assertTrue(changed);self.assertEqual(len(opened),1)
        with self.assertRaises(OSError):os.fstat(opened[0])

    def test_postread_path_replacement_is_refused_even_if_original_FD_bytes_match(self):
        read0=os.read;changed=False
        def reading(fd,size):
            nonlocal changed
            block=read0(fd,size)
            if block and not changed:
                changed=True;self.path.rename(self.folder/'held-original.json');self.path.write_bytes(block)
            return block
        with patch.object(os,'read',side_effect=reading):
            with self.assertRaisesRegex(ValueError,'during SHA read'):M._verify_snapshot(self.snapshot)

    def test_ancestor_mode_change_during_read_detected_by_final_unique_check(self):
        read0=os.read;changed=False
        def reading(fd,size):
            nonlocal changed
            block=read0(fd,size)
            if block and not changed:
                changed=True;os.chmod(self.folder,0o750 if self.folder.stat().st_mode&0o777!=0o750 else 0o700)
            return block
        with patch.object(os,'read',side_effect=reading):
            with self.assertRaisesRegex(ValueError,'ancestor identity changed during SHA read'):M._verify_snapshot(self.snapshot)

    def test_directory_replacement_before_read_is_refused(self):
        old=self.root/'former-inputs';self.folder.rename(old);self.folder.mkdir();self.path.write_bytes(b'original-proof')
        with patch.object(os,'read',side_effect=AssertionError('directory replacement must reject before input read')):
            with self.assertRaisesRegex(ValueError,'ancestor identity changed'):M._verify_snapshot(self.snapshot)

    def test_read_error_still_closes_only_opened_proof_FD(self):
        open0=os.open;opened=[]
        def opening(path,*args,**kwargs):
            fd=open0(path,*args,**kwargs);opened.append(fd);return fd
        with patch.object(os,'open',side_effect=opening),patch.object(os,'read',side_effect=OSError('synthetic read error')):
            with self.assertRaisesRegex(OSError,'synthetic read error'):M._verify_snapshot(self.snapshot)
        self.assertEqual(len(opened),1)
        with self.assertRaises(OSError):os.fstat(opened[0])

    def test_success_closes_FD_and_missing_ancestor_still_refused(self):
        open0=os.open;opened=[]
        def opening(path,*args,**kwargs):
            fd=open0(path,*args,**kwargs);opened.append(fd);return fd
        with patch.object(os,'open',side_effect=opening):M._verify_snapshot(self.snapshot)
        self.assertEqual(len(opened),1)
        with self.assertRaises(OSError):os.fstat(opened[0])
        missing=copy.deepcopy(self.snapshot);missing['directories'].pop()
        with self.assertRaisesRegex(ValueError,'ancestor coverage'):M._verify_snapshot(missing)


if __name__=='__main__':unittest.main()
