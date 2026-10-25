"""Real-filesystem fault, fencing and immutable publication regressions."""
from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contracts import Refused, atomic_new, canonical_bytes, digest, job_key, sha_file, stat_identity
from work_queue import WorkQueue
from sidecar import SidecarStore, StageCache
import sidecar as publication
from test_quality_schema import fixture
import schema


class PublicationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="redim-schema-publication-");self.root=Path(self.temp.name)
        args=list(fixture());wav=self.root/'a.wav'
        with wave.open(str(wav),'wb') as f:
            f.setnchannels(2);f.setsampwidth(2);f.setframerate(44100);f.writeframes(b'\x00'*88200*2*2)
        args[0]['audio_path']=str(wav)
        args[1].update(bytes=wav.stat().st_size,file_sha256=sha_file(wav),stat_before=stat_identity(wav),stat_after=stat_identity(wav))
        self.args=args;self.source=args[0];self.contract_sha=args[2]['sha256']
        shard=self.root/'chunk.jsonl';shard.write_bytes(canonical_bytes(self.source))
        self.index=self.root/'index.json';self.index.write_bytes(canonical_bytes({'schema':'redimnet2-shard-index/v1','source_count':1,
            'shards':[{'id':'chunk-000','path':shard.name,'count':1,'sha256':sha_file(shard)}]}))
        self.index_sha=sha_file(self.index)
        self.queue=WorkQueue(self.index,self.index_sha,self.root/'queue',self.contract_sha)
        self.store=SidecarStore(self.root/'labels',{'version':'fixture-v1'},self.contract_sha)
        self.claims=[]

    def tearDown(self):
        for c in self.claims:c.close()
        self.temp.cleanup()

    def claim(self, worker='worker-0'):
        c=self.queue.claim_next(worker)
        if c:self.claims.append(c)
        return c

    def record(self,claim,source=None):
        args=deepcopy(self.args)
        if source is not None:args[0]=source
        args[6].update(job_key=job_key(args[0],self.contract_sha),attempt_id=claim.attempt_id,worker_id=claim.token['worker_id'])
        return schema.build_record(*args)

    def test_complete_and_resume_do_not_duplicate_or_reinfer(self):
        c=self.claim();r=self.record(c);receipt=self.store.publish(self.source,r,c);c.finish([receipt]);c.close()
        self.assertTrue(self.queue.is_complete());raw=self.store.path(self.source).read_bytes()
        resumed=WorkQueue(self.index,self.index_sha,self.root/'queue',self.contract_sha)
        self.assertTrue(resumed.is_complete());self.assertIsNone(resumed.claim_next('new-worker'))
        self.assertEqual(self.store.lookup(self.source)['sidecar_sha256'],receipt['sidecar_sha256'])
        self.assertEqual(raw,self.store.path(self.source).read_bytes())

    def test_crash_after_sidecar_before_commit_recovers_existing_output(self):
        c=self.claim();r=self.record(c)
        def fault(stage):
            if stage=='after_sidecar_before_commit':raise RuntimeError('injected after sidecar')
        with self.assertRaisesRegex(RuntimeError,'injected'):self.store.publish(self.source,r,c,fault)
        p=self.store.path(self.source);old=p.read_bytes();self.assertFalse(p.with_name(p.name+'.commit.json').exists());c.close()
        n=self.claim('recovery');self.assertEqual(n.token['generation'],2)
        recovered=self.store.lookup(self.source);self.assertEqual(old,p.read_bytes())
        n.finish([recovered]);self.assertTrue(self.queue.is_complete())

    def test_crash_before_output_recovers_with_new_generation(self):
        c=self.claim()
        def fault(stage):
            if stage=='before_sidecar':raise RuntimeError('injected before sidecar')
        with self.assertRaises(RuntimeError):self.store.publish(self.source,self.record(c),c,fault)
        self.assertFalse(self.store.path(self.source).exists());c.close()
        n=self.claim('recovery');self.assertEqual(n.token['generation'],2)
        receipt=self.store.publish(self.source,self.record(n),n);n.finish([receipt]);self.assertTrue(self.queue.is_complete())

    def test_partial_temp_is_preserved_and_never_published(self):
        c=self.claim();r=self.record(c);path=self.store.path(self.source);tmp=path.with_name('.injected-owned.tmp')
        real=publication.atomic_new
        def partial(p,value):
            Path(p).parent.mkdir(parents=True,exist_ok=True);tmp.write_bytes(b'{"incomplete":')
            raise RuntimeError('injected partial temp')
        with patch.object(publication,'atomic_new',partial):
            with self.assertRaises(RuntimeError):self.store.publish(self.source,r,c)
        self.assertFalse(path.exists());self.assertEqual(tmp.read_bytes(),b'{"incomplete":')
        receipt=self.store.publish(self.source,r,c);c.finish([receipt])
        self.assertEqual(tmp.read_bytes(),b'{"incomplete":');self.assertTrue(self.queue.is_complete())

    def test_old_claimed_time_does_not_steal_live_lock(self):
        with patch('work_queue.utcnow',return_value='1900-01-01T00:00:00Z'):c=self.claim()
        self.assertIsNone(self.queue.claim_next('contender'));c.assert_current()

    def test_stale_generation_callback_refused(self):
        old=self.claim();token=deepcopy(old.token);old.close();new=self.claim('new-worker')
        fake=copy(new);fake.token=token
        with self.assertRaisesRegex(Refused,'stale attempt'):fake.assert_current()
        new.assert_current()

    def test_foreign_source_cannot_finish_same_count_chunk(self):
        c=self.claim()
        with self.assertRaisesRegex(Refused,'foreign sources'):
            c.finish([{'source_id':'src_'+'e'*64,'job_key':'invented'}])
        self.assertFalse(self.queue.chunk_receipt_path(c.shard['id']).exists())

    def test_publication_requires_membership_of_pinned_chunk(self):
        c=self.claim();foreign=deepcopy(self.source);foreign['relative_path']='speaker/b.wav';foreign['sidecar_relpath']='fixture/speaker/b.wav.annotation.json'
        foreign['source_id']='src_'+hashlib.sha256((foreign['dataset']+'\0'+foreign['relative_path']).encode()).hexdigest()
        with self.assertRaisesRegex(Refused,'foreign or changed source'):
            self.store.publish(foreign,self.record(c,foreign),c)
        self.assertFalse(self.store.path(foreign).exists())

    def test_corrupt_existing_sidecar_not_overwritten(self):
        c=self.claim();path=self.store.path(self.source);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'{bad-json')
        with self.assertRaises((ValueError,Refused)):self.store.lookup(self.source)
        self.assertEqual(path.read_bytes(),b'{bad-json')

    def test_output_version_cannot_switch_contract(self):
        with self.assertRaisesRegex(Refused,'another contract'):
            SidecarStore(self.root/'labels',{'version':'fixture-v1'},'a'*64)

    def test_stage_cache_verified_winner_survives_different_losing_output(self):
        cache=StageCache(self.root/'cache');inputs={'source_pcm':'1'*64,'model':'2'*64,'target':'repeated repeated'}
        real=publication.atomic_new
        def racing_winner(path,loser):
            winner=deepcopy(loser);winner['output']={'native':'winner'};winner['output_sha256']=digest(winner['output'])
            real(path,winner)
            return real(path,loser)
        with patch.object(publication,'atomic_new',racing_winner):
            self.assertEqual(cache.put('asr',inputs,{'native':'loser'}),{'native':'winner'})
        self.assertEqual(cache.get('asr',inputs),{'native':'winner'})

    def test_stage_cache_corrupt_winner_does_not_hide_failure(self):
        cache=StageCache(self.root/'cache');inputs={'a':1}
        def corrupt(path,value):
            Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text('{bad')
            raise Refused('injected existing invalid output')
        with patch.object(publication,'atomic_new',corrupt):
            with self.assertRaises((ValueError,Refused)):cache.put('asr',inputs,{'native':'loser'})


if __name__=='__main__':unittest.main()
