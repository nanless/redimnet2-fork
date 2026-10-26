"""Synthetic fresh terminal references; no platform/GPU/training locks."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.ptn_resume import refresh_terminal_refs as R


class RefreshTests(unittest.TestCase):
    def setup_case(self,root):
        base=root/'base';base.mkdir();run=base/'ptn-resume-model47-capture-20261026-v2'
        control=run.with_name(run.name+'-control');control.mkdir(mode=0o700)
        inputs=root/'inputs';inputs.mkdir();refs={};old={}
        for role,(task,oldrun,states)in R.ROLES.items():
            canonical=control/(role+'-terminal.json');raw=('old-'+role+'\n').encode();canonical.write_bytes(raw);canonical.chmod(0o600);old[role]=raw
            state=next(iter(states));record={'schema':'ptn-predecessor-terminal/v1','task_id':task,'run_root':oldrun,
                'state':state,'active_workers':0,'worker_states':[{'name':'worker_0','state':state}],
                'observed_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
            path=inputs/(role+'.json');path.write_bytes(R.C.canonical(record));refs[role]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        return base,dict(run=str(run),control=str(control),audit=str(root/'audit'),refs=refs),old

    def run_case(self,args,base,execute=False):
        with patch.object(R.C,'BASE',str(base)):return R.refresh(**args,execute=execute)

    def test_dryrun_archives_original_bytes_without_canonical_mutation_or_admission(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp).resolve();base,args,old=self.setup_case(root);before={r:R.current(Path(args['control'])/(r+'-terminal.json'))for r in R.ROLES}
            result=self.run_case(args,base);self.assertTrue(result['passed']);self.assertTrue(result['dry_run']);self.assertFalse(result['runtime_admission'])
            for role in R.ROLES:
                self.assertEqual((Path(args['audit'])/(role+'-before.json')).read_bytes(),old[role])
                self.assertEqual(R.current(Path(args['control'])/(role+'-terminal.json')),before[role])

    def test_execute_preserves_exact_input_observation_and_oldbyte_archive(self):
        with tempfile.TemporaryDirectory()as temp:
            base,args,old=self.setup_case(Path(temp).resolve());result=self.run_case(args,base,True)
            self.assertTrue(result['passed']);self.assertFalse(result['atomic_two_file_transaction']);self.assertFalse(result['observed_utc_modified'])
            for role in R.ROLES:
                raw=Path(args['refs'][role]['path']).read_bytes();self.assertEqual((Path(args['control'])/(role+'-terminal.json')).read_bytes(),raw)
                self.assertEqual((Path(args['audit'])/(role+'-before.json')).read_bytes(),old[role])
            self.assertEqual(result['replaced_roles'],['original','predecessor'])

    def test_wrong_task_stale_future_and_wrong_sha_refused_without_updates(self):
        for mode in ('task','stale','future','sha'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as temp:
                base,args,old=self.setup_case(Path(temp).resolve());ref=args['refs']['original'];p=Path(ref['path']);v=json.loads(p.read_bytes())
                if mode=='task':v['task_id']='foreign'
                if mode in ('stale','future'):v['observed_utc']=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=-2 if mode=='stale'else 2)).isoformat()
                p.write_bytes(R.C.canonical(v));ref['sha256']='0'*64 if mode=='sha'else hashlib.sha256(p.read_bytes()).hexdigest()
                result=self.run_case(args,base,True);self.assertFalse(result['passed']);self.assertEqual(result['actual_changed_roles'],[])
                for role in R.ROLES:self.assertEqual((Path(args['control'])/(role+'-terminal.json')).read_bytes(),old[role])

    def test_claimed_run_symlink_unsafe_control_frozen_audit_and_input_overlap_refused(self):
        for mode in ('claim','symlink','unsafe','frozen','overlap'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as temp:
                root=Path(temp).resolve();base,args,old=self.setup_case(root)
                if mode=='claim':Path(args['run']).mkdir()
                if mode=='symlink':
                    link=base/'link';link.symlink_to(args['control'],target_is_directory=True);args['control']=str(link)
                if mode=='unsafe':Path(args['control']).chmod(0o777)
                if mode=='frozen':
                    package=base/'ptn-resume-model47-capture-package-test';package.mkdir();args['audit']=str(package/'new-audit')
                if mode=='overlap':args['refs']['original']['path']=str(Path(args['control'])/'original-terminal.json')
                with self.assertRaises(ValueError):self.run_case(args,base,True)
                self.assertFalse(Path(args['audit']).exists())

    def test_run_appears_immediately_before_replacement_refuses_mutation(self):
        with tempfile.TemporaryDirectory()as temp:
            base,args,old=self.setup_case(Path(temp).resolve());original=R.C.terminal;calls=[0]
            def audit(*a,**kw):
                result=original(*a,**kw);calls[0]+=1
                if calls[0]==2:Path(args['run']).mkdir()
                return result
            with patch.object(R.C,'terminal',side_effect=audit):result=self.run_case(args,base,True)
            self.assertFalse(result['passed']);self.assertEqual(result['actual_changed_roles'],[])

    def test_partial_replace_failure_keeps_old_and_actual_after_evidence(self):
        for after_replace in (False,True):
            with self.subTest(after_replace=after_replace),tempfile.TemporaryDirectory()as temp:
                base,args,old=self.setup_case(Path(temp).resolve());replace=R.replace_one;calls=[0]
                def fail(path,raw,**kwargs):
                    calls[0]+=1
                    if after_replace:
                        replace(path,raw,**kwargs);raise OSError('synthetic post-publication error')
                    if calls[0]==2:raise OSError('synthetic second-file error')
                    return replace(path,raw,**kwargs)
                with patch.object(R,'replace_one',side_effect=fail):result=self.run_case(args,base,True)
                self.assertFalse(result['passed']);self.assertTrue(result['partial_update_possible']);self.assertEqual(result['actual_changed_roles'],['original'])
                self.assertEqual((Path(args['control'])/'predecessor-terminal.json').read_bytes(),old['predecessor'])
                for role in R.ROLES:self.assertEqual((Path(args['audit'])/(role+'-before.json')).read_bytes(),old[role])


if __name__=='__main__':unittest.main()
