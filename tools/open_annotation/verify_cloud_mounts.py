"""CPU-only exact persistent mount guard; never creates run directories."""
from pathlib import Path
import argparse,json,os,subprocess
from cloud_gate import Reader,need,sha
def validate_row(expected,value):
    rows=value.get('filesystems',[]);need(len(rows)==1,'ambiguous findmnt result')
    row=rows[0];opts=set(row['options'].split(','));mode=expected['mode']
    need(row['target']==expected['target'] and row['source']==expected['source'] and row['fstype']==expected['fstype']=='gpfs' and mode in opts and ({'ro','rw'}-{mode}).isdisjoint(opts),'persistent mount source/target/type/mode differs: '+json.dumps(row))
    return row
def validate_cloud(cloud):
    base=Path(cloud['task_root']);run=Path(cloud['run_root'])
    need(base.is_absolute() and base not in {Path('/'),Path('/root')},'independent explicit task root required')
    need(base in run.parents and not any(p in {'..','.'} for p in run.parts),'run escapes independent Ultra root')
    need(not run.exists() and not run.is_symlink(),'run already exists; inspect/query duplicates, do not overwrite')
    profiles={m['target']:m for m in cloud['mounts']}
    need(set(profiles)=={'/root','/shared/group-shared'} and len(cloud['mounts'])==2,'exact two persistent mount profiles required')
    need(profiles['/root']['mode']=='rw' and profiles['/shared/group-shared']['mode']=='ro','user RW/group RO required')
    proof={}
    def query(target,expected):
        result=subprocess.run(['findmnt','-J','-T',target,'-o','TARGET,SOURCE,FSTYPE,OPTIONS'],capture_output=True,text=True,timeout=5,check=True)
        need(not result.stderr.strip(),'findmnt stderr: '+result.stderr)
        return validate_row(expected,json.loads(result.stdout))
    for target,expected in profiles.items():proof[target]=query(target,expected)
    need(base.exists() and not base.is_symlink() and base.resolve()==base,'task root not an existing physical persistent directory')
    ancestor=run.parent
    while not ancestor.exists():ancestor=ancestor.parent
    need(base==ancestor or base in ancestor.parents,'run ancestor escapes task root')
    need(ancestor.resolve()==ancestor,'run ancestor has symlink traversal')
    proof[str(ancestor)]=query(str(ancestor),profiles['/root'])
    stats=os.statvfs(base)
    return {'schema':'ultra-cloud-persistent-mounts/v1','passed':True,'mounts':proof,'run_root':str(run),'available_bytes':stats.f_bavail*stats.f_frsize,'available_inodes':stats.f_favail,'quota_known':False,'run_created':False,'GPU_or_model_execution':False}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha',required=True);a=p.parse_args();raw=a.spec.read_bytes();need(sha(raw)==a.spec_sha,'spec SHA differs');spec=json.loads(raw);print(json.dumps(validate_cloud(spec['cloud']),indent=2))
if __name__=='__main__':main()
