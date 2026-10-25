"""One bounded external terminal receipt; never initialize CUDA or launch a child."""
import datetime,hashlib,json,os,stat,time
from pathlib import Path
import strict_restore as S
RECEIPT=Path('/root/experiments/redimnet2/b6-mix-official-20261019/ptn-resume-control-20261024/predecessor-wb58x-terminal.json')
TASK='t-20261024151842-wb58x'
RUN='/root/experiments/redimnet2/b6-mix-official-20261019/ptn-resume-model35-20261024-v2'
class Publishing(Exception):pass
def snap(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink]
def read_terminal(path=RECEIPT):
 path=Path(path)
 try:d=path.parent.lstat();a=path.lstat()
 except FileNotFoundError:raise Publishing('receipt not published')
 S.need(stat.S_ISDIR(d.st_mode) and d.st_uid==0 and not d.st_mode&0o022,'unsafe terminal receipt directory')
 S.need(stat.S_ISREG(a.st_mode) and a.st_uid==0 and not a.st_mode&0o022 and 1<=a.st_size<=65536,'unsafe terminal receipt regular file')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  b=os.fstat(fd);data=os.read(fd,65537);c=os.fstat(fd)
 finally:os.close(fd)
 try:end=path.lstat()
 except FileNotFoundError:raise Publishing('receipt changed during atomic publication')
 if snap(a)!=snap(b) or snap(b)!=snap(c) or snap(c)!=snap(end) or end.st_nlink!=1:raise Publishing('receipt publication not yet stable')
 value=json.loads(data);S.need(value['schema']=='ptn-predecessor-terminal/v1' and value['task_id']==TASK and value['run_root']==RUN,'wrong terminal predecessor identity')
 S.need(value['state']=='Terminated' and value['active_workers']==0 and len(value['worker_states'])==1 and value['worker_states'][0]['name']=='worker_0' and value['worker_states'][0]['state']=='Terminated','predecessor not actually terminal')
 observed=datetime.datetime.fromisoformat(value['observed_utc'].replace('Z','+00:00'));S.need(observed.tzinfo is not None,'terminal UTC timezone missing')
 age=(datetime.datetime.now(datetime.timezone.utc)-observed).total_seconds();S.need(-300<=age<=3600,'terminal receipt stale/future')
 return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'stable_stat':snap(end),'payload':value,'cuda_initialized_by_gate':False}
def wait_for_terminal(timeout=1800.):
 start=time.monotonic();last=None
 while True:
  try:return read_terminal()
  except Publishing as exc:
   reason=str(exc)
   if reason!=last:print('WAITING_PREDECESSOR_TERMINAL '+json.dumps({'task_id':TASK,'path':str(RECEIPT),'reason':reason,'maximum_wait_seconds':timeout}),flush=True);last=reason
   S.need(time.monotonic()-start<timeout,'predecessor terminal wait exhausted; no GPU or retry was launched');time.sleep(min(1.,max(.001,timeout-(time.monotonic()-start))))
def verify_unchanged(expected):
 start=time.monotonic()
 while True:
  try:actual=read_terminal();break
  except Publishing:S.need(time.monotonic()-start<2.,'terminal receipt unstable under admission locks');time.sleep(.05)
 S.need(actual['sha256']==expected['sha256'] and actual['stable_stat']==expected['stable_stat'],'terminal receipt changed after wait; refuse new admission')
 return actual
