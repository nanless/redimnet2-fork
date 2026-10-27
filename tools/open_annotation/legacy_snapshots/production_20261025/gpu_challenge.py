"""Active CUDA-context attestation; explicitly NOT a host/container PID translation.

No torch import here. Parent consumes real fresh NVML samples; child uses native
cudaMalloc/free before loading any ASR/FA model. Every phase is serialized by ack.
"""
import ctypes,json,os,re,secrets,time,math
from pathlib import Path
from gpu_common import need,write,sha
SCHEMA='cuda-context-active-attestation/v1'
MIB=1024**2

def atomic(path,obj):
 path=Path(path);tmp=path.with_name(path.name+'.writing');write(tmp,obj);os.rename(tmp,path)

def minor_map(uuids):
 result={}
 for info in Path('/proc/driver/nvidia/gpus').glob('*/information'):
  text=info.read_text();u=re.search(r'^GPU UUID:\s*(GPU-[0-9a-f-]+)',text,re.M|re.I);m=re.search(r'^Device Minor:\s*(\d+)',text,re.M)
  if u and m and u[1] in uuids:result[u[1]]=int(m[1])
 need(set(result)==set(uuids),'kernel UUID/minor information incomplete')
 return result

def normalized_uuid(raw):
 found=re.findall(r'(?:GPU-)?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',str(raw),re.I)
 need(len(found)==1,'actual CUDA UUID unavailable/unparseable')
 return 'GPU-'+found[0].lower()

class ParentChallenge:
 def __init__(self,root,nonce,uuids,known_worker_pid):
  self.root=Path(root);self.root.mkdir();self.nonce=nonce;self.uuids=list(uuids);need(len(self.uuids)==1,'production bootstrap requires exactly one CUDA UUID');self.known_worker_pid=known_worker_pid;self.seq=0;self.opaque_pid=None;self.baseline=None;self.samples=[];self.complete=False;self.last_phase_sample=None;self.phase_stable=0;self.phase_started=time.monotonic();self.records=[];self.last_sample_stamp=None;self.phase_samples=[]
  self.todo=[('baseline',None,0)]
  for j in range(3):
   order=list(range(len(self.uuids)));secrets.SystemRandom().shuffle(order)
   for gpu in order:self.todo.extend([('allocate',gpu,None),('free',gpu,0)])
  self._issue()
 def _issue(self):
  action,gpu,size=self.todo[self.seq]
  if action=='allocate':size=(4+secrets.randbelow(21))*16*MIB
  self.current={'schema':SCHEMA,'seq':self.seq,'nonce':self.nonce,'action':action,'gpu':gpu,'bytes':size};atomic(self.root/f'request_{self.seq}.json',self.current)
  self.phase_started=time.monotonic();self.last_phase_sample=None;self.phase_stable=0;self.phase_samples=[]
 def rows(self,snapshot,initial=False):
  by={u:[p for p in snapshot['compute_processes'] if p['gpu_uuid']==u] for u in self.uuids}
  need(all(len(x)<=1 for x in by.values()),'multiple compute rows on target UUID; refuse unknown process')
  if initial and any(len(x)==0 for x in by.values()):return None
  need(all(len(x)==1 for x in by.values()),'attested CUDA context disappeared')
  pids={x[0]['pid'] for x in by.values()};need(len(pids)==1,'one worker must have one opaque NVML PID across its visible UUIDs')
  pid=next(iter(pids))
  if self.opaque_pid is not None:need(pid==self.opaque_pid,'opaque NVML row identity changed')
  for x in by.values():need(type(x[0]['used_mib'])is int and x[0]['used_mib']>=0,'unknown NVML process memory prevents attestation')
  return {u:by[u][0]['used_mib'] for u in self.uuids},pid
 def advance(self,snapshot):
  need(not self.complete,'challenge already complete')
  limit=120 if self.seq==0 and not getattr(self,'baseline_response_seen',False) else 20
  need(time.monotonic()-self.phase_started<limit,'worker initialization or CUDA allocation phase timed out')
  stamp=snapshot['monotonic'];need(type(stamp)in(int,float) and math.isfinite(stamp) and (self.last_sample_stamp is None or stamp>self.last_sample_stamp),'snapshot must be strictly fresh/increasing');self.last_sample_stamp=stamp
  observed=self.rows(snapshot,initial=self.baseline is None)
  reply=self.root/f'response_{self.seq}.json'
  if not reply.is_file():return
  event=json.loads(reply.read_text());need(event['schema']==SCHEMA and event['seq']==self.seq and event['nonce']==self.nonce and event['worker_pid']==self.known_worker_pid,'challenge response identity differs')
  need(event['request']==self.current,'challenge allocation request differs')
  need(event['actual_cuda_uuids']==self.uuids,'actual CUDA UUID evidence differs')
  need(event['minor_fds_verified'] is True,'worker physical minor FD proof missing')
  eventtime=event['observed_monotonic'];need(type(eventtime)in(int,float) and math.isfinite(eventtime) and 0<=time.monotonic()-stamp<5 and 0<=time.monotonic()-eventtime<20,'phase sample/event stale or nonfinite')
  if self.seq==0 and not getattr(self,'baseline_response_seen',False):self.phase_started=eventtime;self.baseline_response_seen=True
  if stamp<eventtime or observed is None:return
  mem,pid=observed
  if self.baseline is None:
   # Two stable fresh observations after warmup establish only the baseline;
   # no identity is accepted until all 12 random allocation/free phases match.
   target=mem
  else:
   target=dict(self.baseline)
   if self.current['action']=='allocate':target[self.uuids[self.current['gpu']]]+=self.current['bytes']//MIB
   if any(abs(mem[u]-target[u])>1 for u in self.uuids):
    self.last_phase_sample=None;self.phase_stable=0;self.phase_samples=[];return
  current=(pid,tuple(mem[u] for u in self.uuids))
  if current==self.last_phase_sample:self.phase_stable+=1
  else:self.last_phase_sample=current;self.phase_stable=1;self.phase_samples=[]
  self.phase_samples.append(snapshot)
  if self.phase_stable<2:return
  if self.baseline is None:self.baseline=dict(mem);self.opaque_pid=pid
  record={'seq':self.seq,'request':self.current,'response':event,'NVML_memory_mib':mem,'expected_memory_mib':target,'opaque_nvml_pid':pid,'accepted_fresh_snapshots':list(self.phase_samples),'stable_samples':self.phase_stable}
  self.records.append(record);atomic(self.root/f'accepted_{self.seq}.json',record);atomic(self.root/f'ack_{self.seq}.json',{'seq':self.seq,'nonce':self.nonce})
  self.seq+=1
  if self.seq==len(self.todo):
   self.complete=True
   atomic(self.root/'attestation.json',{'schema':SCHEMA,'complete':True,'known_container_worker_pid':self.known_worker_pid,'opaque_nvml_pid':self.opaque_pid,'UUIDs':self.uuids,'random_allocation_rounds_per_GPU':3,'all_free_phases_returned_to_baseline':True,'PID_translation_claimed':False,'meaning':'controlled native CUDA memory pattern uniquely associated with a stable NVML row; not kernel PID mapping','phases':self.records})
   atomic(self.root/f'request_{self.seq}.json',{'schema':SCHEMA,'seq':self.seq,'nonce':self.nonce,'action':'complete','gpu':None,'bytes':0})
  else:self._issue()
 def monitor(self,snapshot):return self.rows(snapshot,initial=False)

def worker_challenge(root,nonce,uuids,torch,minor_info):
 root=Path(root);libpaths={line.split(None,5)[5] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcudart.so' in line and len(line.split(None,5))==6}
 need(len(libpaths)==1,'one mapped libcudart required')
 path=next(iter(libpaths));need(not path.endswith('(deleted)') and Path(path).is_file(),'mapped libcudart unavailable')
 lib=ctypes.CDLL(path)
 lib.cudaSetDevice.argtypes=[ctypes.c_int];lib.cudaSetDevice.restype=ctypes.c_int
 lib.cudaMalloc.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t];lib.cudaMalloc.restype=ctypes.c_int
 lib.cudaFree.argtypes=[ctypes.c_void_p];lib.cudaFree.restype=ctypes.c_int
 lib.cudaMemset.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t];lib.cudaMemset.restype=ctypes.c_int
 lib.cudaDeviceSynchronize.argtypes=[];lib.cudaDeviceSynchronize.restype=ctypes.c_int
 def call(name,*a):need(getattr(lib,name)(*a)==0,'native '+name+' failed')
 actual=[normalized_uuid(torch.cuda.get_device_properties(i).uuid) for i in range(len(uuids))];need(actual==uuids,'CUDA UUID does not match allocation')
 # Warm up native runtime and memcpy/memset paths before baseline.
 for i in range(len(uuids)):
  call('cudaSetDevice',i);ptr=ctypes.c_void_p();call('cudaMalloc',ctypes.byref(ptr),MIB);call('cudaMemset',ptr,7,MIB);call('cudaDeviceSynchronize');call('cudaFree',ptr);call('cudaDeviceSynchronize')
 fds=[]
 for fd in Path('/proc/self/fd').iterdir():
  try:target=str(fd.readlink())
  except FileNotFoundError:continue
  if re.fullmatch(r'/dev/nvidia\d+',target):fds.append(target)
 need(all('/dev/nvidia'+str(minor_info[u]) in fds for u in uuids),'physical CUDA minor FD evidence incomplete')
 allocated={};seq=0
 try:
  while True:
   request_path=root/f'request_{seq}.json';start=time.monotonic()
   while not request_path.exists():need(time.monotonic()-start<20,'parent request missing');time.sleep(.05)
   req=json.loads(request_path.read_text());need(req['schema']==SCHEMA and req['nonce']==nonce and req['seq']==seq,'parent request identity differs')
   if req['action']=='complete':need(seq==1+6*len(uuids) and not allocated,'challenge not fully released');return
   if req['action']=='allocate':
    i=req['gpu'];size=req['bytes'];need(i in range(len(uuids)) and type(size)is int and 64*MIB<=size<=384*MIB and size%(16*MIB)==0 and i not in allocated,'invalid native allocation request')
    call('cudaSetDevice',i);ptr=ctypes.c_void_p();call('cudaMalloc',ctypes.byref(ptr),size);allocated[i]=ptr;call('cudaMemset',ptr,seq%251,size);call('cudaDeviceSynchronize')
   elif req['action']=='free':
    i=req['gpu'];need(i in allocated,'free without owned allocation');call('cudaSetDevice',i);call('cudaFree',allocated.pop(i));call('cudaDeviceSynchronize')
   else:need(req['action']=='baseline' and seq==0,'unexpected phase action')
   event={'schema':SCHEMA,'seq':seq,'nonce':nonce,'worker_pid':os.getpid(),'request':req,'actual_cuda_uuids':actual,'physical_minor_fds':fds,'minor_fds_verified':True,'libcudart_path':path,'libcudart_sha256':sha(path),'observed_monotonic':time.monotonic()}
   atomic(root/f'response_{seq}.json',event);ack=root/f'ack_{seq}.json';start=time.monotonic()
   while not ack.exists():need(time.monotonic()-start<20,'parent NVML attestation missing');time.sleep(.05)
   value=json.loads(ack.read_text());need(value['seq']==seq and value['nonce']==nonce,'parent phase ack differs');seq+=1
 finally:
  for i,ptr in list(allocated.items()):call('cudaSetDevice',i);call('cudaFree',ptr);call('cudaDeviceSynchronize')
