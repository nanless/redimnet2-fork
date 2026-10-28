"""按已保存的小样本清单只读取原音；不遍历数据目录、不写远端、不调用模型。

默认最多24个独立来源、总编码数据256MiB。远端每个文件核打开前后身份和
已记录编码SHA，内存有上限；本地只展开已知qXX文件，保留实际采集收据。
"""
from __future__ import annotations
import argparse, hashlib, io, json, pathlib, subprocess, tarfile

def need(value, message):
    if not value: raise ValueError(message)

MAX_BYTES=256*1024**2
REMOTE=r'''
import io,json,hashlib,os,stat,sys,tarfile
rows=INPUT_ROWS
maximum=MAXIMUM
blobs=[]; receipts=[]; total=0
stamp=lambda s: [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
for row in rows:
    path=row['audio_path']; initial=os.stat(path)
    if not stat.S_ISREG(initial.st_mode) or initial.st_size>maximum-total: raise ValueError('bounded regular source required')
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        if stamp(initial)!=stamp(before): raise ValueError('source changed before open')
        with os.fdopen(fd,'rb',closefd=False) as f: raw=f.read(maximum-total+1)
        after=os.fstat(fd)
        if len(raw)!=before.st_size or stamp(before)!=stamp(after) or stamp(after)!=stamp(os.stat(path)): raise ValueError('source changed during read')
    finally: os.close(fd)
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=row['expected_sha256']: raise ValueError('source encoded SHA mismatch')
    total+=len(raw)
    if total>maximum: raise ValueError('aggregate audio budget exceeded')
    blobs.append((row['archive_name'],raw))
    receipts.append(dict(row,actual_sha256=digest,bytes=len(raw),stat_before=stamp(before),stat_after=stamp(after)))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
    for name,raw in blobs+ [('collection-receipt.json',(json.dumps(dict(schema='bounded-original-audio-collection/v1',sources=receipts,total_bytes=total,remote_writes=0,GPU_calls=0),ensure_ascii=False,indent=2)+'\n').encode())]:
        info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o600;tar.addfile(info,io.BytesIO(raw))
'''

def collect(selection: pathlib.Path, output: pathlib.Path, host: str):
    items=json.loads(selection.read_text()); need(isinstance(items,list) and 1<=len(items)<=24,"select 1..24 sources")
    need(len({x['row']['source_id'] for x in items})==len(items),'duplicate source')
    rows=[]
    for i,x in enumerate(items,1):
        need(x['review_id']==f'q{i:02d}','ordered qXX review identity required')
        p=pathlib.PurePosixPath(x['row']['audio_path'])
        need(p.is_absolute() and p.suffix.lower() in ('.wav','.flac'),'absolute WAV/FLAC source required')
        rows.append(dict(source_id=x['row']['source_id'],audio_path=str(p),expected_sha256=x['row']['audio_recorded_file_sha256'],archive_name=f'q{i:02d}/source{p.suffix.lower()}'))
    need(output.is_absolute() and not output.exists(),'new absolute output required')
    output.mkdir(mode=0o700)
    archive=output/'originals.tar'
    code=REMOTE.replace('INPUT_ROWS',repr(rows)).replace('MAXIMUM',str(MAX_BYTES))
    (output/'remote-reader.py').write_text(code)
    cmd=['ssh','-o','ControlMaster=no','-o','ControlPath=none','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',host,'nice -n 15 python3 -I -S -B -']
    with archive.open('xb') as f:
        r=subprocess.run(cmd,input=code.encode(),stdout=f,stderr=subprocess.PIPE,timeout=180)
    (output/'ssh-stderr.txt').write_bytes(r.stderr)
    if r.returncode: raise RuntimeError(f'readonly SSH failed ({r.returncode}); evidence preserved')
    expected={r['archive_name'] for r in rows}|{'collection-receipt.json'}
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        need(len(members)==len(expected) and {m.name for m in members}==expected,'archive member set differs')
        need(sum(m.size for m in members if m.name!='collection-receipt.json')<=MAX_BYTES,'local audio aggregate budget exceeded')
        for m in members:
            need(m.isfile() and m.size<=MAX_BYTES,'bounded regular archive member required')
            p=output/m.name;p.parent.mkdir(exist_ok=True)
            raw=tar.extractfile(m).read()
            p.write_bytes(raw)
    receipt=json.loads((output/'collection-receipt.json').read_text())
    for row in rows:
        need(hashlib.sha256((output/row['archive_name']).read_bytes()).hexdigest()==row['expected_sha256'],'local source SHA differs')
    print(json.dumps(dict(sources=len(rows),total_bytes=receipt['total_bytes'],remote_writes=0,GPU_calls=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--host',default='dev_a800_2gpus');a=p.parse_args();collect(a.selection,a.output,a.host)
