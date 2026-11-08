"""从已完整核验的联合消费索引按来源唯一读取；不按mtime推断版本。"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import sqlite3
import sys
from urllib.parse import quote

sys.path.insert(0,str(Path(__file__).resolve().parent))
from union_coverage import need,physical,ref,read_json_lines,child,sha_file,read_result,digest


class UnionConsumer:
    def __init__(self,spec_path,spec_sha,database_path,database_sha):
        self.spec=ref({'path':str(spec_path),'sha256':spec_sha});self.path=physical(database_path)
        need(sha_file(self.path)==database_sha,'consumer database SHA differs')
        self.initial=self.path.stat();self.db=sqlite3.connect('file:'+quote(str(self.path),safe='/')+'?mode=ro',uri=True)
        try:
            self.db.execute('PRAGMA query_only=ON');self.db.execute('PRAGMA temp_store=FILE')
            self.db.execute('PRAGMA cache_size=-8192');self.db.execute('BEGIN')
            meta={k:json.loads(v) for k,v in self.db.execute('SELECT key,value FROM metadata')}
            need(meta['spec_sha256']==spec_sha and meta['spec']==self.spec and
                 meta['validation']['mode']=='complete' and meta['validation']['exact_union_and_disjoint'] is True,
                 'consumer requires complete index bound to exact spec')
            need(meta['validation']['source_count']==self.spec['expected_full_source_count'], 'consumer full denominator differs')
            self.index=ref(self.spec['full_index']);self.cache=collections.OrderedDict()
            self.producers={p['id']:p for p in (self.spec['old_producer'],self.spec['new_producer'])}
            self.contracts={key:ref(value['contract']) for key,value in self.producers.items()}
        except BaseException:self.db.close();raise

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.db.close()

    def _source(self,ordinal,row_number,source_sha,sid):
        descriptor=self.index['shards'][ordinal];p=child(Path(self.spec['full_index']['path']).parent,descriptor['path'])
        current=p.stat()
        stamp=(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)
        cached=self.cache.get(ordinal)
        if cached is None or cached[0]!=stamp:
            rows,raw=read_json_lines(p)
            import hashlib
            need(hashlib.sha256(raw).hexdigest()==descriptor['sha256'],'original source shard changed during consumption')
            self.cache[ordinal]=(stamp,rows);self.cache.move_to_end(ordinal)
            while len(self.cache)>4:self.cache.popitem(last=False)
        rows=self.cache[ordinal][1];need(0<=row_number<len(rows),'source ordinal outside original shard')
        source=rows[row_number];need(source['source_id']==sid and digest(source)==source_sha,'source selection digest differs')
        return source

    def resolve_source(self,source_id):
        current=self.path.stat()
        need((current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)==
             (self.initial.st_dev,self.initial.st_ino,self.initial.st_size,self.initial.st_mtime_ns),'consumer index changed after opening')
        row=self.db.execute('SELECT source_sha,shard,row_number,relative,selected,record_sha,commit_sha FROM sources WHERE sid=?',(source_id,)).fetchone()
        need(row is not None,'source absent from union')
        source=self._source(row[1],row[2],row[0],source_id)
        need(source['sidecar_relpath']==row[3] and row[4] in self.producers and row[5] and row[6], 'selected producer/result missing')
        record,rs,cs=read_result(source,self.producers[row[4]],row[5],row[6],contract=self.contracts[row[4]])
        return {'source_id':source_id,'producer_id':row[4],'source':source,'record':record,
                'sidecar_sha256':rs,'commit_sha256':cs,'training_or_evaluation_eligibility_released':False}

    def iter_selected_sources(self):
        for (sid,) in self.db.execute('SELECT sid FROM sources ORDER BY shard,row_number'):
            yield self.resolve_source(sid)


def resolve_source(spec_path,spec_sha,source_id,*,database_path,database_sha):
    with UnionConsumer(spec_path,spec_sha,database_path,database_sha) as consumer:return consumer.resolve_source(source_id)


def iter_selected_sources(spec_path,spec_sha,*,database_path,database_sha):
    with UnionConsumer(spec_path,spec_sha,database_path,database_sha) as consumer:
        yield from consumer.iter_selected_sources()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','spec-sha','database','database-sha','source-id'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=resolve_source(a.spec,a.spec_sha,a.source_id,database_path=a.database,database_sha=a.database_sha)
    print(json.dumps(r,ensure_ascii=False))


if __name__=='__main__':main()
