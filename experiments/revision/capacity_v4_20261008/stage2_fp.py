"""Opt-in crossed graph declarations; reuse the unchanged FP entry point."""
import argparse
import sys
p=argparse.ArgumentParser()
for name in ('capsule','spec','out','identity-manifest','graph-declarations'):p.add_argument('--'+name,required=True)
a=p.parse_args();sys.path.insert(0,a.capsule)
from models import compression
compression.COMPRESSION_LADDER.update(X1=(.5,.5),X2=(.25,1.))
import scripts.compression_jobs as cj
from experiments.revision_queue import read_json,file_hash
for item in read_json(a.identity_manifest)['files']:
    assert file_hash(item['path'])==item['sha256'],item['path']
spec=read_json(a.spec)
declared=read_json(a.graph_declarations)['graphs'][spec['graph']]
for key,value in compression.compression_graph_spec(spec['graph']).items():
    assert declared[key]==value,key
cj.warmup(spec,a.out)
