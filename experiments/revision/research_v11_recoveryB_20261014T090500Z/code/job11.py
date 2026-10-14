"""Publish a stage receipt around a frozen existing entry point."""
import subprocess
import sys
import time
import resource
import common11 as c

name=sys.argv[1]
job=c.read(c.ROOT/'manifest.json')['jobs'][name]
started=time.monotonic()
before=resource.getrusage(resource.RUSAGE_CHILDREN)
result=subprocess.run(job['command'],cwd=c.REPO,stdin=subprocess.DEVNULL)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
outputs=[]
if result.returncode==0:
    for item in job['outputs']:
        if item['path'].endswith('/stage_results/'+name+'.json'):continue
        outputs.append(dict(path=item['path'],sha256=c.sha(item['path'])))
outcome='completed' if result.returncode==0 else 'technical_failed'
if result.returncode==0 and job.get('deferred_action'):
    assert c.read(job['outputs'][0]['path'])['status']=='DEFERRED'
    outcome='DEFERRED'
c.save(c.ROOT/'stage_results'/(name+'.json'),dict(outcome=outcome,job=name,returncode=result.returncode,seconds=time.monotonic()-started,process_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,outputs=outputs))
sys.exit(result.returncode)
