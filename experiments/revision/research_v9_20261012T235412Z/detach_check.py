"""Two CPU sentinels verify that a child outlives its launcher."""
import sys,time,os,subprocess,json
from pathlib import Path
import common_v9 as c
if sys.argv[1]=='launch':
    p=subprocess.Popen([c.PYTHON,str(Path(__file__)),'child'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    print(p.pid)
else:
    time.sleep(.4)
    for i in range(2):
        c.save(c.ROOT/f'detach_sentinel{i}.json',dict(pid=os.getpid(),parent_pid=os.getppid(),sentinel=i,at=c.q.now()));time.sleep(.1)
