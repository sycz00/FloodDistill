import json,os,sys,time
from pathlib import Path
time.sleep(2)
Path(sys.argv[1]).write_text(json.dumps({"status":"complete","pid":os.getpid(),"ppid":os.getppid(),"sid":os.getsid(0),"stdin_null":os.readlink("/proc/self/fd/0")=="/dev/null"}))
