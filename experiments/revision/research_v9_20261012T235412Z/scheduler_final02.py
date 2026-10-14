"""Frozen additive recovery: explicit evaluation numerics; unchanged training."""
from policy_final02 import activate
activate()
import common_v9 as c
import scheduler_v9 as scheduler
scheduler.__file__=__file__
original_run=scheduler.subprocess.run
def subprocess_run(command,*args,**kwargs):
    if isinstance(command,list) and len(command)>1 and command[1]==str(c.ROOT/'collect_v9.py'):
        command=list(command);command[1]=str(c.ROOT/'collect_v9_final02.py')
    return original_run(command,*args,**kwargs)
scheduler.subprocess.run=subprocess_run
if __name__=='__main__':scheduler.main()
