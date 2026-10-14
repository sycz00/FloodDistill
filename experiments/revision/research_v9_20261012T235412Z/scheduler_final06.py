"""Cost-gated final readout; keeps the admitted training guardians and clocks."""
from pathlib import Path
from policy_final03 import activate
activate()
import common_v9 as c
import scheduler_v9 as scheduler
old_read=c.read
def read(path,default=None):
    if Path(path)==c.ROOT/'manifest.json':
        result=old_read(c.ROOT/'manifest_final06.json',default)
        return result
    if Path(path)==c.ROOT/'sealed_cost_admission04.json':return old_read(path,default)
    return old_read(path,default)
c.read=read
scheduler.__file__=__file__
original_availability=scheduler.availability
def availability(row,job,history,now):
    if job.get('full_matrix_cost_gate'):
        gate=c.read(c.ROOT/'sealed_cost_admission04.json')
        if not gate or not gate['admitted']:return False,'full final predictor matrix exceeds conservative cost ceiling',False
    return original_availability(row,job,history,now)
scheduler.availability=availability
original_status=scheduler.status
def status(state,manifest):
    # Turn a deterministic cost refusal into a visible deferral, not eight hours of GPU waiting.
    gate=c.read(c.ROOT/'sealed_cost_admission04.json')
    if gate and not gate['admitted'] and state['jobs'].get('sealed_teacher_B0_final03') not in scheduler.TERMINAL:
        state['jobs']['sealed_teacher_B0_final03']='DEFERRED_BUDGET'
    return original_status(state,manifest)
scheduler.status=status
original_run=scheduler.subprocess.run
def run(command,*args,**kwargs):
    if isinstance(command,list) and len(command)>1 and command[1]==str(c.ROOT/'collect_v9.py'):
        command=list(command);command[1]=str(c.ROOT/'collect_v9_final06.py')
    return original_run(command,*args,**kwargs)
scheduler.subprocess.run=run
if __name__=='__main__':scheduler.main()
