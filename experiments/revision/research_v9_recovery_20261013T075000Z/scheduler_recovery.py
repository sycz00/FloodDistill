"""Repair queue retains the original clocks and shared/incremental GPU caps."""
from pathlib import Path
from policy_final03 import activate
activate()
import common_v9 as c
import scheduler_v9 as scheduler
scheduler.__file__=__file__
original_reserve=scheduler.reserve_block
def reserve(ledger,run,block,seconds,protected,cap=108000):
    parent=c.read(c.ROOT/'manifest.json')['parent_run']
    familycharges=sum(v['seconds'] for k,v in ledger.data['charges'].items() if k.startswith(parent+'/') or k.startswith(run+'/'))
    ownreserve=sum(v['remaining_seconds'] for k,v in ledger.data['reservations'].items() if k.startswith(run+':'))
    key=run+':'+block
    add=max(0,seconds-ledger.data['reservations'].get(key,{}).get('remaining_seconds',0))
    guard=0 if run+':final' in ledger.data['reservations'] else protected
    if familycharges+ownreserve+add+guard>14400:return False
    return original_reserve(ledger,run,block,seconds,protected,cap)
scheduler.reserve_block=reserve
original_status=scheduler.status
def status(state,manifest):
    gate=c.read(c.ROOT/'sealed_cost_admission04.json')
    if gate and not gate['admitted'] and state['jobs'].get('sealed_readout_retry01') not in scheduler.TERMINAL:
        state['jobs']['sealed_readout_retry01']='DEFERRED_BUDGET'
    return original_status(state,manifest)
scheduler.status=status
original_availability=scheduler.availability
def availability(row,job,history,now):
    if job.get('full_matrix_cost_gate'):
        gate=c.read(c.ROOT/'sealed_cost_admission04.json')
        if not gate or not gate['admitted']:return False,'full sealed matrix cost gate refused',False
    return original_availability(row,job,history,now)
scheduler.availability=availability
original_run=scheduler.subprocess.run
def run(command,*args,**kwargs):
    if isinstance(command,list) and len(command)>1 and command[1]==str(c.ROOT/'collect_v9.py'):
        command=list(command);command[1]=str(c.ROOT/'collect_recovery.py')
    return original_run(command,*args,**kwargs)
scheduler.subprocess.run=run
if __name__=='__main__':scheduler.main()
