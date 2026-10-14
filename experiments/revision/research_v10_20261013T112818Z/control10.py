"""Fresh-terminal controls around the frozen CPU worker; restart keeps its original clocks."""
import sys,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import worker10 as w

def main():
 action=sys.argv[1] if len(sys.argv)>1 else 'status'
 assert action in ['status','pause','resume','stop']
 subprocess.run([w.c.PYTHON,str(ROOT/'code/worker10.py'),action],check=True)
 state=w.c.read(ROOT/'state.json',{})
 if action=='resume' and state and not state.get('finished') and not w.q.alive(state.get('worker')):
  with open(ROOT/'logs/worker_restart.log','a') as output:
   child=subprocess.Popen([w.c.PYTHON,str(ROOT/'code/worker10.py'),'run'],cwd=w.c.REPO,env=w.env(),stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
  print('Detached owned worker restart',child.pid,'original deadline',state['deadline'])
 elif action=='resume' and state.get('finished'):print('Bounded queue already collected; no deadline reset or duplicate runs')
if __name__=='__main__':main()
