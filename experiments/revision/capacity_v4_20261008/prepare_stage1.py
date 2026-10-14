"""Freeze opt-in checks and a small Stage-1 manifest using the existing runner."""
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments import revision_queue as q
from experiments.compression_prepare import snapshot,DATA,TEACHER,TEACHER_SHA

BASE=Path(__file__).resolve().parent
V2=ROOT/'experiments/revision/compression_v2_20261007'
capsule,hashes=snapshot(BASE)
version=BASE/'code/v1';version.mkdir(parents=True,exist_ok=False)
for name in ('stage1_checks.py','simulator_v4.py'):
    shutil.copy2(BASE/name,version/name);(version/name).chmod(0o444)
shared=[]
def pin(path):return {'path':str(Path(path).resolve()),'sha256':q.file_hash(path)}
receipt=q.read_json(DATA/'original_N2500_receipt.json')
assert receipt['dataset_sha256']=='1ac4ef17efae3ebb0d3fb8daad38b4541ef04627c07a60e1e2c42e00d2efbc60'
assert q.file_hash(TEACHER)==TEACHER_SHA
shared.append(pin(TEACHER))
for row in receipt['files']:
    p=Path(receipt['pool'])/row['name']
    assert p.stat().st_size==row['bytes'] and q.file_hash(p)==row['sha256']
    shared.append({'path':str(p),'sha256':row['sha256']})
for name in ('original_N2500_index.json','original_N2500_receipt.json','original_source_manifest.json'):
    shared.append(pin(DATA/name))
for name,sha in hashes.items():shared.append({'path':str(capsule/name),'sha256':sha})
common=BASE/'shared_input_identity.json'
q.atomic_json(common,{'files':shared,'capsule':str(capsule),'teacher_sha256':TEACHER_SHA,
    'cohort_sha256':receipt['dataset_sha256'],'n_cache_files':len(receipt['files']),
    'script_sources':[pin(version/name) for name in ('stage1_checks.py','simulator_v4.py')],
    'task':pin(ROOT/'prompts/prompt6_corrected.md')})
common.chmod(0o444)
old=q.read_json(V2/'export_remaining_20261008/manifest.json')
directory=BASE/'stage1_measurement';directory.mkdir(exist_ok=False)
shutil.copy2(capsule/'experiments/revision_queue.py',directory/'revision_queue.py')
(directory/'revision_queue.py').chmod(0o444)
jobs=[]
for cell in ('A3_W8A8','A3_W4A8','A3_W4A4','A1_W4A8'):
    spec=V2/'pilot_final/specs'/f'{cell}_B1_s42.json'
    inputs=[pin(common),pin(spec),pin(version/'stage1_checks.py'),pin(version/'simulator_v4.py'),
        pin(V2/'results'/f'{cell}_interaction.pt'),pin(BASE/'simulator_unit.json')]
    out=BASE/'results'/f'{cell}_measurement.json'
    job={'id':cell+'_measurement','kind':'command','cwd':str(capsule),'gpu':True,'deps':[],
         'argv':[sys.executable,str(version/'stage1_checks.py'),'--capsule',str(capsule),
                 '--identity-manifest',str(common),'--spec',str(spec),'--cell',cell,'--out',str(out),'--batches','8'],
         'inputs':inputs,'outputs':[{'path':str(out),'type':'json','equals':{'status':'complete','outcome':'completed'},
                     'required_keys':['validation','rows','integer_norm_and_PTQ_gate','summary'],
                     'finite':['elapsed_seconds','peak_reserved_mb']}],
         'required_free_mb':12288,'max_seconds':1200,'shutdown_grace_seconds':10}
    jobs.append(job)
manifest={k:copy.deepcopy(old[k]) for k in ('schema_version','env','gpu_devices','gpu_lock','headroom_mb','host_limits','nice','poll_seconds')}
manifest['env']['PYTHONPATH']=str(capsule)+':/home/schmalstieg/repos/_deps/efficientvit:/home/schmalstieg/repos/_deps/flooddistill_revision_python'
manifest.update(jobs=jobs,gpu_budget_seconds=4800,stage='1.1/1.3 first measurement gates',
    stage1_cap_seconds=7200,programme_target_seconds=108000,
    scope='Fixed-weight focal attribution and simulator/floor checks only; not full Stage1 adoption or training.',
    remaining_stage1_seconds=2400,later_stages='gated; not yet launched or declared queued',
    scientific_state='No simulator adopted until all Stage1.3 checks including selected-checkpoint parity and TRAIN forward disagreement are reviewed.')
q.atomic_json(directory/'manifest.json',manifest);(directory/'manifest.json').chmod(0o444)
assert (directory/'manifest.json').stat().st_size<10*1024**2
q.load_manifest(directory/'manifest.json')
print(json.dumps({'manifest':str(directory/'manifest.json'),'capsule':str(capsule),'jobs':len(jobs),
                  'manifest_bytes':(directory/'manifest.json').stat().st_size,'reserved_gpu_seconds':4800}))
