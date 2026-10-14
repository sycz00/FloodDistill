"""Admission uses measured full exposed-set runtime and target WINDOW COUNT only."""
import common_v9 as c
from policy_final03 import activate
activate()
source=c.read(c.ROOT/'gpu_inference_normal.json');target=c.read(c.ROOT/'sealed_windows.json')
assert source['outcome']=='completed'
count=target['window_count'];records=source['records']
estimate=sum(r['seconds']/r['windows']*count for r in records if r['windows'])
conservative=1.25*estimate+120
c.save(c.ROOT/'sealed_cost_admission04.json',dict(created=c.q.now(),target_window_count=count,normal_measured_seconds=source['seconds'],model_records=len(records),full_matrix_estimated_seconds=estimate,conservative_seconds=conservative,admitted=conservative<=7200,ceiling_seconds=7200,
    criteria='All predictor rows together; cost/provenance only; 25% runtime headroom+120s setup; no target scores inspected; never drop slower models or seed controls to fit',status='READY' if conservative<=7200 else'DEFERRED_BUDGET'))
