"""Explicit numerical readout policy and additive artifact names, frozen before GT."""
from pathlib import Path
import common_v9 as c
ALIASES={name:name.replace('.json','_final02.json') for name in ['manifest.json','final_models.json','source_operating_points.json','gpu_inference_normal.json','gpu_inference_sealed.json','evaluation_source.json','final_evaluation_freeze.json','conditional_admission.json']}
def activate():
    if getattr(c,'_final02_active',False):return
    c._final02_active=True
    rd,sv,sh=c.read,c.save,c.sha
    def mapped(p):
        p=Path(p)
        return c.ROOT/ALIASES[p.name] if p.parent==c.ROOT and p.name in ALIASES else p
    c.read=lambda p,default=None:rd(mapped(p),default)
    c.save=lambda p,value:sv(mapped(p),value)
    # Receipt hashes always identify their literal, preserved artifact paths.
    c.sha=sh
