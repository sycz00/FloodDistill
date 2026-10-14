"""Explicit source-only runtime action; the worker reserves complete readout blocks."""
import sys,time
import common10 as c
import evaluate_gpu10 as evaluator

def main():
    name=sys.argv[1];job=c.read(c.ROOT/'manifest.json')['jobs'][name]
    evaluator.profile_final(output_name=job['profile_output'],require_final_freeze=job['require_final_freeze'])
    result=c.read(c.ROOT/job['profile_output'])
    assert result.get('outcome')=='completed' and result.get('source_only') is True
    assert result['frozen_model_manifest_sha256']==c.sha(c.ROOT/'readout_models.json')
    assert result['frozen_settings_sha256']==c.sha(c.ROOT/'operating_points.json')
    c.save(c.ROOT/'stage_results'/(name+'.json'),dict(result,job=name,finished_at=time.time(),profile_path=str(c.ROOT/job['profile_output']),profile_sha256=c.sha(c.ROOT/job['profile_output'])))

if __name__=='__main__':main()
