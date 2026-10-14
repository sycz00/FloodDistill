#!/bin/bash
# usage: run.sh <gpu> <python args...>  -- runs with the frozen-capsule environment of the stage manifests
GPU=$1; shift
eval "$(/home/schmalstieg/miniconda3/envs/esi/bin/python -c "
import json,shlex;e=json.load(open('$(dirname "$0")/env.json'))
print(' '.join('export %s=%s;'%(k,shlex.quote(v)) for k,v in e.items()))")"
export CUDA_VISIBLE_DEVICES=$GPU
exec nice -n 10 /home/schmalstieg/miniconda3/envs/esi/bin/python "$@"
