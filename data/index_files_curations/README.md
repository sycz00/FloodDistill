# Curation indices behind the paper numbers

These are the subset indices the reported runs were trained on. They were built on our
GDACS pseudo-label pool of 23,720 tiles. The pool itself (38 GB) is not included.

| File | Used for |
|---|---|
| `diversity_N{20,40,50,100,150,200,252}_index.json` | N sweep, Figs. 2, 4, 5 and S4; N=252 is Table I |
| `diversity_N2500_index.json` | Table II, deployed student, Fig. S1 (FPS) |
| `diversity_N4000_index.json` | N=4,000 scaling point in Figs. 4, 5 and S4 |
| `diversity_raw_index.json`, `confidence_filtered_index.json`, `confidence_raw_index.json`, `glister_index.json`, `glister_raw_index.json` | Selector comparison, Fig. S1 |
| `pool_features.csv` | Per-tile descriptor and statistics of the pool |
| `consistency.csv` | Teacher consistency per tile (input of R2-Core) |
| `source_manifest.csv` | Pool file name to GDACS location and Sentinel-2 scene ID |

All FPS indices are prefixes of `diversity_N4000_index.json`.

## Using them on your own pool

A pool you generate yourself (README step 3) gets different file numbers, and a few scenes
may be missing or differ slightly. The file names in these indices only refer to our pool.
Map them through the scene IDs:

```bash
python experiments/map_index_to_pool.py \
  --src_pool data/index_files_curations --src_index diversity_N4000_index.json \
  --dst_pool data/pseudo_gdacs --n 252 --out_name mapped_diversity_N252_index.json
```

Use the N=4,000 ranking as source. If a scene is missing from your pool, the script takes
the next tile in the FPS order until N is reached, and writes how many tiles came from the
original cut into the output. `experiments/nsweep_pseudo_vs_gt.sh` reads the mapped files
with `IDX_PREFIX=mapped_diversity`, `experiments/repro_n2500.sh` with
`IDX=mapped_diversity_N2500_index.json`.

This was tested with the FPS indices. For the selector comparison on a new pool, rebuild
the other indices with `experiments/build_indices.sh`.
