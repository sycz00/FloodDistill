# CKA cache (supplement Sec. S8, Fig. S3)

Four seeds x seven conditions (untrained, CE/Logit-KD/OFA on Sen1Floods11 N=252 and on
pseudo-labels N=2,500), spatial CKA on an 8x8 grid, teacher `--official_ckpt hf`.
Each file records the probe it was measured with under `"probe"`. Regenerate with:

    python analysis/cka.py --official_ckpt hf --student_arch efficientvit_b0 \
        --data_root $DATA_ROOT/sen1floods11 --seed 42 --n_batches 100 --cka_mode spatial --spatial_size 8 \
        --out_json analysis/cka_cache/cka_spatial_gt_vs_pseudo_s<SEED>.json \
        --compare_ckpts random: gt-ce:<run>/best.pt gt-lkd:<run>/best.pt gt-ofa:<run>/best.pt \
            ps2500-ce:<run>/best.pt ps2500-lkd:<run>/best.pt ps2500-ofa:<run>/best.pt

Plot it with `python figures/fig_cka.py --src_dir analysis/cka_cache`. Without `--src_dir`,
the script reads the paper source records.

