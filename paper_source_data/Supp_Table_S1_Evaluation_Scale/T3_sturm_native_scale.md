# STURM evaluation scale

The same checkpoints evaluated at 128, 224 and 512 pixels. STURM's ground
extent stays fixed at 1.28 km. Results are also in `T3_sturm_native_scale.csv`.

```bash
for size in 128 224; do
  python evaluation/perscene_eval.py --gpu 1 --datasets sturm \
    --families sturmnative --conds clean --sturm_img_size "$size" \
    --out_dir "runs/perscene_sturmsize/$size"
done
python evaluation/scale_control_collect.py --block deployed
```

## Accuracy

Four training seeds where applicable; the teacher and index are seedless.

| model | 128 px (10.00 m) | 224 px (5.71 m) | 512 px (2.50 m) | 128 − 512 |
|---|---|---|---|---|
| **micro mIoU** | | | | |
| MNDWI > 0 (2-band rule) | 0.7443 | 0.7449 | 0.7467 | −0.0024 |
| Teacher Prithvi-EO-2.0 300 M | **0.7619** | 0.7569 | 0.7550 | **+0.0069** |
| pseudo-EvIT-CE N=2500 | 0.7422 ± .004 | 0.7558 ± .003 | 0.7537 ± .002 | −0.0116 |
| pseudo-EvIT-LKD N=2500 | 0.7436 ± .006 | 0.7596 ± .004 | 0.7578 ± .003 | −0.0142 |
| pseudo-EvIT-OFA N=2500 (deployed) | 0.7464 ± .005 | 0.7590 ± .004 | 0.7561 ± .005 | −0.0097 |
| pseudo-UNet-CE N=2500 | 0.7260 ± .011 | 0.7424 ± .006 | 0.7297 ± .014 | −0.0037 |
| pseudo-UNet-LKD N=2500 | 0.7353 ± .010 | 0.7523 ± .003 | 0.7469 ± .005 | −0.0116 |
| **micro flood IoU** | | | | |
| MNDWI > 0 | **0.6416** | 0.6422 | 0.6447 | −0.0031 |
| Teacher | **0.6538** | 0.6460 | 0.6434 | +0.0103 |
| pseudo-EvIT-OFA N=2500 | 0.6258 ± .009 | 0.6461 ± .007 | 0.6421 ± .008 | −0.0163 |
| pseudo-EvIT-LKD N=2500 | 0.6219 ± .011 | 0.6482 ± .007 | 0.6472 ± .006 | −0.0253 |
| pseudo-EvIT-CE N=2500 | 0.6200 ± .008 | 0.6431 ± .005 | 0.6406 ± .004 | −0.0206 |
| pseudo-UNet-LKD N=2500 | 0.6059 ± .018 | 0.6365 ± .005 | 0.6299 ± .007 | −0.0240 |
| pseudo-UNet-CE N=2500 | 0.5912 ± .018 | 0.6208 ± .011 | 0.6035 ± .023 | −0.0123 |
| **macro flood IoU** | | | | |
| MNDWI > 0 | **0.3893** | 0.3895 | 0.3891 | +0.0002 |
| Teacher | 0.3711 | 0.3753 | 0.3723 | −0.0012 |
| pseudo-EvIT-OFA N=2500 | 0.3436 ± .006 | 0.3618 ± .004 | 0.3664 ± .004 | −0.0228 |

## Paired flood-map differences

Student/rule minus teacher flood IoU, with wins out of 46 maps and sign-test p.

| model | 512 px (published) | 224 px | 128 px (native) |
|---|---|---|---|
| pseudo-EvIT-CE N=2500 | −0.0108, 23/46, p=1 | −0.0225, 26/46, p=0.46 | **−0.0384, 14/46, p=0.011** |
| pseudo-EvIT-LKD N=2500 | +0.0029, 24/46, p=0.88 | −0.0097, 22/46, p=0.88 | −0.0298, 16/46, p=0.054 |
| pseudo-EvIT-OFA N=2500 | −0.0059, 20/46, p=0.46 | −0.0135, 18/46, p=0.18 | −0.0275, 16/46, p=0.054 |
| pseudo-UNet-CE N=2500 | −0.0283, 20/46, p=0.46 | −0.0312, 13/46, p=0.0045 | **−0.0496, 9/46, p=4.1e-5** |
| pseudo-UNet-LKD N=2500 | −0.0068, 21/46, p=0.66 | −0.0162, 14/46, p=0.011 | **−0.0396, 8/46, p=9.3e-6** |
| MNDWI > 0 | +0.0168, 28/46, p=0.18 | +0.0141, 24/46, p=0.88 | +0.0182, 23/46, p=1 |
