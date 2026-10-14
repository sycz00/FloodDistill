"""Pseudo-labelled dataset and pseudo-label generation.

PseudoLabeledDataset    loads (image, mask) .pt files written by generate_pseudo_labels()
generate_pseudo_labels  runs the frozen teacher over the unlabelled GDACS pool and saves
                        confident predictions
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.unlabeled import FloodUnlabeledDataset


class PseudoLabeledDataset(Dataset):
    """Loads (image, mask) pairs saved by generate_pseudo_labels().

    Each .pt file holds {"image": Tensor[6, H, W], "mask": Tensor[H, W]} and optionally
    "confidence" (Tensor[H, W], float16, per-pixel teacher confidence) and "soft_logits"
    (Tensor[2, H, W], float16, teacher logits).

    Args:
        pseudo_dir:           Directory with the .pt files.
        n_samples:            Number of files, drawn with `seed` from all files or, with
                              use_curated, from the index. None loads all.
        pixel_conf_threshold: Pixels with lower confidence get mask -1 (ignored by the loss).
                              Needs "confidence"; 0.0 disables it.
        use_curated:          Restrict the pool to `index_file` (e.g. diversity_N<n>_index.json)
                              and draw `n_samples` uniformly from it. The selection score is only
                              used when the index is built, so an index larger than n_samples
                              gives a random subset of it.
    """

    def __init__(
        self,
        pseudo_dir: str,
        n_samples: int = 6753,
        pixel_conf_threshold: float = 0.0,
        seed: int | None = None,
        use_curated: bool = False,
        index_file: str = "curated_index.json",
        extra_keys: frozenset | None = None,
        mult_noise: bool = False,
        mult_noise_range: tuple[float, float] = (0.80, 1.20),
        mult_noise_p: float = 0.5,
        geom_aug: bool = False,
        geom_aug_p: float = 0.5,
    ) -> None:
        from pathlib import Path
        import json
        import random

        base = Path(pseudo_dir)

        if use_curated:
            # Indices ship with the repo under data/ and data/index_files_curations/ (built on
            # our own pool); PSEUDO_DIR is checked first so a pool-local index always wins.
            index_path = next(
                (p for p in (base / index_file,
                              Path("data") / index_file,
                              Path("data/index_files_curations") / index_file)
                 if p.exists()),
                None,
            )
            if index_path is None:
                raise FileNotFoundError(
                    f"Curated index {index_file!r} not found in {pseudo_dir}, data/, or "
                    f"data/index_files_curations/.\n"
                    f"Generate it first:\n"
                    f"  python -m data.curate_pseudo_gdacs --pseudo_dir {pseudo_dir}\n"
                    f"Or for diversity index:\n"
                    f"  python -m data.diversity_pseudo --pseudo_dir {pseudo_dir}"
                )
            with open(index_path) as f:
                idx = json.load(f)
            all_files = [base / fname for fname in idx["files"]]
            n_pool = idx.get("n_curated", idx.get("n_selected", len(all_files)))
            # Draw N files from the curated pool with the training seed.
            if n_samples is None or n_samples >= len(all_files):
                self.files = all_files
            else:
                rng = random.Random(seed)
                self.files = rng.sample(all_files, n_samples)
            ff_range = (f"  ff∈[{idx['min_ff']},{idx['max_ff']}]"
                        if "min_ff" in idx else "")
            print(f"[pseudo] {index_file}: {len(self.files)} of {n_pool} patches"
                  f" (seed={seed}, from {idx.get('n_total', len(all_files))} total)"
                  f"{ff_range}")
        else:
            all_files = sorted(base.glob("*.pt"))
            if n_samples is None:
                self.files = all_files
            else:
                # Separate RNG, so the selection does not depend on the global random state.
                rng = random.Random(seed)
                self.files = rng.sample(all_files, min(n_samples, len(all_files)))

        if not self.files:
            raise FileNotFoundError(
                f"Pseudo-label pool is empty: no .pt files in {pseudo_dir!r}.\n"
                f"Generate it first (needs the teacher and the unlabeled GDACS tiles):\n"
                f"  python scripts/train.py pseudo generate --official_ckpt hf \\\n"
                f"      --unlabeled_root <tiles> --unlabeled_source flood --out_dir {pseudo_dir}\n"
                f"See docs/01_datasets.md and docs/EXTERNAL_SETUP.md."
            )

        # Check that the index belongs to this pool (indices are in git, tiles are not).
        absent = [f for f in self.files if not f.exists()]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)} of {len(self.files)} scenes listed in "
                f"{index_file if use_curated else '<pool>'} are missing from "
                f"{pseudo_dir!r} (first: {absent[0].name}).\n"
                f"The index ships with the repo; the pseudo-label tiles do not. "
                f"Build the pool before running a pseudo-label experiment:\n"
                f"  python scripts/train.py pseudo generate --official_ckpt hf \\\n"
                f"      --unlabeled_root <tiles> --unlabeled_source flood --out_dir {pseudo_dir}\n"
                f"See docs/01_datasets.md and docs/EXTERNAL_SETUP.md."
            )
        self.pixel_conf_threshold = pixel_conf_threshold
        if not use_curated:
            print(f"[pseudo] {len(self.files)} pseudo-labeled scenes from {pseudo_dir!r}"
                  f"  pixel_conf_threshold={pixel_conf_threshold}")

        # Optional keys are taken from the first file, and every sample returns this key set
        # (missing keys are filled with zeros) so default_collate works. An explicit
        # extra_keys (e.g. frozenset()) is used as given, e.g. when mixing with labelled data.
        _OPTIONAL = ("soft_logits", "gradcam", "sens_map", "gradcam_contrast",
                     "frd_resp", "frd_resp_local", "frd_resp_cf")
        _peek = torch.load(self.files[0], map_location="cpu", weights_only=True)
        if extra_keys is not None:
            self._extra_keys: frozenset = frozenset(extra_keys) & frozenset(_OPTIONAL)
        else:
            self._extra_keys = frozenset(
                k for k in _OPTIONAL if k in _peek
            )
        H, W = _peek["mask"].shape
        self._hw = (H, W)
        # Per-channel multiplicative reflectance jitter (see _apply_mult_noise); off by default.
        self.mult_noise = bool(mult_noise)
        self.mult_noise_range = tuple(mult_noise_range)
        self.mult_noise_p = float(mult_noise_p)
        if self.mult_noise:
            from data.sen1floods11 import NORM_MEAN, NORM_STD
            # mu/sigma for the offset term (k-1)*mu/sigma, see _apply_mult_noise.
            self._mn_shift = (torch.tensor(NORM_MEAN, dtype=torch.float32)
                              / torch.tensor(NORM_STD, dtype=torch.float32)).view(-1, 1, 1)
            # Separate stream so augmentation does not change the file selection above.
            # Only used with num_workers=0; see _mn_generator.
            self._mn_seed = 0 if seed is None else int(seed)
            self._mn_gen  = torch.Generator().manual_seed(self._mn_seed)
            print(f"[pseudo] multiplicative reflectance jitter ON: "
                  f"k~U{self.mult_noise_range} per channel, p={self.mult_noise_p}")
        # Flips and rot90 as in the Sen1Floods11 training transform (data/sen1floods11.py):
        # A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5). Off by default.
        self.geom_aug = bool(geom_aug)
        self.geom_aug_p = float(geom_aug_p)
        if self.geom_aug:
            if H != W:
                raise ValueError(f"geom_aug needs square tiles for rot90; got {H}x{W}")
            self._geom_seed = 0 if seed is None else int(seed)
            self._geom_gen = torch.Generator().manual_seed(self._geom_seed ^ 0x5DEECE66D)
            print(f"[pseudo] geometric augmentation ON: hflip/vflip/rot90, "
                  f"p={self.geom_aug_p} each (matches the GT train_tf)")
        # K (number of FRD perturbations) locked in from the first enriched file
        # so the zero fallback for partially-enriched dirs has the right shape.
        self._frd_k = _peek["frd_resp"].shape[0] if "frd_resp" in _peek else 4
        self._frd_k_local = (_peek["frd_resp_local"].shape[0]
                             if "frd_resp_local" in _peek else 4)
        self._frd_k_cf = (_peek["frd_resp_cf"].shape[0]
                          if "frd_resp_cf" in _peek else 4)
        if self._extra_keys:
            print(f"[pseudo] optional keys in dataset: {sorted(self._extra_keys)}")

    def __len__(self) -> int:
        return len(self.files)

    def _mn_generator(self, idx: int) -> torch.Generator:
        """RNG for the jitter of sample `idx`, also correct with num_workers > 0.

        Each worker holds a copy of the dataset, so one seeded stream on `self` would repeat in
        every worker. In a worker the generator is derived from `worker_info.seed` and the sample
        index, which gives independent draws per sample and new draws each epoch.
        """
        wi = torch.utils.data.get_worker_info()
        if wi is None:
            return self._mn_gen          # single process: the dedicated stream is already fine
        # 2^61-1 keeps the seed inside the positive int64 range manual_seed accepts.
        return torch.Generator().manual_seed(
            (int(wi.seed) * 0x9E3779B97F4A7C15 + idx * 0xBF58476D1CE4E5B9) % ((1 << 61) - 1)
        )

    def _apply_mult_noise(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Per-channel multiplicative reflectance jitter on a normalised tensor.

        Matches the Sen1Floods11 training augmentation
        A.MultiplicativeNoise(multiplier=(0.80, 1.20), per_channel=True, p=0.5), which is applied
        in reflectance space. The .pt files store x = (rho - mu) / sigma, so scaling rho by k gives

            x' = k * x + (k - 1) * mu / sigma

        and no tiles need to be regenerated. One k per channel; the transform is applied with
        probability p per sample.
        """
        g = self._mn_generator(idx)
        if float(torch.rand(1, generator=g)) >= self.mult_noise_p:
            return x
        lo, hi = self.mult_noise_range
        k = torch.empty(x.shape[0], 1, 1, dtype=torch.float32).uniform_(
            lo, hi, generator=g)
        return (k * x.float() + (k - 1.0) * self._mn_shift).to(x.dtype)

    def _geom_generator(self, idx: int) -> torch.Generator:
        """RNG for the geometric augmentation, per worker as in _mn_generator and separate from the jitter stream."""
        wi = torch.utils.data.get_worker_info()
        if wi is None:
            return self._geom_gen
        return torch.Generator().manual_seed(
            (int(wi.seed) * 0xD6E8FEB86659FD93 + idx * 0xA24BAED4963EE407 + 0x9E3779B9)
            % ((1 << 61) - 1)
        )

    def _apply_geom(self, out: dict, idx: int) -> dict:
        """Apply one dihedral transform to all spatial tensors of the sample.

        All returned values have shape (..., H, W), so image, mask and soft_logits stay aligned.
        """
        g = self._geom_generator(idx)
        gates = torch.rand(3, generator=g)
        k = int(torch.randint(0, 4, (1,), generator=g).item())
        ops = []
        if float(gates[0]) < self.geom_aug_p:
            ops.append(lambda t: torch.flip(t, dims=[-1]))          # HorizontalFlip
        if float(gates[1]) < self.geom_aug_p:
            ops.append(lambda t: torch.flip(t, dims=[-2]))          # VerticalFlip
        if float(gates[2]) < self.geom_aug_p and k:
            ops.append(lambda t: torch.rot90(t, k, dims=[-2, -1]))  # RandomRotate90
        if not ops:
            return out
        for key, t in out.items():
            if torch.is_tensor(t) and t.ndim >= 2:
                for op in ops:
                    t = op(t)
                out[key] = t.contiguous()
        return out

    def __getitem__(self, idx: int) -> dict:
        item = torch.load(self.files[idx], weights_only=True)
        mask = item["mask"].clone()

        # Ignore uncertain pixels in the loss; accepted scenes still have boundary pixels
        # with confidence around 0.55-0.75.
        if self.pixel_conf_threshold > 0.0 and "confidence" in item:
            uncertain = item["confidence"].float() < self.pixel_conf_threshold
            mask[uncertain] = -1

        img = item["image"]
        if self.mult_noise:
            img = self._apply_mult_noise(img, idx)
        out = {"image": img, "mask": mask}

        # Same optional keys for every sample; missing ones are filled with zeros.
        if "soft_logits" in self._extra_keys:
            out["soft_logits"] = item.get(
                "soft_logits",
                torch.zeros((2,) + self._hw, dtype=torch.float16),
            )
        if "gradcam" in self._extra_keys:
            out["gradcam"] = item.get(
                "gradcam",
                torch.zeros(self._hw, dtype=torch.float16),
            )
        if "sens_map" in self._extra_keys:
            out["sens_map"] = item.get(
                "sens_map",
                torch.zeros((6,) + self._hw, dtype=torch.float16),
            )
        if "gradcam_contrast" in self._extra_keys:
            out["gradcam_contrast"] = item.get(
                "gradcam_contrast",
                torch.zeros(self._hw, dtype=torch.float16),
            )
        if "frd_resp" in self._extra_keys:
            out["frd_resp"] = item.get(
                "frd_resp",
                torch.zeros((self._frd_k,) + self._hw, dtype=torch.float16),
            )
        if "frd_resp_local" in self._extra_keys:
            out["frd_resp_local"] = item.get(
                "frd_resp_local",
                torch.zeros((self._frd_k_local,) + self._hw, dtype=torch.float16),
            )
        if "frd_resp_cf" in self._extra_keys:
            out["frd_resp_cf"] = item.get(
                "frd_resp_cf",
                torch.zeros((self._frd_k_cf,) + self._hw, dtype=torch.float16),
            )

        # Applied last so one transform covers all keys; order relative to the jitter does not matter.
        if self.geom_aug:
            out = self._apply_geom(out, idx)

        return out


@torch.no_grad()
def generate_pseudo_labels(
    teacher,
    unlabeled_root: str,
    out_dir: str,
    unlabeled_source: str = "flood",
    confidence_threshold: float = 0.90,
    min_flood_frac: float = 0.0,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 8,
    device: torch.device | None = None,
    store_pixel_confidence: bool = False,
    store_soft_logits: bool = False,
) -> int:
    """Run the teacher over unlabelled scenes and save confident predictions as masks.

    A scene is kept if its mean pixel confidence >= confidence_threshold and its flood
    fraction >= min_flood_frac.

    Args:
        unlabeled_root:         Root directory of the unlabelled data.
        unlabeled_source:       'flood' (GDACS pool, B8A); the only supported value.
        confidence_threshold:   Minimum mean pixel confidence of a scene.
        min_flood_frac:         Minimum predicted flood fraction; 0.0 keeps non-flood scenes.
        store_pixel_confidence: Also save per-pixel confidence (float16).
        store_soft_logits:      Also save teacher logits (float16, 2xHxW).

    Inputs are normalised with the teacher's fine-tuning statistics; saved images stay
    normalised, as the student expects them.

    Files are numbered in save order (`pseudo_{saved:06d}.pt`), so the number is not the scene
    index. Each file stores a "source" entry (scene directory relative to unlabeled_root), and
    `source_manifest.csv` is written next to the files. It links tiles to patch_quality.csv.
    """
    import csv
    from pathlib import Path
    if device is None:
        device = next(teacher.parameters()).device

    os.makedirs(out_dir, exist_ok=True)
    teacher.eval()

    if unlabeled_source != "flood":
        raise ValueError(
            f"unlabeled_source={unlabeled_source!r} is not supported; the only pool this "
            f"project builds is the GDACS flood-event one ('flood')."
        )
    ds = FloodUnlabeledDataset(unlabeled_root, img_size=img_size, normalize=True)

    print(f"[generate] {len(ds)} scenes from {unlabeled_root!r}  "
          f"bands={ds._BAND_ORDER}  normalize=True")

    # shuffle=False keeps DataLoader emission order == ds.scenes order, so the running
    # `total` counter indexes straight into ds.scenes and gives each saved tile its source.
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, drop_last=False)

    manifest    = open(os.path.join(out_dir, "source_manifest.csv"), "w", newline="")
    mwriter     = csv.writer(manifest)
    mwriter.writerow(["file", "source", "location", "scene_system_index",
                      "flood_frac", "mean_conf"])

    saved       = 0
    total       = 0
    flood_fracs = []
    mean_confs  = []

    for batch_idx, batch in enumerate(dl):
        images = batch["image"].to(device, dtype=torch.float32)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = teacher(images)   # (B, 2, H, W)

        probs      = F.softmax(logits.float(), dim=1)   # (B, 2, H, W)
        confidence = probs.max(dim=1).values             # (B, H, W)
        hard_mask  = probs.argmax(dim=1)                 # (B, H, W)
        flood_frac = hard_mask.float().mean(dim=(1, 2))  # (B,)
        mean_conf  = confidence.mean(dim=(1, 2))         # (B,)

        flood_fracs.extend(flood_frac.cpu().tolist())
        mean_confs.extend(mean_conf.cpu().tolist())

        for i in range(images.shape[0]):
            scene = ds.scenes[total]          # shuffle=False, so emission order == ds.scenes
            total += 1
            if (mean_conf[i].item() >= confidence_threshold
                    and flood_frac[i].item() >= min_flood_frac):
                base    = f"pseudo_{saved:06d}.pt"
                fname   = os.path.join(out_dir, base)
                # "<location>/<scene>": the last two path parts (relative_to fails for symlinked roots).
                rel     = f"{scene.parent.name}/{scene.name}"
                payload = {
                    "image":  images[i].cpu(),
                    "mask":   hard_mask[i].cpu(),
                    "source": rel,
                }
                if store_pixel_confidence:
                    payload["confidence"] = confidence[i].cpu().half()
                if store_soft_logits:
                    payload["soft_logits"] = logits[i].cpu().half()
                torch.save(payload, fname)
                mwriter.writerow([base, rel, scene.parent.name, scene.name,
                                  f"{flood_frac[i].item():.6f}",
                                  f"{mean_conf[i].item():.6f}"])
                saved += 1

        if (batch_idx + 1) % 20 == 0:
            manifest.flush()
            print(f"  [generate] {total}/{len(ds)} scanned  "
                  f"{saved} saved ({100*saved/max(total,1):.1f}%)")

    manifest.close()

    ff = np.array(flood_fracs)
    mc = np.array(mean_confs)
    print(f"\n[generate] Teacher predictions on {total} scenes:")
    print(f"  flood_frac  — mean={ff.mean():.4f}  "
          f"p50={np.percentile(ff,50):.4f}  "
          f"p95={np.percentile(ff,95):.4f}  "
          f"max={ff.max():.4f}")
    print(f"  mean_conf   — mean={mc.mean():.4f}  "
          f"min={mc.min():.4f}  "
          f"p5={np.percentile(mc,5):.4f}")
    print(f"  scenes with flood_frac>1%:  {(ff>0.01).sum()}")
    print(f"  scenes with flood_frac>5%:  {(ff>0.05).sum()}")
    print(f"  scenes with mean_conf>={confidence_threshold}: "
          f"{(mc>=confidence_threshold).sum()}")
    if (ff > 0.01).sum() == 0:
        print(f"\n  *** The teacher predicted flood on no scene in this pool. ***")
        print(f"  Check that --unlabeled_root points at the GDACS download tree and that "
              f"the teacher loaded (--official_ckpt hf).")
    print(f"\n[generate] saved {saved}/{total} scenes "
          f"(conf>={confidence_threshold}, flood_frac>={min_flood_frac})")
    print(f"[generate] provenance → {os.path.join(out_dir, 'source_manifest.csv')}")
    return saved


@torch.no_grad()
def relabel_pseudo_labels(
    teacher,
    pseudo_dir: str,
    out_dir: str,
    index_file: str | None = "diversity_N2500_index.json",
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device | None = None,
    store_pixel_confidence: bool = True,
    store_soft_logits: bool = False,
) -> int:
    """Relabel an existing pseudo pool with another teacher.

    The stored image tensors are kept unchanged; mask, confidence and soft_logits are replaced.
    File names stay the same, so index files remain valid and are copied to out_dir. No
    confidence filtering is applied, so pool membership does not change.

    Args:
        teacher:    Frozen nn.Module, forward(x) -> logits, e.g. the UNet ensemble from
                    models.bridge_teacher.build_bridge_teacher.
        index_file: Only relabel the files in this JSON index and copy it to out_dir.
                    None or "" relabels every *.pt.

    Returns:
        Number of files written.
    """
    import json
    import shutil
    from pathlib import Path

    if device is None:
        device = next(teacher.parameters()).device
    teacher.eval()

    src = Path(pseudo_dir)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    if index_file:
        index_path = src / index_file
        with open(index_path) as f:
            idx = json.load(f)
        files = [src / fname for fname in idx["files"]]
        shutil.copy2(index_path, dst / index_file)
        print(f"[relabel] {len(files)} files from {index_file} "
              f"(index copied to {dst})")
    else:
        files = sorted(src.glob("*.pt"))
        print(f"[relabel] {len(files)} files (full pool, no index)")

    class _Imgs(Dataset):
        def __len__(self):
            return len(files)

        def __getitem__(self, i):
            item = torch.load(files[i], map_location="cpu", weights_only=True)
            return {"image": item["image"], "old_mask": item["mask"], "idx": i}

    dl = DataLoader(_Imgs(), batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, drop_last=False)

    saved         = 0
    changed_px    = 0
    total_px      = 0
    flood_fracs   = []
    mean_confs    = []

    for batch in dl:
        images = batch["image"].to(device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = teacher(images)                     # (B, 2, H, W)
        probs      = F.softmax(logits.float(), dim=1)
        confidence = probs.max(dim=1).values             # (B, H, W)
        hard_mask  = probs.argmax(dim=1)                 # (B, H, W)

        changed_px += int((hard_mask.cpu() != batch["old_mask"]).sum())
        total_px   += int(batch["old_mask"].numel())
        flood_fracs.extend(hard_mask.float().mean(dim=(1, 2)).cpu().tolist())
        mean_confs.extend(confidence.mean(dim=(1, 2)).cpu().tolist())

        for i in range(images.shape[0]):
            fpath   = files[int(batch["idx"][i])]
            payload = {
                "image": batch["image"][i],              # unchanged input tensor
                "mask":  hard_mask[i].cpu(),
            }
            if store_pixel_confidence:
                payload["confidence"] = confidence[i].cpu().half()
            if store_soft_logits:
                payload["soft_logits"] = logits[i].float().cpu().half()
            torch.save(payload, dst / fpath.name)
            saved += 1

        if saved % (batch_size * 20) < batch_size:
            print(f"  [relabel] {saved}/{len(files)}")

    ff = np.array(flood_fracs)
    mc = np.array(mean_confs)
    print(f"\n[relabel] wrote {saved} files → {dst}")
    print(f"  labels changed vs source pool: {changed_px:,}/{total_px:,} px "
          f"({100 * changed_px / max(total_px, 1):.2f}%)")
    print(f"  flood_frac — mean={ff.mean():.4f}  p50={np.percentile(ff, 50):.4f}  "
          f"p95={np.percentile(ff, 95):.4f}")
    print(f"  mean_conf  — mean={mc.mean():.4f}  min={mc.min():.4f}  "
          f"p5={np.percentile(mc, 5):.4f}")
    return saved
