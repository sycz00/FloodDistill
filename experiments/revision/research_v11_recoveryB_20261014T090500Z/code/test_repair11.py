"""CPU-only tests for V11's changed interfaces and scientific invariants."""
import copy
import json
import os
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '2'

import repair11 as r
torch, c, _, v = r.setup(2)
import numpy as np
from models.compression_quant import FixedQATConv2d, compression_train_mode, precision_map
from source11 import deficit_gate


def tiny_model(in_channels=2):
    conv = torch.nn.Conv2d(in_channels, 3, 1)
    quant = FixedQATConv2d(conv, torch.tensor([.1]), torch.tensor([0]),
                           torch.full((3,), .05), weight_bits=4, activation_bits=4)
    return torch.nn.Sequential(quant, torch.nn.BatchNorm2d(3), torch.nn.ReLU(),
                               torch.nn.Dropout2d(.2), torch.nn.Conv2d(3, 2, 1))


class ScoreTests(unittest.TestCase):
    def test_binary_vector_and_signed_effect(self):
        torch.manual_seed(7)
        teacher, fp, quant = [torch.randn(3, 2, 5, 4, dtype=torch.float64) * 9 for _ in range(3)]
        vector = r.priority_scores(teacher, fp, quant)['D']
        torch.testing.assert_close(vector, r.binary_extra_loss(teacher, fp, quant), rtol=1e-10, atol=1e-12)
        teacher = torch.tensor([0., 8.]).reshape(1, 2, 1, 1)
        fp = torch.tensor([0., 2.]).reshape_as(teacher)
        toward = torch.tensor([0., 4.]).reshape_as(teacher)
        away = torch.tensor([0., -2.]).reshape_as(teacher)
        self.assertLess(r.priority_scores(teacher, fp, toward)['D'].item(), 0)
        self.assertGreater(r.priority_scores(teacher, fp, away)['D'].item(), 0)
        self.assertTrue(torch.equal(r.priority_scores(teacher, fp, fp)['D'], torch.zeros(1, 1, 1)))

    def test_weights_counts_invalid_and_ties(self):
        score = torch.tensor([[[2., 2., 2., 2.], [1., 0., -1., float('nan')]],
                              [[float('nan')] * 4] * 2])
        valid = torch.ones_like(score, dtype=torch.bool)
        valid[0, 1, 3] = False
        valid[1] = False
        teacher = torch.tensor([[[1, 0, 1, 0], [1, 0, 1, 0]], [[0] * 4] * 2])
        chosen, weights, counts = r.select_priority(score, valid, teacher)
        self.assertTrue(chosen[0, 0, 0])
        self.assertEqual(int(chosen.sum()), 1)
        self.assertEqual(counts, [[0, 1], [0, 0]])
        self.assertAlmostEqual(float(weights[valid].mean()), 1., places=6)
        self.assertLessEqual(float(weights[valid].max() / weights[valid].min()), 4.)
        self.assertTrue((weights[~valid] == 0).all())
        for control, seed in [(score, None), (-score, None), (score, 1729)]:
            a, _, got = r.select_priority(control, valid, teacher, counts=counts, random_seed=seed)
            self.assertEqual(got, counts)
            self.assertFalse(a[~valid].any())
        singleton = torch.zeros((1, 1, 1), dtype=torch.bool)
        chosen, weights, counts = r.select_priority(torch.ones(1, 1, 1), ~singleton, singleton.long())
        self.assertFalse(chosen.any())
        self.assertEqual(weights.item(), 1.)

    def test_exact_uniform_fallback_and_gradients(self):
        torch.manual_seed(2)
        z = torch.randn(2, 2, 4, 4, requires_grad=True)
        teacher = torch.randn_like(z, requires_grad=True)
        mask = torch.randint(2, (2, 4, 4))
        mask[0, :2] = -1
        weights = (mask != -1).float()
        actual = r.weighted_kd_loss(z, mask, teacher, weights)
        expected = v.offline_kd_loss(z, mask, teacher.detach())
        self.assertTrue(torch.equal(actual, expected))
        weights[1, 0, 0] = 2.
        r.weighted_kd_loss(z, mask, teacher, weights).backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.abs().sum()), 0.)
        invalid = torch.full_like(mask, -1)
        zero = r.weighted_kd_loss(z, invalid, teacher, torch.zeros_like(weights))
        self.assertEqual(zero.item(), 0.)

    def test_same_live_weights_modes_buffers_rng_and_bypass(self):
        torch.manual_seed(10)
        model = tiny_model()
        compression_train_mode(model)
        image = torch.randn(2, 2, 5, 5)
        before = r.tensor_hash(model.state_dict())
        modes = [m.training for m in model.modules()]
        rng, py, np_state = torch.random.get_rng_state().clone(), random.getstate(), np.random.get_state()
        fp, quant = r.paired_logits(model, image)
        self.assertFalse(fp.requires_grad or quant.requires_grad)
        self.assertEqual(before, r.tensor_hash(model.state_dict()))
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(py, random.getstate())
        np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
        self.assertTrue(model[0].fake_quant_enabled)
        # Both activation and weight quantization must be absent in this FP path.
        plain = copy.deepcopy(model).eval()
        plain[0] = torch.nn.Conv2d(2, 3, 1)
        with torch.no_grad():
            plain[0].weight.copy_(model[0].weight)
            plain[0].bias.copy_(model[0].bias)
            expected = plain(image)
        torch.testing.assert_close(fp, expected, rtol=0, atol=0)
        with torch.no_grad():
            model[0].weight.add_(.21)
        new_fp, _ = r.paired_logits(model, image)
        self.assertFalse(torch.equal(new_fp, fp), 'FP shadow must follow current weights')
        # Scoring after the gradient forward must not invalidate saved QAT buffers.
        z = model(image)
        r.paired_logits(model, image)
        z.square().mean().backward()
        self.assertTrue(torch.isfinite(model[0].weight.grad).all())

    def test_private_random_selection_does_not_shift_input_stream(self):
        score = torch.arange(32.).reshape(2, 4, 4)
        valid = torch.ones_like(score, dtype=torch.bool)
        teacher = torch.zeros_like(score, dtype=torch.long)
        _, _, counts = r.select_priority(score, valid, teacher)
        before = torch.random.get_rng_state().clone()
        a, _, _ = r.select_priority(score, valid, teacher, counts=counts, random_seed=41)
        b, _, _ = r.select_priority(score, valid, teacher, counts=counts, random_seed=41)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_gate_does_not_multiply_reference_pixels_by_seed(self):
        metric = dict(equal_event_water_IoU=.8,
                      dark_land_false_water=dict(rate=.01, denominator=400),
                      bright_water_miss=dict(rate=None, denominator=0))
        results = [dict(seed=s, FP=copy.deepcopy(metric), Q0=copy.deepcopy(metric)) for s in r.SEEDS]
        for row in results:
            row['Q0']['dark_land_false_water']['rate'] = .05
        self.assertEqual(deficit_gate(results)['outcome'], 'NO_MATERIAL_SOURCE_DEFICIT')
        for row in results:
            row['Q0']['equal_event_water_IoU'] = .795
        self.assertEqual(deficit_gate(results)['outcome'], 'ADMITTED')


class AdapterTests(unittest.TestCase):
    def test_source_reader_joint_codes_and_threshold_use_source_only(self):
        from source11 import evaluate_source
        from models.compression_quant import strip_compression_qat
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = torch.randn(6, 8, 8)
            mask = np.tile(np.array([0, 1] * 4), (8, 1))
            mask[0, 0] = -1
            source = root / 's11val_00000.npz'
            np.savez(source, image=image.numpy(), mask=mask, cloud=np.array([]))
            target = root / 'teacher.npy'
            np.save(target, np.ones((8, 8), np.float32))
            windows = root / 'windows.json'
            r.save(windows, dict(windows=[dict(path=str(source), name='synthetic', event='source', dataset='s11val')]))
            entry = dict(id='synthetic', sha256='synthetic', quantized=True)
            inputs = dict(point='core', source_windows=dict(path=str(windows)), parents={'42': dict(FP_parent=entry)},
                          source_teacher_predictions=[dict(path=str(source), file=str(target), sha256=r.sha(target))])
            model = tiny_model(6).eval()
            with torch.no_grad():
                model[-1].bias[0] = 0
                model[-1].bias[1] = 80
            with patch.object(c, 'load', return_value=strip_compression_qat(model).eval()):
                result = evaluate_source(model, inputs, root, entry, seed=42, arm='D', step=0)
            self.assertEqual(result['windows'], 1)
            self.assertEqual(result['priority_diagnostics'], 'completed')
            histogram = np.load(result['source_histogram'])
            self.assertEqual(int(histogram.sum()), 63)
            self.assertTrue(np.isfinite(result['source_matched_tau']))
            self.assertEqual(result['source_clipped_tail_counts']['at_or_above32'], 63)
            self.assertEqual(result['source_matched_positive_fraction'], 1.)
            self.assertEqual(result['source_matching_fraction_error'], 0.)
            import csv
            with open(root / 'source_diagnostics/synthetic/priority_diagnostics.csv') as stream:
                all_rows = [row for row in csv.DictReader(stream) if row['stratum'] == 'all']
            self.assertEqual(sum(sum(json.loads(row['joint_counts'])) for row in all_rows), 63)

    def test_v10_eight_view_cache_alignment_and_arm_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'teacher_logits').mkdir()
            image = torch.arange(32.).reshape(2, 4, 4)
            # V10 normalization expects six bands for photometric jitter.
            image = image.repeat(3, 1, 1) / 20
            mask = torch.tensor([[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, -1]])
            path = root / 'tile.pt'
            torch.save(dict(image=image, mask=mask), path)
            teacher = torch.stack([torch.full((2, 4, 4), float(i)) for i in range(8)])
            cache_path = root / 'teacher_logits/tile.pt'
            torch.save(dict(logits=teacher), cache_path)
            r.save(root / 'teacher_cache_policy.json', dict(mode='float32_eight_geometric_views'))
            r.save(root / 'index.json', dict(files=['tile.pt'] * 2500, n_selected=2500))
            spec = dict(pseudo=str(root), index=str(root / 'index.json'))
            cache = dict(files=[dict(file=str(cache_path), sha256=r.sha(cache_path), input_sha256=r.sha(path))])
            with patch.object(c, 'ROOT', root), patch.object(c, 'DATA', root):
                a = next(iter(r._training_loader(spec, 42, cache)))
                b = next(iter(r._training_loader(spec, 42, cache)))
            self.assertEqual(r.tensor_hash({k: a[k] for k in ['image', 'mask', 'soft_logits']}),
                             r.tensor_hash({k: b[k] for k in ['image', 'mask', 'soft_logits']}))
            for observed, logits in zip(a['mask'], a['soft_logits']):
                codes = [4 * flip + rot for flip in [0, 1] for rot in range(4)
                         if torch.equal(observed, (mask.flip(-1) if flip else mask).rot90(rot, (-2, -1)))]
                self.assertEqual(len(codes), 1)
                self.assertTrue(torch.equal(logits, teacher[codes[0]]))

    def test_synthetic_50_update_profile_uses_frozen_trainer_and_checkpoint_rule(self):
        # Small synthetic arrays exercise real Trainer gradients/checkpointing;
        # no GDACS tiles, source labels, pretrained checkpoints or GPU are used.
        class Synthetic(torch.utils.data.Dataset):
            def __len__(self):
                return 2496

            def __getitem__(self, i):
                gen = torch.Generator().manual_seed(i)
                return dict(image=torch.randn(2, 4, 4, generator=gen),
                            mask=torch.randint(2, (4, 4), generator=gen),
                            soft_logits=torch.randn(2, 4, 4, generator=gen), sample_id=str(i))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            r.save(root / 'repair_inputs.json', dict(synthetic=True))
            r.save(root / 'spec.json', {})
            r.save(root / 'cache.json', dict(seconds=0.))
            parent = dict(id='synthetic', graph='R4', checkpoint='synthetic', sha256='synthetic')
            inputs = dict(point='core', graph='R4', precision='W4A4',
                          parents={'42': dict(entry=parent, FP_parent=parent)},
                          training_spec=dict(path=str(root / 'spec.json')),
                          cache_manifest=dict(path=str(root / 'cache.json')), lr=.0005,
                          schedule='cosine T_max10, 156 updates/epoch',
                          backend='synthetic CPU', augmentation='synthetic', stream='synthetic')
            r.save(root / 'repair/core/source_gate.json', dict(outcome='ADMITTED', inputs_sha256=r.sha(root / 'repair_inputs.json')))
            base = tiny_model()
            base[3] = torch.nn.Identity()
            loader = torch.utils.data.DataLoader(Synthetic(), batch_size=16, num_workers=0,
                                                 generator=torch.Generator().manual_seed(42))
            def metric(model, inputs, root, entry, **kw):
                return dict(equal_event_water_IoU=.5, pooled=dict(water_IoU=.5, mIoU=.5),
                            checkpoint_sha256=entry['sha256'], per_model_source_tile_seconds=.001)
            with patch.object(r, 'validate_inputs', return_value=inputs), patch.object(c, 'load', side_effect=lambda e: copy.deepcopy(base)), patch.object(r, '_training_loader', return_value=loader), patch('source11.evaluate_source', side_effect=metric):
                receipt = r.train(root, threads=2, arm='D', seed=42, profile=True)
            self.assertEqual(receipt['completed_updates'], 50)
            self.assertEqual([x['updates'] for x in receipt['checkpoints']], [0, 50])
            saved = torch.load(receipt['checkpoints'][-1]['checkpoint'], weights_only=False)
            self.assertEqual(saved['scheduler']['T_max'], 10)
            self.assertEqual(saved['scheduler']['last_epoch'], 0)
            self.assertEqual(saved['training_state']['global_step'], 50)
            self.assertTrue(saved['model']['0._compression_enabled'].item())
            trace = [json.loads(line) for line in (root / 'repair/core/profile_D_s42/D_trace.jsonl').read_text().splitlines()]
            self.assertEqual(len(trace), 50)
            self.assertIn('augmentation_sha256', trace[0]['batch_identity'])
            self.assertEqual(torch.get_num_threads(), 2)


if __name__ == '__main__':
    start, cpu = time.monotonic(), time.process_time()
    unittest.main(exit=False)
    print(json.dumps(dict(elapsed_seconds=time.monotonic() - start,
                          process_cpu_seconds=time.process_time() - cpu, threads=2)))
