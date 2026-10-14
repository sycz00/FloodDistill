"""Numerical and provenance tests for the independent bounded conditioner."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import conditioner11 as c


class ConditionerTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1729)
        self.h = self.rng.normal(size=(40, 4))
        self.phi = np.c_[self.h, np.ones(40)]
        self.z = self.rng.normal(size=(40, 12))
        self.m0 = self.rng.normal(size=40)
        self.q = self.rng.uniform(.05, .95, size=40)
        self.weights = np.ones(40) / 40

    def test_gradient_hessian_finite_difference(self):
        X = c.design_matrix(self.phi, self.z, "C2")
        theta = self.rng.normal(scale=.2, size=X.shape[1])
        value, g, H = c.objective_gradient_hessian(theta, X, self.m0, self.q, self.weights, .1)
        eps = 1e-5
        for j in range(len(theta)):
            step = np.zeros_like(theta); step[j] = eps
            vp, gp, _ = c.objective_gradient_hessian(theta + step, X, self.m0, self.q, self.weights, .1)
            vm, gm, _ = c.objective_gradient_hessian(theta - step, X, self.m0, self.q, self.weights, .1)
            self.assertAlmostEqual(g[j], (vp-vm)/(2*eps), places=8)
            np.testing.assert_allclose(H[:, j], (gp-gm)/(2*eps), atol=1e-9, rtol=1e-7)
        self.assertGreater(np.linalg.eigvalsh(H).min(), 0)

    def test_objective_decrease_and_iteration_cap(self):
        X = c.design_matrix(self.phi, self.z, "C3")
        _, result = c.fit_newton(X, self.m0, self.q, self.weights, .1)
        self.assertLessEqual(result['iterations'], 10)
        self.assertTrue(result['converged'], result)
        for row in result['history']:
            self.assertLess(row['objective_after'], row['objective_before'])
        with self.assertRaises(ValueError):
            c.fit_newton(X, self.m0, self.q, self.weights, .1, max_iterations=11)

    def test_zero_H_and_bias_only_controls(self):
        a = self.rng.normal(size=5)
        full = np.r_[a, np.zeros(60)]
        np.testing.assert_allclose(c.design_matrix(self.phi, self.z, 'C3') @ full, self.phi @ a)
        beta = self.rng.normal(size=12)
        H = np.zeros((5, 12)); H[-1] = beta
        full = np.r_[a, H.ravel()]
        np.testing.assert_allclose(c.design_matrix(self.phi, self.z, 'C3') @ full,
                                   c.design_matrix(self.phi, self.z, 'C2') @ np.r_[a, beta])
        for form in c.FORMS:
            X = c.design_matrix(self.phi, self.z, form)
            np.testing.assert_array_equal(self.m0 + X @ np.zeros(X.shape[1]), self.m0)

    def test_explicit_native_and_folded_margin_parity(self):
        mu = self.rng.normal(size=4); scale = self.rng.uniform(.1, 2, size=4)
        phi = np.c_[(self.h-mu)/scale, np.ones(40)]
        z = self.z[0]
        weight = self.rng.normal(size=(2, 4)); bias = self.rng.normal(size=2)
        base = self.h @ (weight[1]-weight[0]) + bias[1]-bias[0]
        for form in c.FORMS:
            X = c.design_matrix(phi, np.tile(z, (40, 1)), form)
            theta = self.rng.normal(size=X.shape[1])
            dw, db = c.native_correction(theta, z, mu, scale, form)
            w, b = c.fold_two_logit_head(weight, bias, dw, db)
            folded = self.h @ (w[1]-w[0]) + b[1]-b[0]
            np.testing.assert_allclose(folded, base + X @ theta, rtol=1e-12, atol=1e-12)

    def test_input_only_descriptor_mask_and_sampling(self):
        image = self.rng.normal(size=(6, 4, 4))
        valid = np.ones((4, 4), bool); valid[0] = False
        first = c.input_descriptor(image, valid)
        image[:, 0] = np.nan
        np.testing.assert_array_equal(c.input_descriptor(image, valid), first)
        coords = c.sample_coordinates('tile', valid, limit=6)
        np.testing.assert_array_equal(coords, c.sample_coordinates('tile', valid, limit=6))
        self.assertTrue(valid[coords[:, 0], coords[:, 1]].all())
        self.assertEqual(len(coords), 6)
        with self.assertRaises(ValueError): c.input_descriptor(image, np.ones((4, 4), bool))
        with self.assertRaises(ValueError): c.input_descriptor(image, valid.astype(int))

    def test_equal_tile_statistics_train_guard(self):
        features = np.array([[0., 1.], [0., 1.], [3., 1.]])
        descriptors = np.zeros((2, 12))
        rows = ['a', 'a', 'b']
        statistics = c.train_statistics(features, descriptors, rows, ['a', 'b'], split='GDACS_TRAIN')
        np.testing.assert_allclose(statistics['feature_mean'], [1.5, 1])
        np.testing.assert_allclose(statistics['feature_scale'], [1.5, 1])
        np.testing.assert_allclose(c.equal_tile_weights(rows), [.25, .25, .5])
        for split in ['S11_VALIDATION', 'S11_TEST', 'WF_SEALED']:
            with self.assertRaises(ValueError): c.train_statistics(features, descriptors, rows, ['a', 'b'], split=split)

    def test_shuffling_selection_and_parameter_count(self):
        z = np.arange(60).reshape(5, 12)
        shuffled, permutation = c.shuffled_descriptors(z, ['a', 'b', 'c', 'd', 'e'])
        self.assertTrue(np.all(permutation != np.arange(5)))
        self.assertEqual(sorted(permutation), list(range(5)))
        np.testing.assert_array_equal(shuffled, z[permutation])
        scores = {lam: {seed: .8 for seed in c.SEEDS} for lam in c.LAMBDAS}
        self.assertEqual(c.source_select_regularization(scores)[0], 10)
        scores[.1][42] = .9
        self.assertEqual(c.source_select_regularization(scores)[0], .1)
        self.assertEqual(c.cost_inventory()['C3_C4_coefficients'], 65)
        with self.assertRaises(ValueError): c.design_matrix(np.ones((2, 40)), np.ones((2, 12)), 'C3')

    def test_bad_fit_rows_and_unavailable_parents(self):
        X = self.phi.copy(); X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            c.objective_gradient_hessian(np.zeros(5), X, self.m0, self.q, self.weights, .1)
        with tempfile.TemporaryDirectory() as directory:
            result = c.dependency_status(directory)
        self.assertEqual(result['status'], 'WAITING_DEPENDENCY')
        self.assertEqual(len(result['missing']), 5)

    def test_real_micro_frozen_bn_and_folded_output(self):
        import sys
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / 'capacity_v4_20261008/code/v5'))
        import torch
        import rungs
        torch.set_num_threads(2)
        model = rungs.ContextConv(widths=(4, 4, 6, 8), dec=4).eval().requires_grad_(False)
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        captured = []
        hook = model.cls.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        x = torch.randn(1, 6, 32, 32)
        with torch.no_grad():
            y = model(x)
        hook.remove()
        h = captured[0]
        dw = torch.tensor([.1, -.2, .3, -.1]); db = .05
        native_delta = torch.einsum('bdhw,d->bhw', h, dw) + db
        delta = torch.nn.functional.interpolate(native_delta[:, None], size=(32, 32), mode='bilinear', align_corners=False)[:, 0]
        with torch.no_grad():
            native_logits = torch.nn.functional.conv2d(h, model.cls.weight + torch.stack([-dw, dw])[:, :, None, None]/2,
                                                    model.cls.bias + torch.tensor([-db, db])/2)
            folded = torch.nn.functional.interpolate(native_logits, size=(32, 32), mode='bilinear', align_corners=False)
        torch.testing.assert_close(folded[:, 1]-folded[:, 0], y[:, 1]-y[:, 0]+delta, atol=1e-6, rtol=1e-5)
        for name, tensor in model.state_dict().items(): torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))


if __name__ == '__main__':
    started, cpu = time.monotonic(), time.process_time()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConditionerTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
                   failures=len(result.failures), errors=len(result.errors),
                   wall_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
                   allocated_threads=2, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   scientific_training=False, external_labels_opened=False)
    print(json.dumps(receipt))
    if os.environ.get('V11_CONDITIONER_TEST_RECEIPT'):
        Path(os.environ['V11_CONDITIONER_TEST_RECEIPT']).write_text(json.dumps(receipt, indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
