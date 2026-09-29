#!/usr/bin/env python
"""Property tests for the pure-PyTorch DCD port (CPU only, no CUDA needed)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dcd_metric import calc_dcd, chamfer_distance, normalise_to_reference


def grid_points(n=20):
    axis = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(axis, axis)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(n * n)], axis=1).astype(np.float32)


def dcd_value(observed, ideal, alpha=1.0, normalize=False):
    obs = torch.as_tensor(observed, dtype=torch.float32).unsqueeze(0)
    ref = torch.as_tensor(ideal, dtype=torch.float32).unsqueeze(0)
    if normalize:
        obs, ref = normalise_to_reference(obs, ref)
    with torch.no_grad():
        dcd, _cd_p, _cd_t = calc_dcd(obs, ref, alpha=alpha)
    return float(dcd[0])


class TestDCD(unittest.TestCase):
    def test_matches_hand_computed_case(self):
        """Two-point observed set vs one ideal point, alpha=1, no normalisation:
        dcd = (loss1 + loss2) / 2 with loss1 = 1 - frac_21 = 0.5 and
        loss2 = (1 - 1 + (1 - e^-1)) / 2, giving 0.25 + (1 - 1/e)/4."""
        observed = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        ideal = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        expected = 0.25 + (1.0 - np.exp(-1.0)) / 4
        self.assertAlmostEqual(dcd_value(observed, ideal, alpha=1.0, normalize=False), expected, places=4)

    def test_identical_point_sets_are_zero(self):
        points = grid_points()
        self.assertLess(dcd_value(points, points), 1e-4)

    def test_structured_gap_scores_above_equal_count_random_thinning(self):
        """A contiguous removed half-plane is a coverage hole; the same number of
        points removed uniformly only thins density. DCD must score the hole higher."""
        ideal = grid_points()
        scores = {"structured": [], "random": []}
        for seed in range(5):
            generator = np.random.default_rng(seed)
            removal = generator.permutation(len(ideal))[: len(ideal) // 2]
            scores["random"].append(dcd_value(np.delete(ideal, removal, axis=0), ideal))
            scores["structured"].append(dcd_value(ideal[ideal[:, 0] <= 0.5], ideal))
        self.assertGreater(np.mean(scores["structured"]), np.mean(scores["random"]))

    def test_monotone_in_contiguous_removal(self):
        ideal = grid_points()
        values = [dcd_value(ideal[ideal[:, 0] <= cut], ideal) for cut in (1.01, 0.8, 0.6, 0.4)]
        self.assertTrue(all(b > a for a, b in zip(values, values[1:])))

    def test_normalisation_gives_unit_rms_reference(self):
        points = torch.as_tensor(grid_points(6), dtype=torch.float32).unsqueeze(0) * 3.7
        obs, ref = normalise_to_reference(points * 0.5, points)
        radius = ((ref - ref.mean(dim=1, keepdim=True)) ** 2).sum(-1).mean(-1).sqrt()
        self.assertAlmostEqual(float(radius[0]), 1.0, places=5)
        # Observed set is at half the scale of the reference: pairwise distances must shrink by 0.5.
        self.assertAlmostEqual(float((obs[0, 1] - obs[0, 0]).norm() / (ref[0, 1] - ref[0, 0]).norm()), 0.5, places=5)

    def test_chamfer_distance_identical_sets(self):
        points = torch.as_tensor(grid_points(8), dtype=torch.float32).unsqueeze(0)
        cd_p, cd_t = chamfer_distance(points, points)
        self.assertLess(float(cd_t[0]), 1e-8)
        self.assertLess(float(cd_p[0]), 1e-4)


if __name__ == "__main__":
    unittest.main()
