#!/usr/bin/env python
"""Pure-PyTorch Density-aware Chamfer Distance (DCD) baseline metric.

Faithful port of ``calc_dcd`` from the official implementation of Wu et al.,
"Density-aware Chamfer Distance as a Comprehensive Metric for Point Cloud
Completion" (NeurIPS 2021), github.com/wutong16/Density_aware_Chamfer_Distance
(utils/model_utils.py). The compiled chamfer3D CUDA kernel is replaced by
``torch.cdist``; the math is unchanged: squared nearest-neighbour distances,
``exp(-alpha * d^2)`` matching, multiplicity-based density weights, and the
official size-ratio corrections. Defaults alpha=1000, n_lambda=1, non_reg=False.
"""

import torch


def _nearest_squared(source, target):
    """Return (squared distance, index) of the nearest target point per source point.

    source: (B, Ns, 3); target: (B, Nt, 3). Output shapes (B, Ns) and (B, Ns).
    """
    squared = torch.cdist(source, target, p=2) ** 2
    distance, index = squared.min(dim=2)
    return distance, index


def chamfer_distance(observed, ideal):
    """Symmetric Chamfer Distance. Returns (cd_p sqrt-form, cd_t squared-form), each (B,)."""
    ideal_to_obs, _ = _nearest_squared(ideal, observed)
    obs_to_ideal, _ = _nearest_squared(observed, ideal)
    cd_t = ideal_to_obs.mean(dim=1) + obs_to_ideal.mean(dim=1)
    cd_p = (ideal_to_obs.sqrt().mean(dim=1) + obs_to_ideal.sqrt().mean(dim=1)) / 2
    return cd_p, cd_t


def normalise_to_reference(observed, reference, minimum_radius=1e-6):
    """Move and scale both sets by the reference's centroid and RMS radius.

    Normalising by the ideal surface (not the observed returns) makes the metric
    independent of object size, sensor units, and of how many returns happened
    to be observed, which keeps the baseline portable across datasets.
    """
    centroid = reference.mean(dim=1, keepdim=True)
    radius = ((reference - centroid) ** 2).sum(dim=-1).mean(dim=1).sqrt().clamp(min=minimum_radius)
    scale = radius.view(-1, 1, 1)
    return (observed - centroid) / scale, (reference - centroid) / scale


def calc_dcd(observed, ideal, alpha=1000.0, n_lambda=1, non_reg=False, return_raw=False):
    """Official DCD. observed: (B, Nx, 3); ideal: (B, Ngt, 3). Returns (dcd, cd_p, cd_t).

    Direction 1 (ideal->observed) penalises ideal-surface samples whose nearest
    return is far or crowds into one observed point (coverage holes); direction
    2 (observed->ideal) penalises returns far from any ideal surface sample.
    """
    observed = observed.float()
    ideal = ideal.float()
    batch_size, n_x, _ = observed.shape
    _, n_gt, _ = ideal.shape
    if observed.shape[0] != ideal.shape[0]:
        raise ValueError("observed and ideal must share the batch dimension")

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    dist1, idx1 = _nearest_squared(ideal, observed)   # per ideal sample: nearest observed
    dist2, idx2 = _nearest_squared(observed, ideal)   # per observed point: nearest ideal
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    loss1, loss2 = [], []
    for b in range(batch_size):
        count1 = torch.bincount(idx1[b], minlength=n_x)
        weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append((-exp_dist1[b] * weight1 + 1.0).mean())

        count2 = torch.bincount(idx2[b], minlength=n_gt)
        weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append((-exp_dist2[b] * weight2 + 1.0).mean())

    dcd = (torch.stack(loss1) + torch.stack(loss2)) / 2
    cd_p, cd_t = chamfer_distance(observed, ideal)
    if return_raw:
        return dcd, cd_p, cd_t, dist1, dist2, idx1, idx2
    return dcd, cd_p, cd_t
