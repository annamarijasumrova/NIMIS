import argparse
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from torch.utils.data import DataLoader

from scripts.dataset_nimis_parquet import NimisMixtureParquetDataset, split_dataset
from scripts.collate import build_con_J_hard_overlap, collate128


def parse_args():
    p = argparse.ArgumentParser(description="Non-learned baselines for peak-to-component grouping.")
    p.add_argument("--dataset", type=Path, help="single parquet file, split into train/val/test")
    p.add_argument("--train-dataset", type=Path, help="parquet file with training mixtures (unused, accepted for CLI parity with the training script)")
    p.add_argument("--val-dataset", type=Path, help="parquet file with validation mixtures (unused, accepted for CLI parity with the training script)")
    p.add_argument("--test-dataset", type=Path, help="parquet file with test mixtures; use instead of --dataset to skip splitting")
    p.add_argument("--split-mode", choices=["random", "molecule_disjoint"], default="molecule_disjoint")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=0, help="must match the run being compared against")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--integral-noise-std", type=float, default=0.10, help="must match the run being compared against")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--j-tol", type=float, default=0.1,
                    help="J-coupling baseline: two peaks match if any of their J values are within this (Hz)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the random-assignment floor baseline")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return args
    if bool(args.dataset) == bool(args.test_dataset):
        p.error("pass exactly one of --dataset or --test-dataset")
    return args


def _best_match_accuracy(pred_labels, tgt_v, n_components):
    """Score predicted cluster labels against true labels via best-overlap
    (Hungarian) matching -- same matching principle the model uses to align
    predicted slots with true components before scoring accuracy.
    """
    n_pred = int(pred_labels.max()) + 1 if pred_labels.size else 0
    overlap = np.zeros((n_pred, n_components), dtype=np.int64)
    for p in range(n_pred):
        for t in range(n_components):
            overlap[p, t] = np.sum((pred_labels == p) & (tgt_v == t))
    row_ind, col_ind = linear_sum_assignment(-overlap)
    return int(overlap[row_ind, col_ind].sum())


def cluster_accuracy(adj, mask, targets, n_components):
    """Connected components of a peak-adjacency graph (e.g. shared-J-value graph)."""
    valid = mask.bool().numpy()
    n_peaks = int(valid.sum())
    if n_peaks == 0:
        return 0, 0

    adj_v = adj.numpy()[valid][:, valid]
    tgt_v = targets.numpy()[valid]
    _, pred_labels = connected_components(adj_v, directed=False)
    return _best_match_accuracy(pred_labels, tgt_v, n_components), n_peaks


def position_only_accuracy(peaks, mask, targets, n_components):
    """Groups peaks by chemical-shift proximity alone (1-D hierarchical clustering,
    cut to n_components clusters) -- no J-coupling, no integrals.
    """
    valid = mask.bool().numpy()
    n_peaks = int(valid.sum())
    if n_peaks == 0:
        return 0, 0

    ppm = peaks.numpy()[valid].reshape(-1, 1)
    tgt_v = targets.numpy()[valid]
    if n_peaks <= n_components:
        pred_labels = np.arange(n_peaks)
    else:
        Z = linkage(ppm, method="single")
        pred_labels = fcluster(Z, t=n_components, criterion="maxclust") - 1
    return _best_match_accuracy(pred_labels, tgt_v, n_components), n_peaks


def random_baseline_accuracy(mask, targets, n_components, rng):
    valid = mask.bool().numpy()
    n_peaks = int(valid.sum())
    if n_peaks == 0:
        return 0, 0

    tgt_v = targets.numpy()[valid]
    pred_labels = rng.integers(0, n_components, size=n_peaks)
    return _best_match_accuracy(pred_labels, tgt_v, n_components), n_peaks


H_MAX = 9
H_PENALTY = 0.05


def _scale_cost(integrals, scale):
    """Per-peak cost of explaining `integrals` as integer multiples of one component's per-H scale.

    Within a component every integral is (number of H) * (component scale), so
    integral/scale should be a small integer. The H_PENALTY term prefers the
    explanation with the smallest proton counts, which stops scale/2, scale/3, ...
    from always fitting equally well.
    """
    ratio = integrals / scale
    h = np.clip(np.round(ratio), 1, H_MAX)
    return np.abs(ratio - h) / h + H_PENALTY * h


def _fit_scale(integrals):
    candidates = (integrals[:, None] / np.arange(1, H_MAX + 1)).ravel()
    cost = np.array([_scale_cost(integrals, c).mean() for c in candidates])
    return float(candidates[np.argmin(cost)])


def j_integral_labels(adj, integrals, n_components, use_j=True, leftover="integral", rng=None, n_refine=3):
    """Cluster peaks into exactly n_components groups from J-couplings and integrals.

    1. J seeds (use_j): connected components of the J-match graph with >=2 peaks are
       probable same-component groups; the n_components largest become seed clusters
       (their peaks are locked) and each gets a per-H integral scale from _fit_scale.
    2. If fewer seeds than components, extra clusters start from the unassigned peak
       worst explained by every existing scale (farthest-first).
    3. Every remaining peak (e.g. singlets with no J) is assigned to the cluster whose
       scale explains its integral best (leftover="integral"), or to a random cluster
       (leftover="random", the J-only control); scales are then re-fit from the members
       and the unlocked peaks reassigned, n_refine times.
    """
    n = len(integrals)
    labels = np.full(n, -1)
    locked = np.zeros(n, dtype=bool)
    scales = []

    if use_j:
        _, comp = connected_components(adj, directed=False)
        sizes = np.bincount(comp)
        seed_ids = [c for c in np.argsort(-sizes) if sizes[c] >= 2][:n_components]
        for k, c in enumerate(seed_ids):
            members = comp == c
            labels[members] = k
            locked[members] = True
            scales.append(_fit_scale(integrals[members]))

    while len(scales) < n_components:
        free = np.flatnonzero(labels < 0)
        if free.size == 0:
            scales.append(float(integrals.mean()))
            continue
        if scales:
            cost = np.min([_scale_cost(integrals[free], s) for s in scales], axis=0)
            pick = free[np.argmax(cost)]
        else:
            pick = free[np.argmax(integrals[free])]
        labels[pick] = len(scales)
        scales.append(float(integrals[pick]))

    free = labels < 0
    if leftover == "random":
        labels[free] = rng.integers(0, n_components, size=int(free.sum()))
        return labels

    for _ in range(n_refine + 1):
        cost = np.stack([_scale_cost(integrals, s) for s in scales], axis=1)
        labels = np.where(locked, labels, cost.argmin(axis=1))
        for k in range(n_components):
            members = labels == k
            if members.any():
                scales[k] = _fit_scale(integrals[members])
    return labels


def j_integral_accuracy(adj, integrals, mask, targets, n_components, **kwargs):
    valid = mask.bool().numpy()
    n_peaks = int(valid.sum())
    if n_peaks == 0:
        return 0, 0
    pred_labels = j_integral_labels(adj.numpy()[valid][:, valid], integrals.numpy()[valid].astype(np.float64),
                                    n_components, **kwargs)
    return _best_match_accuracy(pred_labels, targets.numpy()[valid], n_components), n_peaks


def demo():
    mask = torch.tensor([True, True, True, True])
    targets = torch.tensor([0, 0, 1, 1])
    adj = torch.tensor([
        [1., 1., 0., 0.],
        [1., 1., 0., 0.],
        [0., 0., 1., 1.],
        [0., 0., 1., 1.],
    ])
    assert cluster_accuracy(adj, mask, targets, n_components=2) == (4, 4)

    peaks = torch.tensor([1.0, 1.05, 5.0, 5.1])
    assert position_only_accuracy(peaks, mask, targets, n_components=2) == (4, 4)

    rng = np.random.default_rng(0)
    correct, total = random_baseline_accuracy(mask, targets, n_components=2, rng=rng)
    assert total == 4 and 0 <= correct <= 4
    integrals = torch.tensor([0.8, 2.4, 0.2, 0.4, 0.8, 0.6])
    targets = torch.tensor([0, 0, 1, 1, 0, 1])
    mask = torch.ones(6, dtype=torch.bool)
    adj = torch.eye(6)
    adj[0, 1] = adj[1, 0] = 1.0
    adj[2, 3] = adj[3, 2] = 1.0
    assert j_integral_accuracy(adj, integrals, mask, targets, 2) == (6, 6)
    assert j_integral_accuracy(adj, integrals, mask, targets, 2, use_j=False)[1] == 6
    print("baselines self-check passed")


def main():
    args = parse_args()
    if args.self_test:
        demo()
        return

    if args.test_dataset:
        test_df = pd.read_parquet(args.test_dataset)
        test_df = test_df[test_df["n_components"] >= 2].reset_index(drop=True)
    else:
        df = pd.read_parquet(args.dataset)
        df = df[df["n_components"] >= 2].reset_index(drop=True)
        _, _, test_df = split_dataset(df, args.split_mode, args.val_frac, args.test_frac, args.split_seed)
    print(f"Test rows: {len(test_df):,}")

    collate_fn = functools.partial(collate128, K_max=args.K, noise_std=args.integral_noise_std)
    loader = DataLoader(NimisMixtureParquetDataset(test_df), batch_size=args.batch_size,
                         shuffle=False, collate_fn=collate_fn)

    rng = np.random.default_rng(args.seed)
    totals = {"j_coupling_only": [0, 0], "j_coupling_k_clusters": [0, 0], "integral_only": [0, 0],
              "j_plus_integral": [0, 0], "position_only": [0, 0], "random_floor": [0, 0]}

    for batch in loader:
        mask, targets, peaks = batch["mask"], batch["targets"], batch["mix_peaks"]
        con_j = build_con_J_hard_overlap(batch["j_vals"], batch["j_mask"], tol=args.j_tol)

        for b in range(mask.shape[0]):
            n_comp = int(batch["n_components"][b].item())

            c, n = cluster_accuracy(con_j[b], mask[b], targets[b], n_comp)
            totals["j_coupling_only"][0] += c
            totals["j_coupling_only"][1] += n

            integ = batch["mix_integrals"][b]
            for name, kwargs in (("j_coupling_k_clusters", {"leftover": "random", "rng": rng}),
                                 ("integral_only", {"use_j": False}),
                                 ("j_plus_integral", {})):
                c, n = j_integral_accuracy(con_j[b], integ, mask[b], targets[b], n_comp, **kwargs)
                totals[name][0] += c
                totals[name][1] += n

            c, n = position_only_accuracy(peaks[b], mask[b], targets[b], n_comp)
            totals["position_only"][0] += c
            totals["position_only"][1] += n

            c, n = random_baseline_accuracy(mask[b], targets[b], n_comp, rng)
            totals["random_floor"][0] += c
            totals["random_floor"][1] += n

    print("\nBaseline peak-to-component accuracy on test set (compare against the model's own Test: Accuracy= line):")
    for name, (correct, total) in totals.items():
        acc = correct / total if total else 0.0
        print(f"  {name:22s}: {acc*100:.2f}%  ({correct}/{total} peaks)")


if __name__ == "__main__":
    main()
