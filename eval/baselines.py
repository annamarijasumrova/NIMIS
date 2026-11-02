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
    totals = {"j_coupling_only": [0, 0], "position_only": [0, 0], "random_floor": [0, 0]}

    for batch in loader:
        mask, targets, peaks = batch["mask"], batch["targets"], batch["mix_peaks"]
        con_j = build_con_J_hard_overlap(batch["j_vals"], batch["j_mask"], tol=args.j_tol)

        for b in range(mask.shape[0]):
            n_comp = int(batch["n_components"][b].item())

            c, n = cluster_accuracy(con_j[b], mask[b], targets[b], n_comp)
            totals["j_coupling_only"][0] += c
            totals["j_coupling_only"][1] += n

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
