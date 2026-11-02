import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BINS = [1, 5, 10, 15, 20, np.inf]
LABELS = ["1-5%", "5-10%", "10-15%", "15-20%", ">20%"]


def concentration_table(df: pd.DataFrame, mixture_sizes=(2, 3, 4, 5)) -> pd.DataFrame:
    """Count how many components fall in each concentration bucket, per mixture size.

    df must have "n_components" and "scales" (per-component fraction, summing to ~1 per row).
    """
    rows = {}
    below_1pct = 0
    for n in mixture_sizes:
        sub = df[df["n_components"] == n]
        pct = np.concatenate([np.asarray(s, dtype=float) * 100 for s in sub["scales"]]) if len(sub) else np.array([])
        below_1pct += int((pct < 1).sum())
        binned = pd.cut(pct, bins=BINS, labels=LABELS, right=False)
        rows[n] = binned.value_counts().reindex(LABELS, fill_value=0)

    table = pd.DataFrame(rows).T
    table.index.name = "n_components"
    if below_1pct:
        print(f"Note: {below_1pct} components had concentration <1% and are not in any bucket.")
    return table


def demo():
    df = pd.DataFrame({
        "n_components": [2, 2, 3, 3, 3],
        "scales": [[0.5, 0.5], [0.03, 0.97], [0.1, 0.3, 0.6], [0.02, 0.18, 0.80], [0.25, 0.25, 0.50]],
    })
    table = concentration_table(df, mixture_sizes=(2, 3))
    assert table.loc[2, "1-5%"] == 1
    assert table.loc[2, ">20%"] == 3
    assert table.loc[3, "10-15%"] == 1
    assert table.loc[3, "15-20%"] == 1
    assert table.loc[3, ">20%"] == 6
    print("dataset_analysis self-check passed")


def parse_args():
    p = argparse.ArgumentParser(description="Bucket per-component mixture concentrations by mixture size.")
    p.add_argument("dataset", type=Path, help="Path to a mixtures parquet dataset")
    p.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        demo()
        return

    df = pd.read_parquet(args.dataset)
    table = concentration_table(df, mixture_sizes=args.sizes)
    print(table.to_string())


if __name__ == "__main__":
    main()
