from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NimisMixtureParquetDataset(Dataset):
    """One row = one pre-built peak-aligned mixture (see new_dataset/all_mix_for_original_nimis.py)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "n_components": int(row["n_components"]),
            "mix_peaks": torch.as_tensor(list(row["mix_peak_positions"]), dtype=torch.float32),
            "mix_integrals": torch.as_tensor(list(row["mix_integrals"]), dtype=torch.float32),
            "target": torch.as_tensor(list(row["target"]), dtype=torch.long),
            "j_vals": torch.as_tensor(np.array(list(row["j_vals"]), dtype=np.float32)),
            "j_mask": torch.as_tensor(np.array(list(row["j_mask"]), dtype=bool)),
            "smiles": list(row["smiles"]),
            "scales": list(row["scales"]),
            "solvent": row["NMR_solvent"],
            "frequency": row["NMR_frequency"],
        }


def _assign_molecule_splits(df, val_frac, test_frac, seed):
    """Greedy, size-first molecule->split assignment for molecule-disjoint splitting.

    Assigning molecules to splits independently of each other drops k-component
    mixtures at a rate that grows like p**k (p = split's molecule fraction), so
    larger mixtures end up almost entirely in train no matter which molecules are
    picked. Processing mixtures largest-first and claiming each one's full molecule
    set atomically -- while its molecules are still unclaimed -- keeps far more
    large mixtures intact, and routes each newly-claimed mixture toward whichever
    split is furthest behind its target share for that size.
    """
    frac = {"train": 1.0 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    splits = list(frac.keys())
    size_totals = df["n_components"].value_counts().to_dict()
    target = {s: {size: total * f for size, total in size_totals.items()} for s, f in frac.items()}
    running = {s: {size: 0.0 for size in size_totals} for s in frac}

    rng = np.random.default_rng(seed)
    smiles_list = df["smiles"].tolist()
    sizes = df["n_components"].to_numpy()
    row_order = rng.permutation(len(df))
    row_order = sorted(row_order, key=lambda i: -sizes[i])

    split_of = {}
    for i in row_order:
        row_smiles = smiles_list[i]
        size = sizes[i]
        assigned = {split_of[m] for m in row_smiles if m in split_of}
        if len(assigned) > 1:
            continue

        if assigned:
            chosen = next(iter(assigned))
        else:
            chosen = max(
                splits,
                key=lambda s: (target[s][size] - running[s][size]) / target[s][size] if target[s][size] > 0 else -1.0,
            )

        for m in row_smiles:
            split_of.setdefault(m, chosen)
        running[chosen][size] += 1

    unresolved = {m for row in smiles_list for m in row} - split_of.keys()
    for m in unresolved:
        split_of[m] = rng.choice(splits, p=[frac[s] for s in splits])

    return split_of


def split_dataset(df: pd.DataFrame, mode: str, val_frac: float, test_frac: float, seed: int):
    """Split a mixtures dataframe into (train_df, val_df, test_df).

    mode="random": split rows uniformly at random.
    mode="molecule_disjoint": no SMILES appears in more than one split. Mixtures whose
    components straddle two splits are dropped rather than leaking a molecule across splits.
    """
    if mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(df))
        n_test = int(len(df) * test_frac)
        n_val = int(len(df) * val_frac)
        test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
        return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]

    if mode != "molecule_disjoint":
        raise ValueError(f"Unknown split mode: {mode!r}, expected 'random' or 'molecule_disjoint'")

    split_of = _assign_molecule_splits(df, val_frac, test_frac, seed)

    def row_split(row_smiles):
        splits = {split_of[s] for s in row_smiles}
        return splits.pop() if len(splits) == 1 else None

    row_splits = df["smiles"].map(row_split)
    n_dropped = int(row_splits.isna().sum())
    if n_dropped:
        print(f"molecule_disjoint split: dropped {n_dropped:,}/{len(df):,} mixtures spanning >1 split")

    return (
        df[row_splits == "train"],
        df[row_splits == "val"],
        df[row_splits == "test"],
    )


def demo():
    rng = np.random.default_rng(0)
    mols = [f"M{i}" for i in range(20)]
    rows = []
    for _ in range(60):
        row_smiles = list(rng.choice(mols, size=int(rng.integers(1, 6)), replace=False))
        rows.append({"smiles": row_smiles, "n_components": len(row_smiles)})
    df = pd.DataFrame(rows)

    train_df, val_df, test_df = split_dataset(df, "molecule_disjoint", 0.2, 0.2, seed=0)
    mol_sets = [{s for row in d["smiles"] for s in row} for d in (train_df, val_df, test_df)]
    assert not (mol_sets[0] & mol_sets[1]) and not (mol_sets[0] & mol_sets[2]) and not (mol_sets[1] & mol_sets[2])

    tr2, va2, te2 = split_dataset(df, "random", 0.2, 0.2, seed=0)
    assert len(tr2) + len(va2) + len(te2) == len(df)

    print("dataset_nimis_parquet self-check passed")


if __name__ == "__main__":
    demo()
