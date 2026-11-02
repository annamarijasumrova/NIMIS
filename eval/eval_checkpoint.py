"""Re-evaluate an already-trained checkpoint on a different test parquet
(no retraining), writing an assignments file in the same format
scripts/train.py produces -- reuses its evaluate_loader so
downstream tools (stratified_test_conj.py, concentration_accuracy.py,
helpers/violin_parser.py) work unchanged.
"""
import argparse
import functools
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.dataset_nimis_parquet import NimisMixtureParquetDataset
from src.nimis_model import NMRPeakSlotAttn
from scripts.collate import collate128
from scripts.train import evaluate_loader
from helpers.violin_parser import plot_confidence_violin

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_test_df(path):
    df = pd.read_parquet(path)
    if "smiles" not in df.columns and "smiles_list" in df.columns:
        df = df.rename(columns={"smiles_list": "smiles"})
    return df[df["n_components"] >= 2].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test-parquet", type=Path, required=True)
    p.add_argument("--assignments-out", type=Path, required=True)
    p.add_argument("--violin-out", type=Path, default=None, help="also save a confidence violin plot here")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--integral-noise-std", type=float, default=0.1)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--mlp-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim-head", type=int, default=64)
    p.add_argument("--slot-iters", type=int, default=5)
    p.add_argument("--per-head-bias", action="store_true")
    p.add_argument("--drop-con", action="store_true", help="must match how the checkpoint was trained")
    p.add_argument("--drop-conj", action="store_true", help="must match how the checkpoint was trained")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model = NMRPeakSlotAttn(
        dim=args.dim, mlp_dim=args.mlp_dim, depth=args.depth,
        heads=args.heads, dim_head=args.dim_head, K=args.K, slot_iters=args.slot_iters,
        per_head_bias=args.per_head_bias,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    test_df = load_test_df(args.test_parquet)
    dataset = NimisMixtureParquetDataset(test_df)
    collate_fn = functools.partial(collate128, K_max=args.K, noise_std=args.integral_noise_std)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    loss, acc, conf_correct, conf_wrong = evaluate_loader(
        model, loader, device, args.K,
        save_assignments_path=args.assignments_out,
        drop_con=args.drop_con, drop_conj=args.drop_conj,
    )
    print(f"test_loss={loss:.4f} test_acc={acc*100:.2f}%")
    print(f"Assignments -> {args.assignments_out}")

    if args.violin_out is not None:
        plot_confidence_violin(conf_correct, conf_wrong, title=str(args.test_parquet), save_path=args.violin_out)
        print(f"Violin plot -> {args.violin_out}")


if __name__ == "__main__":
    main()
