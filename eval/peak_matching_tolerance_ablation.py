"""Re-evaluate an already-trained checkpoint (no retraining) with conJ's
hard-overlap match tolerance swept, to test whether conJ's small effect size
(see stratified_test_conj.py) is a match-density problem rather than a
"J-coupling just isn't that informative" problem.

collate128 hardcodes tol=0.1 Hz when building con_J (scripts/collate.py,
build_con_J_hard_overlap). This rebuilds con_J at each requested tolerance
from the same j_vals/j_mask collate128 already returns, feeds it through the
frozen trained model, and reports test accuracy alongside the peak match
rate (% of valid peaks with >=1 other peak matched within tolerance on some
J value) side by side. A flat accuracy curve against a fast-rising match
rate means match density isn't the bottleneck; accuracy tracking the match
rate rising would say it is. tol=0.0 (exact-value match only) is included
as the lower-bound "no slack" reference point.

Caveat: w_conJ and the attention weights were fit against tol=0.1's match
density during training. This tells you whether the signal is there for the
*existing* model to exploit, not the ceiling of retraining at the new
tolerance too -- if this shows a real effect, that's the retraining
experiment to run next.
"""
import argparse
import csv
import functools
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.dataset_nimis_parquet import NimisMixtureParquetDataset
from src.nimis_model import NMRPeakSlotAttn, slot_hungarian_loss
from scripts.collate import build_con_J_hard_overlap, collate128

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_test_df(path):
    """Mirrors load_multi() in scripts/train.py."""
    df = pd.read_parquet(path)
    return df[df["n_components"] >= 2].reset_index(drop=True)


def collate_with_tol(batch, K_max, noise_std, tol):
    out = collate128(batch, K_max=K_max, noise_std=noise_std)
    out["con_J"] = build_con_J_hard_overlap(out["j_vals"], out["j_mask"], tol=tol)
    return out


def peak_match_rate(con_j, mask):
    """Fraction of valid (masked) peaks that have >=1 other peak matched
    within tolerance on some J value (diagonal self-match excluded)."""
    B, P, _ = con_j.shape
    off_diag = con_j.clone()
    diag_idx = torch.arange(P)
    off_diag[:, diag_idx, diag_idx] = 0.0
    has_partner = off_diag.sum(dim=-1) > 0
    valid = mask.bool()
    return has_partner[valid].float().mean().item()


@torch.no_grad()
def evaluate_at_tolerance(model, loader, device, K, drop_con=False):
    model.eval()
    correct = total = 0
    matched_weighted = n_valid_total = 0
    for batch in loader:
        con = None if drop_con else batch["con"].to(device)
        mask = batch["mask"].to(device)
        component_mask = batch["component_mask"].to(device)
        mix_peaks = batch["mix_peaks"].to(device)
        mix_integrals = batch["mix_integrals"].to(device)
        targets = batch["targets"].to(device)
        conj = batch["con_J"].to(device)

        slots, slot_attn, _, _ = model(mix_peaks, mix_integrals, con, mask, component_mask, conj)
        _, mapping = slot_hungarian_loss(slot_attn, targets, mask, K=K)
        preds = slot_attn.argmax(dim=1)
        mapped_preds = torch.gather(mapping, 1, preds.clamp_min(0))
        mask_flat = mask.bool().view(-1)
        correct += (mapped_preds.view(-1)[mask_flat] == targets.view(-1)[mask_flat]).sum().item()
        total += mask_flat.sum().item()

        n_valid = mask.bool().sum().item()
        matched_weighted += peak_match_rate(conj, mask) * n_valid
        n_valid_total += n_valid

    return correct / total, matched_weighted / n_valid_total


def demo():
    """Self-check: peak_match_rate on a hand-built 1x3 con_J matrix."""
    con_j = torch.tensor([[
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]])
    mask = torch.tensor([[True, True, True]])
    rate = peak_match_rate(con_j, mask)
    assert abs(rate - 2 / 3) < 1e-6, rate
    print("peak_matching_tolerance_ablation self-check passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "outputs/NMRexp_conJ_Slot_attn_8_best.pt",
                    help="a checkpoint trained WITH conj active (e.g. the full con+conJ run)")
    p.add_argument("--test-parquet", type=Path, default=REPO_ROOT / "data/test_molecule_up25_orig_1prct.parquet")
    p.add_argument("--tolerances", type=str, default="0.0,0.1,0.2,0.3,0.5,1.0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--mlp-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim-head", type=int, default=64)
    p.add_argument("--slot-iters", type=int, default=5)
    p.add_argument("--drop-con", action="store_true",
                    help="pass for a checkpoint trained with --drop-con (e.g. run 10) -- must match how it was trained")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--csv", type=Path, default=None, help="also write results to this CSV path")
    p.add_argument("--seed", type=int, default=None,
                    help="fix the integral-noise RNG so results are exactly reproducible run to run")
    args = p.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = NMRPeakSlotAttn(
        dim=args.dim, mlp_dim=args.mlp_dim, depth=args.depth,
        heads=args.heads, dim_head=args.dim_head, K=args.K, slot_iters=args.slot_iters,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    test_df = load_test_df(args.test_parquet)
    dataset = NimisMixtureParquetDataset(test_df)

    rows = []
    print(f"{'tol (Hz)':>10}{'accuracy':>12}{'peaks matched':>16}")
    for tol in (float(t) for t in args.tolerances.split(",")):
        collate_fn = functools.partial(collate_with_tol, K_max=args.K, noise_std=args.noise_std, tol=tol)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        acc, match_rate = evaluate_at_tolerance(model, loader, device, args.K, drop_con=args.drop_con)
        print(f"{tol:>10.2f}{acc*100:>11.2f}%{match_rate*100:>15.2f}%")
        rows.append({"tol_hz": tol, "accuracy": acc, "peaks_matched_frac": match_rate})

    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tol_hz", "accuracy", "peaks_matched_frac"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        main()
