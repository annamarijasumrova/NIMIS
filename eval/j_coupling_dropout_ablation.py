"""Re-evaluate an already-trained checkpoint (no retraining) with a fraction
of con_J's J-coupling matches zeroed out, to simulate missing/undetected J
couplings at inference time.

con_J (build_con_J_hard_overlap in scripts/collate.py) is a (B, P, P) 0/1 matrix:
con_J[b,i,j] = 1 iff peaks i and j share a J value within tol. This zeroes a
random --drop-frac of those off-diagonal 1-entries (symmetrically, so the
matrix stays symmetric) before feeding con_J into the frozen trained model,
and reports test accuracy vs. drop_frac. drop_frac=0.0 is the "full J info"
reference point; drop_frac=1.0 removes all J-coupling signal.
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
from scripts.collate import collate128
from eval.peak_matching_tolerance_ablation import peak_match_rate

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_test_df(path):
    df = pd.read_parquet(path)
    return df[df["n_components"] >= 2].reset_index(drop=True)


def drop_con_J_matches(con_j, drop_frac, generator=None):
    """Zero out drop_frac of con_j's off-diagonal 1-entries, symmetrically."""
    if drop_frac <= 0.0:
        return con_j
    B, P, _ = con_j.shape
    eye = torch.eye(P, dtype=torch.bool, device=con_j.device).unsqueeze(0)
    is_match = (con_j == 1.0) & ~eye
    upper = torch.triu(torch.ones(P, P, dtype=torch.bool, device=con_j.device), diagonal=1).unsqueeze(0)
    rand = torch.rand(B, P, P, device=con_j.device, generator=generator)
    drop_upper = (rand < drop_frac) & upper
    drop = (drop_upper | drop_upper.transpose(1, 2)) & is_match
    out = con_j.clone()
    out[drop] = 0.0
    return out


def collate_with_j_dropout(batch, K_max, noise_std, drop_frac, generator):
    out = collate128(batch, K_max=K_max, noise_std=noise_std)
    out["con_J"] = drop_con_J_matches(out["con_J"], drop_frac, generator=generator)
    return out


@torch.no_grad()
def evaluate_at_drop_frac(model, loader, device, K, drop_con=False, n_component_bins=(2, 3, 4, 5)):
    """Returns (overall_acc, peak_match_rate, {n_components: acc})."""
    model.eval()
    correct = total = 0
    matched_weighted = n_valid_total = 0
    per_n = {c: [0, 0] for c in n_component_bins}
    for batch in loader:
        con = None if drop_con else batch["con"].to(device)
        mask = batch["mask"].to(device)
        component_mask = batch["component_mask"].to(device)
        mix_peaks = batch["mix_peaks"].to(device)
        mix_integrals = batch["mix_integrals"].to(device)
        targets = batch["targets"].to(device)
        conj = batch["con_J"].to(device)
        n_components = batch["n_components"].to(device)

        slots, slot_attn, _, _ = model(mix_peaks, mix_integrals, con, mask, component_mask, conj)
        _, mapping = slot_hungarian_loss(slot_attn, targets, mask, K=K)
        preds = slot_attn.argmax(dim=1)
        mapped_preds = torch.gather(mapping, 1, preds.clamp_min(0))
        is_correct = (mapped_preds == targets)
        mask_flat = mask.bool().view(-1)
        correct += is_correct.view(-1)[mask_flat].sum().item()
        total += mask_flat.sum().item()

        n_valid = mask.bool().sum().item()
        matched_weighted += peak_match_rate(conj, mask) * n_valid
        n_valid_total += n_valid

        n_comp_per_peak = n_components.unsqueeze(1).expand_as(mask)
        for c in n_component_bins:
            peak_mask = mask.bool() & (n_comp_per_peak == c)
            per_n[c][0] += is_correct[peak_mask].sum().item()
            per_n[c][1] += peak_mask.sum().item()

    acc_per_n = {c: (correct_c / total_c if total_c else float("nan")) for c, (correct_c, total_c) in per_n.items()}
    return correct / total, matched_weighted / n_valid_total, acc_per_n


def demo():
    """Self-check: drop_con_J_matches removes matches symmetrically and
    leaves the diagonal untouched."""
    con_j = torch.tensor([[
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]])
    gen = torch.Generator().manual_seed(0)
    out = drop_con_J_matches(con_j, drop_frac=1.0, generator=gen)
    assert out[0, 0, 1].item() == 0.0 and out[0, 1, 0].item() == 0.0
    assert out[0, 1, 2].item() == 0.0 and out[0, 2, 1].item() == 0.0
    assert out[0, 0, 0].item() == 1.0 and out[0, 1, 1].item() == 1.0 and out[0, 2, 2].item() == 1.0
    zero_out = drop_con_J_matches(con_j, drop_frac=0.0, generator=gen)
    assert torch.equal(zero_out, con_j)
    print("j_coupling_dropout_ablation self-check passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "outputs/NMRexp_conJ_Slot_attn_10_best.pt",
                    help="a checkpoint trained WITH conJ active (run 10 was trained with --drop-con, conJ intact)")
    p.add_argument("--test-parquet", type=Path, default=REPO_ROOT / "data/test_molecule_up25_orig_1prct.parquet")
    p.add_argument("--drop-fracs", type=str, default="0.0,0.1,0.25,0.5,0.75,1.0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--mlp-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim-head", type=int, default=64)
    p.add_argument("--slot-iters", type=int, default=5)
    p.add_argument("--drop-con", action="store_true", default=True,
                    help="must match how the checkpoint was trained (run 10 used --drop-con); pass --no-drop-con to disable")
    p.add_argument("--no-drop-con", dest="drop_con", action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--csv", type=Path, default=None, help="also write results to this CSV path")
    p.add_argument("--plot-out", type=Path, default=None, help="also save an accuracy-vs-drop_frac plot here, one line per n_components")
    p.add_argument("--n-component-bins", type=str, default="2,3,4,5")
    p.add_argument("--seed", type=int, default=0, help="seeds both the drop mask RNG and integral noise, for reproducibility")
    args = p.parse_args()
    n_component_bins = tuple(int(c) for c in args.n_component_bins.split(","))

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
    n_comp_header = "".join(f"{f'acc ({c}-comp)':>16}" for c in n_component_bins)
    print(f"{'drop_frac':>10}{'accuracy':>12}{'peaks matched':>16}{n_comp_header}")
    for drop_frac in (float(f) for f in args.drop_fracs.split(",")):
        generator = torch.Generator().manual_seed(args.seed)
        collate_fn = functools.partial(
            collate_with_j_dropout, K_max=args.K, noise_std=args.noise_std,
            drop_frac=drop_frac, generator=generator,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        acc, match_rate, acc_per_n = evaluate_at_drop_frac(
            model, loader, device, args.K, drop_con=args.drop_con, n_component_bins=n_component_bins,
        )
        per_n_str = "".join(f"{acc_per_n[c]*100:>15.2f}%" for c in n_component_bins)
        print(f"{drop_frac:>10.2f}{acc*100:>11.2f}%{match_rate*100:>15.2f}%{per_n_str}")
        row = {"drop_frac": drop_frac, "accuracy": acc, "peaks_matched_frac": match_rate}
        row.update({f"accuracy_{c}comp": acc_per_n[c] for c in n_component_bins})
        rows.append(row)

    fieldnames = ["drop_frac", "accuracy", "peaks_matched_frac"] + [f"accuracy_{c}comp" for c in n_component_bins]
    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")

    if args.plot_out is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        drop_fracs = [r["drop_frac"] for r in rows]
        ax.plot(drop_fracs, [r["accuracy"] * 100 for r in rows], "k--", label="overall", linewidth=1.5)
        for c in n_component_bins:
            ax.plot(drop_fracs, [r[f"accuracy_{c}comp"] * 100 for r in rows], marker="o", label=f"{c}-component")
        ax.set_xlabel("fraction of J-coupling matches zeroed")
        ax.set_ylabel("assignment accuracy (%)")
        ax.set_title(args.checkpoint.stem)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot_out, dpi=150)
        print(f"wrote {args.plot_out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        main()
