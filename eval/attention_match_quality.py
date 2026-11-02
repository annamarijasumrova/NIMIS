"""Does the trained model's attention actually discriminate genuine
same-component peak matches from spurious ones, and does that discrimination
survive once the J-coupling matching tolerance is loosened past what the
model was trained with?

peak_matching_tolerance_ablation.py showed loosening tolerance adds matches
without improving accuracy -- consistent with the new matches being lower
quality, but that was inferred from aggregate accuracy, not observed
directly. This looks at the mechanism itself: feed the model con_J built at
a loosened tolerance (so every matched pair, real or spurious, gets the same
uniform +w_conJ bias -- see RatioAttention.forward, nimis_model.py:70-72),
then compare the actual post-softmax attention weight on each matched pair,
split by:
  - tolerance origin: matched already at the trained tolerance (0.1 Hz) vs.
    matched only once loosened
  - ground truth: the two peaks are actually the same component vs. not
        (targets are known for the test set, so this is exact, not inferred)

If attention only reflects "matched or not" (the crude thing con_J encodes),
true and false matches within a tolerance group would get similar weight.
If it reflects real chemistry, true matches should draw more attention than
false ones in both tolerance groups.
"""
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.dataset_nimis_parquet import NimisMixtureParquetDataset
from src.nimis_model import NMRPeakSlotAttn
from scripts.collate import build_con_J_hard_overlap, collate128

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_test_df(path):
    """Mirrors load_multi() in scripts/train.py."""
    df = pd.read_parquet(path)
    return df[df["n_components"] >= 2].reset_index(drop=True)


def peak_pair_groups(mask, targets, con_tight, con_loose):
    """(B,P,P) boolean mask per group, over off-diagonal valid peak pairs."""
    _, P = mask.shape
    valid = mask.bool()
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)
    diag = torch.eye(P, dtype=torch.bool, device=mask.device).unsqueeze(0)
    valid_pair = valid_pair & ~diag

    same = (targets.unsqueeze(2) == targets.unsqueeze(1)) & valid_pair
    tight = (con_tight > 0.5) & valid_pair
    loose = (con_loose > 0.5) & valid_pair
    loose_only = loose & ~tight
    unmatched = valid_pair & ~loose

    return {
        "tight & same-component (true match)":        tight & same,
        "tight & diff-component (false match)":        tight & ~same,
        "loose-only & same-component (true match)":    loose_only & same,
        "loose-only & diff-component (false match)":   loose_only & ~same,
        "unmatched & same-component":                  unmatched & same,
        "unmatched & diff-component":                  unmatched & ~same,
    }


@torch.no_grad()
def collect_attention_by_group(model, loader, device, tight_tol, loose_tol, drop_con=False):
    group_names = list(peak_pair_groups(
        torch.zeros(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1, 1), torch.zeros(1, 1, 1),
    ).keys())
    depth = None
    layer_sum = {g: None for g in group_names}
    layer_count = {g: None for g in group_names}

    for batch in loader:
        mask = batch["mask"].to(device)
        con = None if drop_con else batch["con"].to(device)
        component_mask = batch["component_mask"].to(device)
        mix_peaks = batch["mix_peaks"].to(device)
        mix_integrals = batch["mix_integrals"].to(device)
        targets = batch["targets"].to(device)
        j_vals = batch["j_vals"].to(device)
        j_mask = batch["j_mask"].to(device)

        con_tight = build_con_J_hard_overlap(j_vals, j_mask, tol=tight_tol)
        con_loose = build_con_J_hard_overlap(j_vals, j_mask, tol=loose_tol)

        _, _, _, attn_maps = model(mix_peaks, mix_integrals, con, mask, component_mask, con_loose)
        if depth is None:
            depth = len(attn_maps)
            for g in group_names:
                layer_sum[g] = [0.0] * depth
                layer_count[g] = [0] * depth

        groups = peak_pair_groups(mask, targets, con_tight, con_loose)
        for layer_idx, attn in enumerate(attn_maps):
            for name, gmask in groups.items():
                if not gmask.any():
                    continue
                gmask_h = gmask.unsqueeze(1).expand_as(attn)
                vals = attn[gmask_h]
                layer_sum[name][layer_idx] += vals.sum().item()
                layer_count[name][layer_idx] += vals.numel()

    return group_names, depth, layer_sum, layer_count


def demo():
    """Self-check: 1 batch, 1 layer, 1 head, P=3. Peaks 0,1 same component
    and matched tight; peak 2 different component, matched only loose."""
    mask = torch.tensor([[True, True, True]])
    targets = torch.tensor([[0, 0, 1]])
    con_tight = torch.tensor([[[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]]])
    con_loose = torch.tensor([[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]])

    groups = peak_pair_groups(mask, targets, con_tight, con_loose)
    assert groups["tight & same-component (true match)"][0, 0, 1].item() is True
    assert groups["loose-only & diff-component (false match)"][0, 0, 2].item() is True
    assert groups["loose-only & diff-component (false match)"][0, 1, 2].item() is True
    assert not groups["unmatched & same-component"].any()

    attn = torch.zeros(1, 1, 3, 3)
    attn[0, 0, 0, 1] = 0.9
    attn[0, 0, 1, 0] = 0.9
    attn[0, 0, 0, 2] = 0.1
    attn[0, 0, 2, 0] = 0.1
    attn[0, 0, 1, 2] = 0.1
    attn[0, 0, 2, 1] = 0.1
    group_mask = groups["tight & same-component (true match)"].unsqueeze(1).expand_as(attn)
    assert abs(attn[group_mask].mean().item() - 0.9) < 1e-6
    group_mask = groups["loose-only & diff-component (false match)"].unsqueeze(1).expand_as(attn)
    assert abs(attn[group_mask].mean().item() - 0.1) < 1e-6

    print("attention_match_quality self-check passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "outputs/NMRexp_conJ_Slot_attn_8_best.pt",
                    help="a checkpoint trained WITH conj active")
    p.add_argument("--test-parquet", type=Path, default=REPO_ROOT / "data/test_molecule_up25_orig_1prct.parquet")
    p.add_argument("--tight-tol", type=float, default=0.1, help="trained matching tolerance, Hz")
    p.add_argument("--loose-tol", type=float, default=1.0, help="loosened matching tolerance fed at eval, Hz")
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
    model.eval()

    test_df = load_test_df(args.test_parquet)
    dataset = NimisMixtureParquetDataset(test_df)

    import functools
    collate_fn = functools.partial(collate128, K_max=args.K, noise_std=args.noise_std)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    group_names, depth, layer_sum, layer_count = collect_attention_by_group(
        model, loader, device, args.tight_tol, args.loose_tol, drop_con=args.drop_con
    )

    print(f"tight tol={args.tight_tol} Hz, loose tol={args.loose_tol} Hz, {depth} layers\n")
    print(f"{'group':>42}{'overall attn':>14}" + "".join(f"{'L'+str(i):>9}" for i in range(depth)))
    csv_rows = []
    for name in group_names:
        total_sum = sum(layer_sum[name])
        total_count = sum(layer_count[name])
        overall = total_sum / total_count if total_count else float("nan")
        per_layer = [
            (layer_sum[name][i] / layer_count[name][i]) if layer_count[name][i] else float("nan")
            for i in range(depth)
        ]
        row = f"{name:>42}{overall:>14.4f}"
        row += "".join(f"{v:>9.4f}" for v in per_layer)
        print(row)
        csv_rows.append({"group": name, "overall_attn": overall,
                          **{f"layer_{i}": v for i, v in enumerate(per_layer)}})

    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        main()
