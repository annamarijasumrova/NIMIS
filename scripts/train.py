import argparse
import functools
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from scripts.dataset_nimis_parquet import NimisMixtureParquetDataset
from src.nimis_model import NMRPeakSlotAttn, slot_hungarian_loss
from scripts.collate import collate128
from helpers.violin_parser import (
    parse_assignment_file,
    plot_confidence_violin,
    plot_accuracy_by_size,
    plot_confidence_violin_grid,
    plot_accuracy_by_size_grid,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train NMRPeakSlotAttn on a mixtures parquet dataset.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--train-dataset", type=Path, required=True, help="Parquet file with training mixtures")
    p.add_argument("--val-dataset", type=Path, required=True, help="Parquet file with validation mixtures")
    p.add_argument("--test-dataset", type=Path, required=True, help="Parquet file with test mixtures")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))

    p.add_argument("--batch-size", type=int, default=128,
                    help="per-GPU batch size; with N GPUs the effective global batch size is N x this")
    p.add_argument("--learning-rate", type=float, default= 5e-4)
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=50)

    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--mlp-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim-head", type=int, default=64)
    p.add_argument("--K", type=int, default=5, help="slot count; must be >= max n_components in dataset")
    p.add_argument("--slot-iters", type=int, default=5)
    p.add_argument("--integral-noise-std", type=float, default=0.10,
                    help="relative noise std applied to peak integrals; con matrix is derived from the same noisy draw")
    p.add_argument("--test-noise-levels", type=str, default="0.0,0.05,0.10,0.15",
                    help="comma-separated relative noise stds for the post-training test noise sweep")
    p.add_argument("--vanilla-attn", action="store_true",
                    help="ablation: drop both con and con_J attention bias -> plain scaled-dot-product self-attention (vanilla transformer)")
    p.add_argument("--drop-con", action="store_true", help="ablation: drop only the con (intensity ratio) attention bias")
    p.add_argument("--per-head-bias", action="store_true", help="learn one w_con/w_conJ per attention head instead of one scalar per layer")
    p.add_argument("--drop-conj", action="store_true", help="ablation: drop only the con_J (J-coupling) attention bias")


    p.add_argument("--init-ckpt", type=Path, default=None,
                   help="warm-start: load model weights only (optimizer/scheduler restart fresh; set --learning-rate to the LR reached)")
    p.add_argument("--start-epoch", type=int, default=0,
                   help="epochs already completed; loop runs start_epoch..num-epochs-1 (--num-epochs stays the TOTAL)")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--wandb-project", default="nimis")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


def setup_distributed():
    """Multi-GPU only activates when launched via torchrun (it sets WORLD_SIZE/RANK/LOCAL_RANK).
    A plain `python -m scripts.train ...` launch is untouched: world_size=1, rank=0."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def make_loader(df, args, shuffle, collate_fn, world_size=1, rank=0):
    dataset = NimisMixtureParquetDataset(df)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        collate_fn=collate_fn,
    )
    return loader, sampler


def evaluate_loader(model, data_loader, device, K, save_assignments_path=None, drop_con=False, drop_conj=False):
    model.eval()
    eval_loss = 0.0
    correct, total = 0, 0
    conf_correct = []
    conf_wrong = []
    num_batches = len(data_loader)

    assign_f = None
    if save_assignments_path is not None:
        assign_f = open(save_assignments_path, "w")
        assign_f.write("Peak assignments\n")

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            con = batch["con"].to(device)
            mask = batch["mask"].to(device)
            component_mask = batch["component_mask"].to(device)
            mix_peaks = batch["mix_peaks"].to(device)
            mix_integrals = batch["mix_integrals"].to(device)
            targets = batch["targets"].to(device)
            smiles = batch["smiles"]
            proportions = batch["proportions"]
            solvents = batch["solvent"]
            frequencies = batch["frequency"]
            conj = batch["con_J"].to(device)
            if drop_con:
                con = None
            if drop_conj:
                conj = None

            slots, slot_attn, peak_features, attn_maps = model(
                mix_peaks, mix_integrals, con, mask, component_mask, conj
            )
            loss, mapping = slot_hungarian_loss(slot_attn, targets, mask, K=K)
            eval_loss += loss.item()

            preds = slot_attn.argmax(dim=1)
            mapped_preds = torch.gather(mapping, 1, preds.clamp_min(0))
            mask_flat = mask.bool().view(-1)
            correct += (mapped_preds.view(-1)[mask_flat] == targets.view(-1)[mask_flat]).sum().item()
            total += mask_flat.sum().item()

            if assign_f is not None:
                preds_cpu = preds.cpu()
                mapping_cpu = mapping.cpu()
                targets_cpu = targets.cpu()
                mask_cpu = mask.cpu()
                mix_peaks_cpu = mix_peaks.cpu()

                B, P = preds_cpu.shape
                assign_f.write(f"\n=== Batch {batch_idx+1}/{num_batches} ===\n")

                for b in range(B):
                    n_comp_b = int(batch["n_components"][b].item())
                    prop_b = proportions[b, :n_comp_b].detach().cpu().tolist()

                    assign_f.write(f"\nSample {b}\n")
                    assign_f.write(f"Solvent: {solvents[b]}\n")
                    assign_f.write(f"Frequency: {frequencies[b]}\n")
                    assign_f.write("SMILES: " + ", ".join(smiles[b]) + "\n")
                    assign_f.write("Component proportions:\n")
                    for comp_idx, (smi, prop) in enumerate(zip(smiles[b], prop_b)):
                        assign_f.write(f"  Component {comp_idx}: {smi}\tproportion={prop:.4f}\n")
                    assign_f.write("peak_position_ppm\tpred_mapped_slot\ttrue_slot\tconfidence\n")

                    for p in range(P):
                        if not mask_cpu[b, p].item():
                            continue

                        raw_slot = int(preds_cpu[b, p].item())
                        mapped_comp = int(mapping_cpu[b, raw_slot].item())
                        true_comp = int(targets_cpu[b, p].item())
                        conf = float(slot_attn[b, raw_slot, p].item())

                        assign_f.write(
                            f"t{mix_peaks_cpu[b, p].item()} ppm\t{mapped_comp}\t"
                            f"{true_comp}\t{conf:.4f}\n"
                        )

                        if mapped_comp == true_comp:
                            conf_correct.append(conf)
                        else:
                            conf_wrong.append(conf)

    if assign_f is not None:
        assign_f.close()

    avg_eval_loss = eval_loss / len(data_loader)
    eval_acc = correct / total if total > 0 else 0.0
    return avg_eval_loss, eval_acc, conf_correct, conf_wrong


def main():
    args = parse_args()
    drop_con = args.vanilla_attn or args.drop_con
    drop_conj = args.vanilla_attn or args.drop_conj
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}") if world_size > 1 else torch.device(args.device)
    if not is_main:
        os.environ["WANDB_MODE"] = "disabled"
    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.run_id,
        config=vars(args),
    )

    if is_main:
        print("\n--- START TRAINING SCRIPT ---", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        print(f"Run ID: {args.run_id} | train: {args.train_dataset} | val: {args.val_dataset} | test: {args.test_dataset}", flush=True)
        if world_size > 1:
            print(f"Distributed: world_size={world_size} (effective global batch size = {world_size * args.batch_size})", flush=True)

    def load_multi(path):
        df = pd.read_parquet(path)
        if "smiles" not in df.columns and "smiles_list" in df.columns:
            df = df.rename(columns={"smiles_list": "smiles"})
        n_dropped_single = int((df["n_components"] < 2).sum())
        if n_dropped_single and is_main:
            print(f"Dropping {n_dropped_single:,}/{len(df):,} single-component mixtures from {path}", flush=True)
        return df[df["n_components"] >= 2].reset_index(drop=True)

    train_df = load_multi(args.train_dataset)
    val_df = load_multi(args.val_dataset)
    test_df = load_multi(args.test_dataset)
    if is_main:
        print(f"Rows -> train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}", flush=True)

    collate_fn = functools.partial(collate128, K_max=args.K, noise_std=args.integral_noise_std)
    train_loader, train_sampler = make_loader(train_df, args, shuffle=True, collate_fn=collate_fn,
                                               world_size=world_size, rank=rank)
    val_loader, _ = make_loader(val_df, args, shuffle=False, collate_fn=collate_fn)
    test_loader, _ = make_loader(test_df, args, shuffle=False, collate_fn=collate_fn)

    loss_history = []
    val_loss_history = []
    val_epoch_history = []
    batch_loss_history = []

    model = NMRPeakSlotAttn(
        dim=args.dim,
        mlp_dim=args.mlp_dim,
        depth=args.depth,
        dim_head=args.dim_head,
        heads=args.heads,
        K=args.K,
        slot_iters=args.slot_iters,
        per_head_bias=args.per_head_bias,
    ).to(device)
    if args.init_ckpt is not None:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device))
        if is_main:
            print(f"Warm-started from {args.init_ckpt}; resuming at epoch {args.start_epoch + 1}", flush=True)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=(drop_con or drop_conj))
    raw_model = model.module if world_size > 1 else model

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=5e-6
    )

    save_assignments = True
    best_val_loss = float("inf")
    global_step = 0
    best_ckpt_path = args.output_dir / f"NMRexp_conJ_Slot_attn_{args.run_id}_best.pt"
    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        total_loss = 0
        counter = 0
        epoch_start = time.time()

        is_last_epoch = (epoch == args.num_epochs - 1)
        should_validate = ((epoch + 1) % args.val_every == 0) or is_last_epoch

        for batch in train_loader:
            batch_start = time.time()
            data_load_time = time.time() - batch_start
            t0 = time.time()

            optimizer.zero_grad()
            con = batch["con"].to(device)
            mask = batch["mask"].to(device)
            component_mask = batch["component_mask"].to(device)
            mix_peaks = batch["mix_peaks"].to(device)
            mix_integrals = batch["mix_integrals"].to(device)
            targets = batch["targets"].to(device)
            conj = batch["con_J"].to(device)
            if drop_con:
                con = None
            if drop_conj:
                conj = None

            slots, slot_attn, peak_features, attn_maps = model(mix_peaks, mix_integrals, con, mask, component_mask, conj)
            loss, _ = slot_hungarian_loss(slot_attn, targets, mask, K=args.K)
            loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_norm = float(total_norm.item() if hasattr(total_norm, "item") else total_norm)
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            compute_time = time.time() - t0

            total_time = data_load_time + compute_time
            total_loss += loss.item()
            batch_loss_history.append(loss.item())
            counter += 1
            global_step += 1

            wandb.log({
                "train/batch_loss": loss.item(),
                "train/grad_norm": total_norm,
                "train/data_load_time": data_load_time,
                "train/compute_time": compute_time,
                "epoch": epoch + 1,
            }, step=global_step)

            if is_main and (counter == 1 or counter % args.log_every == 0 or counter == len(train_loader)):
                print(f"[Epoch {epoch+1:02d} | Batch {counter:04d}/{len(train_loader):04d}] "
                      f"Loss={loss.item():.4f} | "
                      f"GradNorm={total_norm:.4f} | "
                      f"DataLoad={data_load_time:.3f}s | "
                      f"Compute={compute_time:.3f}s | "
                      f"Total={total_time:.3f}s", flush=True)

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        if world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (loss_tensor / world_size).item()
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if is_main:
            print(f"Current LR: {current_lr:.6e}")
            print(f"\n==== Epoch [{epoch+1}/{args.num_epochs}] "
                  f"Completed in {epoch_time/60:.2f} min | "
                  f"Avg Loss={avg_loss:.4f} ====\n")

            wandb_log = {
                "train/epoch_loss": avg_loss,
                "train/lr": current_lr,
                "train/epoch_time_min": epoch_time / 60,
                "epoch": epoch + 1,
            }
            att_layer = raw_model.transformer.layers[0][0]
            if not drop_con:
                print(f"Epoch {epoch+1}: w_con={att_layer.w_con.data}\n")
                vals = att_layer.w_con.data.flatten().tolist()
                if len(vals) == 1:
                    wandb_log["train/w_con"] = vals[0]
                else:
                    wandb_log.update({f"train/w_con_h{i}": v for i, v in enumerate(vals)})
            if not drop_conj:
                print(f"Epoch {epoch+1}: w_conJ={att_layer.w_conJ.data}\n")
                vals = att_layer.w_conJ.data.flatten().tolist()
                if len(vals) == 1:
                    wandb_log["train/w_conJ"] = vals[0]
                else:
                    wandb_log.update({f"train/w_conJ_h{i}": v for i, v in enumerate(vals)})

            wandb.log(wandb_log, step=global_step)

        if not should_validate:
            if is_main:
                print(f"Skipping validation at epoch {epoch+1}; next validation every {args.val_every} epochs or final epoch.\n", flush=True)
            if world_size > 1:
                dist.barrier()
            continue

        if is_main:
            val_assignments_path = None
            if is_last_epoch and save_assignments:
                val_assignments_path = args.output_dir / f"assignments_nmrexp_val_slot_conj_{args.run_id}.txt"

            avg_val_loss, val_acc, conf_correct, conf_wrong = evaluate_loader(
                model=raw_model,
                data_loader=val_loader,
                device=device,
                K=args.K,
                save_assignments_path=val_assignments_path,
                drop_con=drop_con,
                drop_conj=drop_conj,
            )

            val_loss_history.append(avg_val_loss)
            val_epoch_history.append(epoch + 1)

            wandb.log({
                "val/loss": avg_val_loss,
                "val/accuracy": val_acc,
                "epoch": epoch + 1,
            }, step=global_step)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(raw_model.state_dict(), best_ckpt_path)
                print(f"New best val loss {best_val_loss:.4f} -> saved {best_ckpt_path}\n", flush=True)
                wandb.run.summary["best_val_loss"] = best_val_loss

            if val_assignments_path is not None:
                print(
                    f"Validation: Loss={avg_val_loss:.4f} | Accuracy={val_acc*100:.2f}% | "
                    f"Assignments saved to {val_assignments_path}\n",
                    flush=True,
                )
            else:
                print(
                    f"Validation: Loss={avg_val_loss:.4f} | Accuracy={val_acc*100:.2f}%\n",
                    flush=True,
                )
            model.train()

        if world_size > 1:
            dist.barrier()

    if not is_main:
        if world_size > 1:
            dist.destroy_process_group()
        return

    noise_sweep_dir = args.output_dir / f"test_run_id_{args.run_id}"
    noise_sweep_dir.mkdir(parents=True, exist_ok=True)

    test_assignments_path = args.output_dir / f"assignments_nmrexp_test_slot_conj_{args.run_id}.txt"
    test_loss, test_acc, test_conf_correct, test_conf_wrong = evaluate_loader(
        model=raw_model,
        data_loader=test_loader,
        device=device,
        K=args.K,
        save_assignments_path=test_assignments_path,
        drop_con=drop_con,
        drop_conj=drop_conj,
    )
    print(
        f"Test: Loss={test_loss:.4f} | Accuracy={test_acc*100:.2f}% | "
        f"Assignments saved to {test_assignments_path}\n",
        flush=True,
    )
    wandb.log({"test/loss": test_loss, "test/accuracy": test_acc}, step=global_step)

    test_predicted_slot, test_true_slot, test_conf, test_n_components = parse_assignment_file(test_assignments_path)
    test_title_suffix = f"test, noise={args.integral_noise_std*100:.0f}%"
    plot_confidence_violin(
        test_conf_correct, test_conf_wrong,
        title=f"Peak-to-Component Assignment Confidence ({test_title_suffix})",
        save_path=noise_sweep_dir / f"violin_test_{args.run_id}.png",
    )
    plot_accuracy_by_size(
        test_n_components, test_predicted_slot, test_true_slot, test_conf,
        title=f"Precision & Accuracy by Mixture Size ({test_title_suffix})",
        save_path=noise_sweep_dir / f"accuracy_by_size_test_{args.run_id}.png",
    )
    wandb.log({
        "test/violin": wandb.Image(str(noise_sweep_dir / f"violin_test_{args.run_id}.png")),
        "test/accuracy_by_size": wandb.Image(str(noise_sweep_dir / f"accuracy_by_size_test_{args.run_id}.png")),
    })

    torch.save(raw_model.state_dict(), args.output_dir / f"NMRexp_conJ_Slot_attn_{args.run_id}.pt")

    noise_levels = [float(x) for x in args.test_noise_levels.split(",")]
    summary_path = noise_sweep_dir / f"summary_{args.run_id}.txt"
    violin_panels = []
    accsize_panels = []
    with open(summary_path, "w") as summary_f:
        summary_f.write("noise_std\tloss\taccuracy_pct\n")
        for noise in noise_levels:
            noisy_collate = functools.partial(collate128, K_max=args.K, noise_std=noise)
            noisy_test_loader, _ = make_loader(test_df, args, shuffle=False, collate_fn=noisy_collate)
            pct = round(noise * 100)
            assignments_path = noise_sweep_dir / f"assignments_test_noise{pct}pct_{args.run_id}.txt"

            noise_loss, noise_acc, conf_correct, conf_wrong = evaluate_loader(
                model=raw_model,
                data_loader=noisy_test_loader,
                device=device,
                K=args.K,
                save_assignments_path=assignments_path,
                drop_con=drop_con,
                drop_conj=drop_conj,
            )
            summary_f.write(f"{noise:.4f}\t{noise_loss:.4f}\t{noise_acc*100:.2f}\n")
            wandb.log({
                "noise_sweep/noise_std": noise,
                "noise_sweep/loss": noise_loss,
                "noise_sweep/accuracy": noise_acc,
            })
            print(
                f"Test noise sweep @ {pct}%: Loss={noise_loss:.4f} | Accuracy={noise_acc*100:.2f}% | "
                f"Assignments -> {assignments_path}",
                flush=True,
            )

            predicted_slot, true_slot, conf, n_components = parse_assignment_file(assignments_path)
            violin_panels.append((f"noise={pct}%", conf_correct, conf_wrong))
            accsize_panels.append((f"noise={pct}%", n_components, predicted_slot, true_slot, conf))

    plot_confidence_violin_grid(
        violin_panels, save_path=noise_sweep_dir / f"violin_noise_sweep_{args.run_id}.png"
    )
    plot_accuracy_by_size_grid(
        accsize_panels, save_path=noise_sweep_dir / f"accuracy_by_size_noise_sweep_{args.run_id}.png"
    )

    plt.figure(figsize=(6, 4))
    plt.plot(range(args.start_epoch + 1, args.num_epochs + 1), loss_history, marker='o', label='Train Loss')
    if len(val_loss_history) > 0:
        plt.plot(val_epoch_history, val_loss_history, marker='s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss vs Epoch')
    plt.legend()
    plt.tight_layout()
    loss_curve_path = args.output_dir / f'NMRexp_conJ_Slot_attn_{args.run_id}.png'
    plt.savefig(loss_curve_path)

    wandb.log({
        "final/loss_curve": wandb.Image(str(loss_curve_path)),
        "final/noise_sweep_violin": wandb.Image(str(noise_sweep_dir / f"violin_noise_sweep_{args.run_id}.png")),
        "final/noise_sweep_accuracy_by_size": wandb.Image(str(noise_sweep_dir / f"accuracy_by_size_noise_sweep_{args.run_id}.png")),
    })
    wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
