#!/usr/bin/env python3
"""
Script 1: Train baseline GPT on English.
Stops when val loss <= target_val_loss (i) or max_iters.
Saves checkpoints and marks readiness.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import build_model, save_checkpoint
from src.data import TokenBatchLoader
from src.metrics import estimate_loss
from src.utils import append_jsonl, ensure_dir, get_device, load_config, save_json, set_seed, utc_now_iso


def lr_at_step(cfg: dict, step: int) -> float:
    base = cfg["learning_rate"]
    if step < cfg["warmup_iters"]:
        return base * step / max(cfg["warmup_iters"], 1)
    if step > cfg["lr_decay_iters"]:
        return cfg["min_lr"]
    decay_ratio = (step - cfg["warmup_iters"]) / max(
        cfg["lr_decay_iters"] - cfg["warmup_iters"], 1
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg["min_lr"] + coeff * (base - cfg["min_lr"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device()
    print(f"device={device}")

    data_dir = ROOT / cfg["paths"]["data_dir"]
    ckpt_dir = ensure_dir(ROOT / cfg["paths"]["checkpoints_dir"])
    run_log = ensure_dir(ROOT / cfg["paths"]["runs_dir"]) / "baseline_train.jsonl"

    mcfg = cfg["model"]
    bcfg = cfg["baseline"]
    model = build_model(mcfg, device)
    print(f"parameters={model.count_parameters():,}")

    train_loader = TokenBatchLoader(
        data_dir / "train.bin", mcfg["block_size"], bcfg["batch_size"], device
    )
    val_loader = TokenBatchLoader(
        data_dir / "val.bin", mcfg["block_size"], bcfg["batch_size"], device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=bcfg["learning_rate"])
    best_val = float("inf")
    ready = False

    pbar = tqdm(range(bcfg["max_iters"]), desc="baseline")
    for step in pbar:
        for g in optimizer.param_groups:
            g["lr"] = lr_at_step(bcfg, step)

        x, y = train_loader.get_batch()
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), bcfg["grad_clip"])
        optimizer.step()

        if step % bcfg["eval_interval"] == 0 or step == bcfg["max_iters"] - 1:
            train_loss = loss.item()
            val_loss = estimate_loss(model, val_loader, bcfg["eval_iters"])
            best_val = min(best_val, val_loss)
            pbar.set_postfix(train=f"{train_loss:.3f}", val=f"{val_loss:.3f}")
            record = {
                "time": utc_now_iso(),
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_jsonl(run_log, record)

            if step % (bcfg["eval_interval"] * 4) == 0:
                save_checkpoint(
                    ckpt_dir / f"baseline_step_{step}.pt",
                    model,
                    optimizer,
                    {"step": step, "val_loss": val_loss},
                )

            target = cfg["readiness"]["target_val_loss"]
            if val_loss <= target:
                ready = True
                save_checkpoint(
                    ckpt_dir / "baseline_ready.pt",
                    model,
                    optimizer,
                    {"step": step, "val_loss": val_loss, "ready": True},
                )
                save_json(
                    ckpt_dir / "readiness.json",
                    {
                        "ready": True,
                        "step": step,
                        "val_loss": val_loss,
                        "target_val_loss": target,
                        "time": utc_now_iso(),
                    },
                )
                print(f"\nReady for experiment: val_loss={val_loss:.4f} <= {target}")
                break

    save_checkpoint(
        ckpt_dir / "baseline_final.pt",
        model,
        optimizer,
        {"best_val": best_val, "ready": ready},
    )
    if not ready:
        print(
            f"\nDid not reach target val loss {cfg['readiness']['target_val_loss']}. "
            "Increase max_iters or adjust target."
        )


if __name__ == "__main__":
    main()
