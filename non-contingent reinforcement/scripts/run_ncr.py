#!/usr/bin/env python3
"""
Script 2: Non-contingent reinforcement (NCR) experiment.

The model emits one token per second. At random intervals (uniform between
min/max seconds), we reinforce whatever it was just doing by gradient steps
that increase log-likelihood of the recent token window — independent of quality.
This mirrors Skinner's food delivery unrelated to the pigeon's response.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import load_checkpoint, save_checkpoint
from src.metrics import decode_tokens
from src.utils import append_jsonl, ensure_dir, load_config, set_seed, utc_now_iso


def reinforce_window(
    model: torch.nn.Module,
    token_ids: list[int],
    steps: int,
    lr: float,
    device: torch.device,
) -> float:
    """Positive reinforcement: maximize likelihood of current behavior trace."""
    if len(token_ids) < 2:
        return 0.0
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    block = model.config.block_size
    window = token_ids[-block:]
    x = torch.tensor([window[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([window[1:]], dtype=torch.long, device=device)
    last_loss = 0.0
    for _ in range(steps):
        _, loss = model(x, y)
        last_loss = loss.item()
        opt.zero_grad(set_to_none=True)
        (-loss).backward()
        opt.step()
    model.eval()
    return last_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ncfg = cfg["ncr"]
    set_seed(ncfg["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt_path = Path(
        args.checkpoint or ROOT / cfg["paths"]["checkpoints_dir"] / "baseline_ready.pt"
    )
    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    run_dir = ensure_dir(ROOT / cfg["paths"]["runs_dir"] / "ncr")
    log_path = run_dir / "ncr_events.jsonl"
    behavior_path = run_dir / "behavior_trace.jsonl"

    # Rolling context — "what the organism is doing" when food arrives
    ctx = torch.tensor([[enc.eot_token]], dtype=torch.long, device=device)
    token_history: list[int] = [enc.eot_token]

    duration_sec = ncfg["duration_hours"] * 3600
    end_time = time.time() + duration_sec
    next_reinforce_at = time.time() + random.uniform(
        ncfg["min_interval_sec"], ncfg["max_interval_sec"]
    )
    next_gen_at = time.time()

    reinforce_count = 0
    token_count = 0

    print(
        f"NCR running for {ncfg['duration_hours']}h | "
        f"reinforce every {ncfg['min_interval_sec']}-{ncfg['max_interval_sec']}s | "
        f"1 token/{ncfg['generation_interval_sec']}s"
    )

    while time.time() < end_time:
        now = time.time()

        if now >= next_gen_at:
            with torch.no_grad():
                logits, _ = model(ctx[:, -model.config.block_size :])
                logits = logits[:, -1, :] / max(ncfg["temperature"], 1e-8)
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
            ctx = torch.cat([ctx, next_id], dim=1)
            tid = int(next_id.item())
            token_history.append(tid)
            token_count += 1
            append_jsonl(
                behavior_path,
                {
                    "time": utc_now_iso(),
                    "event": "token",
                    "token_id": tid,
                    "token": enc.decode([tid]),
                    "recent_text": decode_tokens(token_history[-80:]),
                },
            )
            next_gen_at = now + ncfg["generation_interval_sec"]

        if now >= next_reinforce_at:
            window = token_history[-ncfg["reinforce_window_tokens"] :]
            loss_before = reinforce_window(
                model,
                window,
                ncfg["reinforce_steps"],
                ncfg["reinforce_lr"],
                device,
            )
            reinforce_count += 1
            append_jsonl(
                log_path,
                {
                    "time": utc_now_iso(),
                    "event": "reinforce",
                    "reinforce_index": reinforce_count,
                    "window_tokens": len(window),
                    "loss_after_reinforce": loss_before,
                    "behavior_at_reinforce": decode_tokens(window),
                },
            )
            next_reinforce_at = now + random.uniform(
                ncfg["min_interval_sec"], ncfg["max_interval_sec"]
            )

        time.sleep(0.05)

    ckpt_out = ensure_dir(ROOT / cfg["paths"]["checkpoints_dir"]) / "after_ncr.pt"
    save_checkpoint(
        ckpt_out,
        model,
        None,
        {
            "phase": "ncr",
            "reinforce_count": reinforce_count,
            "tokens_generated": token_count,
            "duration_hours": ncfg["duration_hours"],
        },
    )
    print(f"Done. reinforcements={reinforce_count}, tokens={token_count}")
    print(f"Saved {ckpt_out}")


if __name__ == "__main__":
    main()
