#!/usr/bin/env python3
"""
Script 3: Extinction phase — continue generating with NO reinforcement.
Tests whether superstitious patterns persist after random rewards stop.
"""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Defaults to checkpoints/after_ncr.pt",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ecfg = cfg["extinction"]
    set_seed(ecfg["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    default_ckpt = ROOT / cfg["paths"]["checkpoints_dir"] / "after_ncr.pt"
    ckpt_path = Path(args.checkpoint or default_ckpt)
    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    run_dir = ensure_dir(ROOT / cfg["paths"]["runs_dir"] / "extinction")
    behavior_path = run_dir / "behavior_trace.jsonl"

    ctx = torch.tensor([[enc.eot_token]], dtype=torch.long, device=device)
    token_history: list[int] = [enc.eot_token]

    duration_sec = ecfg["duration_hours"] * 3600
    end_time = time.time() + duration_sec
    next_gen_at = time.time()
    token_count = 0

    print(f"Extinction for {ecfg['duration_hours']}h (no reinforcement)")

    while time.time() < end_time:
        now = time.time()
        if now >= next_gen_at:
            with torch.no_grad():
                logits, _ = model(ctx[:, -model.config.block_size :])
                logits = logits[:, -1, :] / max(ecfg["temperature"], 1e-8)
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
            next_gen_at = now + ecfg["generation_interval_sec"]
        time.sleep(0.05)

    ckpt_out = ensure_dir(ROOT / cfg["paths"]["checkpoints_dir"]) / "after_extinction.pt"
    save_checkpoint(
        ckpt_out,
        model,
        None,
        {"phase": "extinction", "tokens_generated": token_count},
    )
    print(f"Done. tokens={token_count}. Saved {ckpt_out}")


if __name__ == "__main__":
    main()
