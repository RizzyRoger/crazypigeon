#!/usr/bin/env python3
"""
Pre-experiment: generate validation samples from the ready baseline.
Used later to measure behavioral drift (loss + text snapshots).
"""

import argparse
import sys
from pathlib import Path

import torch
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import load_checkpoint
from src.metrics import decode_tokens
from src.utils import ensure_dir, load_config, save_json, set_seed, utc_now_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Defaults to checkpoints/baseline_ready.pt",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(0)
    device = torch.device("cpu")  # generation is light
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")

    ckpt_path = Path(args.checkpoint or ROOT / cfg["paths"]["checkpoints_dir"] / "baseline_ready.pt")
    model, meta = load_checkpoint(ckpt_path, device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    vcfg = cfg["validation_snapshot"]
    prompts = [
        "Once upon a time",
        "The scientist observed that",
        "In the laboratory, the",
        "Behavior is defined as",
        "Reinforcement schedules affect",
    ]
    while len(prompts) < vcfg["num_prompts"]:
        prompts.append(prompts[len(prompts) % 5] + f" (sample {len(prompts)})")

    samples = []
    for i, prompt in enumerate(prompts[: vcfg["num_prompts"]]):
        ctx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(ctx, vcfg["tokens_per_sample"], temperature=vcfg["temperature"])
        ids = out[0].tolist()
        samples.append(
            {
                "id": i,
                "prompt": prompt,
                "text": decode_tokens(ids),
                "token_ids": ids,
            }
        )

    out_dir = ensure_dir(ROOT / cfg["paths"]["analysis_dir"])
    out_path = out_dir / "validation_snapshot.json"
    save_json(
        out_path,
        {
            "time": utc_now_iso(),
            "checkpoint": str(ckpt_path),
            "checkpoint_meta": meta,
            "config": vcfg,
            "samples": samples,
        },
    )
    print(f"Wrote {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
