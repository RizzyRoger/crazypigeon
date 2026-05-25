#!/usr/bin/env python3
"""
Script 4: Compare baseline vs post-NCR vs post-extinction.

Outputs analysis/summary.csv and analysis/report.json for easy review.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import load_checkpoint
from src.data import TokenBatchLoader
from src.metrics import decode_tokens, estimate_loss
from src.utils import ensure_dir, load_config, save_json, utc_now_iso


def load_validation_snapshot(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)["samples"]


@torch.no_grad()
def loss_on_snapshot(
    model,
    samples: list[dict],
    device: torch.device,
) -> float:
    losses = []
    for s in samples:
        ids = s["token_ids"]
        if len(ids) < 2:
            continue
        block = model.config.block_size
        chunk = ids[: block + 1]
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def regenerate_samples(
    model,
    samples: list[dict],
    device: torch.device,
    temperature: float,
    tokens: int,
) -> list[str]:
    enc = tiktoken.get_encoding("gpt2")
    texts = []
    for s in samples[:10]:
        ctx = torch.tensor([enc.encode(s["prompt"])], dtype=torch.long, device=device)
        out = model.generate(ctx, tokens, temperature=temperature)
        texts.append(decode_tokens(out[0].tolist()))
    return texts


def ngram_diversity(text: str, n: int = 3) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def behavior_stats(jsonl_path: Path) -> dict:
    if not jsonl_path.exists():
        return {"events": 0}
    tokens: list[str] = []
    reinforce = 0
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("event") == "token":
                tokens.append(row.get("token", ""))
            elif row.get("event") == "reinforce":
                reinforce += 1
    bigrams = Counter(zip(tokens, tokens[1:])) if len(tokens) > 1 else Counter()
    top_bigrams = bigrams.most_common(15)
    return {
        "events": len(tokens),
        "unique_tokens": len(set(tokens)),
        "top_bigrams": [(" ".join(a), c) for a, c in top_bigrams],
        "reinforce_events": reinforce,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    ccfg = cfg["compare"]
    mcfg = cfg["model"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt_dir = ROOT / cfg["paths"]["checkpoints_dir"]
    stages = [
        ("baseline", ckpt_dir / "baseline_ready.pt"),
        ("after_ncr", ckpt_dir / "after_ncr.pt"),
        ("after_extinction", ckpt_dir / "after_extinction.pt"),
    ]

    val_bin = ROOT / cfg["paths"]["data_dir"] / "val.bin"
    val_loader = TokenBatchLoader(
        val_bin, mcfg["block_size"], ccfg["eval_batch_size"], device
    )

    snap_path = ROOT / cfg["paths"]["analysis_dir"] / "validation_snapshot.json"
    snapshot = load_validation_snapshot(snap_path) if snap_path.exists() else []

    rows = []
    report = {"time": utc_now_iso(), "stages": {}}

    for name, path in stages:
        if not path.exists():
            print(f"Skipping {name}: missing {path}")
            continue
        model, meta = load_checkpoint(path, device)
        val_loss = estimate_loss(model, val_loader, ccfg["eval_iters"])
        snap_loss = loss_on_snapshot(model, snapshot, device) if snapshot else None
        regen = (
            regenerate_samples(
                model,
                snapshot,
                device,
                cfg["validation_snapshot"]["temperature"],
                120,
            )
            if snapshot
            else []
        )
        diversity = sum(ngram_diversity(t) for t in regen) / max(len(regen), 1)

        row = {
            "stage": name,
            "checkpoint": str(path),
            "val_loss": round(val_loss, 4),
            "validation_snapshot_loss": round(snap_loss, 4) if snap_loss else "",
            "regen_trigram_diversity": round(diversity, 4),
            "parameters": model.count_parameters(),
        }
        rows.append(row)
        report["stages"][name] = {
            **row,
            "meta": meta,
            "regenerated_preview": regen[:3],
        }

    ncr_log = ROOT / cfg["paths"]["runs_dir"] / "ncr" / "behavior_trace.jsonl"
    report["ncr_behavior"] = behavior_stats(ncr_log)
    ext_log = ROOT / cfg["paths"]["runs_dir"] / "extinction" / "behavior_trace.jsonl"
    report["extinction_behavior"] = behavior_stats(ext_log)

    if rows:
        baseline_val = next((r["val_loss"] for r in rows if r["stage"] == "baseline"), None)
        for r in rows:
            if baseline_val is not None and r["stage"] != "baseline":
                r["delta_val_loss_vs_baseline"] = round(r["val_loss"] - baseline_val, 4)

    out_dir = ensure_dir(ROOT / cfg["paths"]["analysis_dir"])
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report_path = out_dir / "report.json"
    save_json(report_path, report)
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
