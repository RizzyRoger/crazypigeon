"""Save/load model checkpoints and readiness markers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.gpt import GPT, GPTConfig


def build_model(model_cfg: dict[str, Any], device: torch.device) -> GPT:
    config = GPTConfig(
        vocab_size=50257,
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        block_size=model_cfg["block_size"],
        dropout=model_cfg["dropout"],
    )
    model = GPT(config)
    return model.to(device)


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": model.config.__dict__,
        "meta": meta,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[GPT, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    return model, ckpt.get("meta", {})
