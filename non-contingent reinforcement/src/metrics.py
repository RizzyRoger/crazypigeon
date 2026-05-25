"""Evaluation and analysis helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.data import TokenBatchLoader
from src.gpt import GPT


@torch.no_grad()
def estimate_loss(
    model: GPT,
    loader: TokenBatchLoader,
    eval_iters: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_iters):
        x, y = loader.get_batch()
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def decode_tokens(ids: list[int]) -> str:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    return enc.decode(ids)
