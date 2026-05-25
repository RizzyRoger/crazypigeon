"""Dataset download, tokenization, and binary loaders."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


ENC = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = 50257


def _tokenize_texts(texts: list[str]) -> np.ndarray:
    ids: list[int] = []
    for text in tqdm(texts, desc="tokenize"):
        if not text or not text.strip():
            continue
        ids.extend(ENC.encode_ordinary(text))
        ids.append(ENC.eot_token)
    return np.array(ids, dtype=np.uint16)


def prepare_wikitext2(out_dir: str | Path) -> dict[str, str]:
    """Download WikiText-2 and write train/val .bin + meta.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_ids = _tokenize_texts(ds["train"]["text"])
    val_ids = _tokenize_texts(ds["validation"]["text"])

    train_path = out / "train.bin"
    val_path = out / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    meta = {
        "dataset": "wikitext-2-raw-v1",
        "vocab_size": VOCAB_SIZE,
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
    }
    meta_path = out / "meta.json"
    import json

    meta_path.write_text(json.dumps(meta, indent=2))
    return {"train": str(train_path), "val": str(val_path), "meta": str(meta_path)}


def prepare_tinystories(out_dir: str | Path, max_docs: int = 50_000) -> dict[str, str]:
    """Optional: simpler English (faster, less vocabulary depth)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts: list[str] = []
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        texts.append(row["text"])
    # 90/10 split
    split = int(len(texts) * 0.9)
    train_ids = _tokenize_texts(texts[:split])
    val_ids = _tokenize_texts(texts[split:])

    train_path = out / "train.bin"
    val_path = out / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    meta = {
        "dataset": "TinyStories",
        "vocab_size": VOCAB_SIZE,
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "max_docs": max_docs,
    }
    import json

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"train": str(train_path), "val": str(val_path)}


class TokenBatchLoader:
    """Memory-mapped random contiguous batches (NanoGPT-style)."""

    def __init__(
        self,
        bin_path: str | Path,
        block_size: int,
        batch_size: int,
        device: "torch.device",
    ):
        import torch

        self._torch = torch
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self) -> tuple["torch.Tensor", "torch.Tensor"]:
        torch = self._torch
        ix = torch.randint(len(self.data) - self.block_size - 1, (self.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy(
                    (self.data[i : i + self.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        return x.to(self.device), y.to(self.device)
