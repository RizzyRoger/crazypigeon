#!/usr/bin/env python3
"""Download and tokenize training data."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_tinystories, prepare_wikitext2
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tokenized dataset")
    parser.add_argument(
        "--dataset",
        choices=["wikitext2", "tinystories"],
        default="wikitext2",
        help="wikitext2 (recommended) or tinystories (simpler/faster)",
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = ROOT / cfg["paths"]["data_dir"]
    if args.dataset == "wikitext2":
        paths = prepare_wikitext2(out)
    else:
        paths = prepare_tinystories(out)
    print("Prepared:", paths)


if __name__ == "__main__":
    main()
