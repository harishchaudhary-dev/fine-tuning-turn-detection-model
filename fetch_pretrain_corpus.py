"""Fetch a small, license-clean bilingual text corpus for scratch-lane pretraining.

The from-scratch experiment showed the model's gap is general language knowledge,
not task signal, so the completion of the from-scratch story is pretraining our
own: a short masked-language-model pass on open text before task fine-tuning.
This script builds that corpus, sized for a 3.7M-parameter model on a laptop,
tens of megabytes, not tens of gigabytes.

Sources, both CC BY-SA Wikipedia dumps streamed from the Hugging Face hub so
nothing large lands on disk beyond the slice itself: Simple English Wikipedia
for English (plain register, closer to speech than full enwiki), and Spanish
Wikipedia for the bilingual half. The task's own synthetic corpus and the
real-call training slice get mixed in at pretrain time by the trainer, not here.

Needs the `datasets` package (pip install datasets), which is deliberately NOT
in requirements.txt; only this fetch step uses it, and the serving image should
not carry it. Outputs are plain text, one paragraph per line, gitignored (they
are derived artifacts, regenerable by rerunning this).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasets import load_dataset

WS = re.compile(r"\s+")


def clean(text: str) -> list[str]:
    """Article text to paragraph lines: squeeze whitespace, drop headings and stubs."""
    out = []
    for para in text.split("\n"):
        para = WS.sub(" ", para).strip()
        if len(para) < 80 or para.startswith("=="):
            continue
        out.append(para)
    return out


def fetch(config: str, target_mb: float, out_path: str) -> None:
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    target = int(target_mb * 1024 * 1024)
    written = 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for article in ds:
            for line in clean(article.get("text", "")):
                f.write(line + "\n")
                written += len(line) + 1
            if written >= target:
                break
    print(f"{out_path}: {written / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--en-mb", type=float, default=16.0)
    ap.add_argument("--es-mb", type=float, default=8.0)
    ap.add_argument("--out-en", default="data/pretrain_en.txt")
    ap.add_argument("--out-es", default="data/pretrain_es.txt")
    args = ap.parse_args()

    fetch("20231101.simple", args.en_mb, args.out_en)
    fetch("20231101.es", args.es_mb, args.out_es)


if __name__ == "__main__":
    main()