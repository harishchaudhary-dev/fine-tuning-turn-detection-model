"""Masked-language-model pretraining for the from-scratch lane.

The scratch experiment isolated its own missing ingredient: general language
knowledge. A task-fitted model owned the synthetic distribution (0.99) and
coin-flipped unseen human phrasing (0.50) even after tokenizer coverage and the
leaked validation split were fixed. This script supplies the ingredient the
from-scratch way, a short self-supervised MLM pass on open text plus the task's
own corpus, so the lane's story completes: random init, then our own pretrain,
then the comparison against a 66M internet-pretrained model, all measured.

Sized for a laptop and a 3.7M-parameter encoder: tens of megabytes of text and
minutes of MPS time, not GPUs and days. The tokenizer is trained here, on the
combined corpus, and the fine-tune stage inherits it via --init-from so the
embedding table and vocabulary stay consistent end to end.

Corpus: Wikipedia slices fetched by fetch_pretrain_corpus.py (CC BY-SA), plus
the synthetic task corpus and the real-call training slice mixed in for spoken
register. Outputs a standard Hugging Face model dir (BertForMaskedLM weights,
fast tokenizer files) that BertForSequenceClassification.from_pretrained can
load with a fresh classification head.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
)

from common import build_input, load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLM-pretrain the scratch encoder on open text plus the task corpus.")
    p.add_argument("--corpus", type=str, default="data/pretrain_en.txt,data/pretrain_es.txt", help="Comma-separated plain-text files, one paragraph per line")
    p.add_argument("--task-train", type=str, default="data/train_scaled.jsonl,data/train_es.jsonl", help="Task jsonl files mixed into the pretraining text")
    p.add_argument("--real", type=str, default="data/ood_train.jsonl", help="Real-call jsonl mixed in (empty string to skip)")
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--intermediate", type=int, default=1024)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="models/eot-scratch-base")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    for path in [p.strip() for p in args.corpus.split(",") if p.strip()]:
        with open(path, encoding="utf-8") as f:
            texts.extend(line.strip() for line in f if line.strip())
    n_corpus = len(texts)
    task_paths = [p.strip() for p in args.task_train.split(",") if p.strip()]
    if args.real:
        task_paths.append(args.real)
    for path in task_paths:
        for r in load_jsonl(path):
            texts.append(build_input(r.get("context", ""), r["text"]))
    print(f"pretraining text: {n_corpus} corpus paragraphs + {len(texts) - n_corpus} task lines")
    return texts


def train_tokenizer(texts: list[str], vocab_size: int, max_len: int) -> PreTrainedTokenizerFast:
    """Byte-level BPE trained on the combined corpus. Lowercase, accents kept, byte fallback."""
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=max_len,
    )


def build_blocks(texts: list[str], tokenizer: PreTrainedTokenizerFast, max_len: int) -> list[dict]:
    """Tokenize every line and split long ones into max_len blocks; drop tiny fragments."""
    blocks: list[dict] = []
    body = max_len - 2
    enc = tokenizer(texts, add_special_tokens=False)["input_ids"]
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    for ids in enc:
        for start in range(0, len(ids), body):
            piece = ids[start : start + body]
            if len(piece) < 16:
                continue
            blocks.append({"input_ids": [cls_id] + piece + [sep_id]})
    return blocks


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    texts = load_texts(args)
    tokenizer = train_tokenizer(texts, args.vocab_size, args.max_len)
    actual_vocab = tokenizer.backend_tokenizer.get_vocab_size()
    print(f"tokenizer trained: vocab {actual_vocab}")

    blocks = build_blocks(texts, tokenizer, args.max_len)
    print(f"{len(blocks)} pretraining blocks of up to {args.max_len} tokens")

    config = BertConfig(
        vocab_size=actual_vocab,
        hidden_size=args.hidden,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate,
        max_position_embeddings=args.max_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLM model from random init: {n_params / 1e6:.2f}M parameters")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)
    loader = DataLoader(blocks, batch_size=args.batch, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            running += out.loss.item()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{total_steps} loss {running / 200:.4f} ({step / (time.time() - t0):.1f} it/s)")
                running = 0.0
        print(f"epoch {epoch} done")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"pretrained base saved to {args.out}")


if __name__ == "__main__":
    main()