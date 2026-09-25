"""Train the end-of-turn detector from random init: own tokenizer, small encoder.

The from-scratch lane. Trains a byte-level BPE tokenizer on the synthetic corpus
itself (bilingual, lowercased, accents kept), then a small BERT-config encoder
from random initialization. The classification recipe tracks train.py on purpose
(same loss with class weights, same schedule shape, same best-checkpoint
selection, same cost-optimal threshold sweep, same ONNX int8 export), so the
scratch-versus-pretrained comparison isolates pretraining, and all quality
comparisons happen on the external referees (gold set, dev set, OOD slice),
never on synthetic validation.

Two scars from the first run are load-bearing design here:

- Byte-level BPE, not WordPiece. The v1 WordPiece tokenizer learned only 1532
  pieces because a template corpus has few unique words, so gold-set text the
  templates never produced tokenized into walls of [UNK] and the model emitted
  a constant on it (gold PR-AUC 0.50 while synthetic val read 0.9998). Byte
  fallback makes every string representable, so unseen words still carry
  subword signal instead of vanishing.
- Template-grouped validation split. A row-level split puts different slot
  fills of the SAME template on both sides, which is memorization measured as
  generalization; v1's perfect val and its 0.07 "cost-optimal" threshold were
  both artifacts of that leak. Rows carry a tpl id from the generators and the
  split keeps each template entirely on one side. Rows without a tpl field
  each become their own group, which degrades to a row split for that file.

The saved directory is drop-in compatible with evaluate.py and serve.py
(AutoTokenizer loads the fast tokenizer files; the ONNX dirs carry
threshold.json). The threshold emitted here is picked on the grouped synthetic
validation split and threshold.json records that source; the shipped comparison
re-picks thresholds on the human dev set when it lands.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
)

from common import LABEL2ID, build_input, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the end-of-turn detector from random init.")
    parser.add_argument("--train", type=str, default="data/train_scaled.jsonl,data/train_es.jsonl", help="Comma-separated jsonl paths (rows should carry tpl ids)")
    parser.add_argument("--real", type=str, default="", help="Optional jsonl of real-call rows (e.g. data/ood_train.jsonl) mixed into training")
    parser.add_argument("--init-from", type=str, default="", help="Warm-start from our own MLM-pretrained base dir (models/eot-scratch-base); loads its tokenizer and encoder")
    parser.add_argument("--real-repeat", type=int, default=4, help="Upsample factor for real rows, applied to the TRAIN side only after the split")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of TEMPLATES held out for validation")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="models/eot-scratch")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TurnDataset(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_rows(paths: list[str]) -> tuple[list[str], list[int], list[str], list[str]]:
    """Load all jsonl files, dedup on (context, text) keeping first occurrence."""
    texts: list[str] = []
    labels: list[int] = []
    langs: list[str] = []
    tpls: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for i, r in enumerate(load_jsonl(path)):
            if r.get("label") not in LABEL2ID:
                continue
            key = (r.get("context", ""), r["text"])
            if key in seen:
                continue
            seen.add(key)
            texts.append(build_input(r.get("context", ""), r["text"]))
            labels.append(LABEL2ID[r["label"]])
            langs.append(r.get("lang", "en"))
            # Real-call rows group by their call id: a turn's full text and its prefixes
            # must never straddle the train/val split.
            tpls.append(r.get("tpl") or r.get("call") or f"{path}:{i}")
    return texts, labels, langs, tpls


def grouped_split(tpls: list[str], labels: list[int], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Hold out whole templates for validation, retrying the shuffle until both labels appear."""
    unique = sorted(set(tpls))
    for attempt in range(20):
        rng = random.Random(seed + attempt)
        shuffled = unique[:]
        rng.shuffle(shuffled)
        n_val = max(2, int(len(shuffled) * val_frac))
        val_tpls = set(shuffled[:n_val])
        val_idx = [i for i, t in enumerate(tpls) if t in val_tpls]
        train_idx = [i for i, t in enumerate(tpls) if t not in val_tpls]
        val_label_counts = Counter(labels[i] for i in val_idx)
        if len(val_label_counts) == 2 and min(val_label_counts.values()) >= 10:
            return train_idx, val_idx
    return train_idx, val_idx


def train_tokenizer(texts: list[str], vocab_size: int, max_len: int) -> PreTrainedTokenizerFast:
    """Byte-level BPE trained on the task corpus. Lowercase, accents kept, byte fallback for full coverage."""
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


def class_weights(labels: list[int], device: str) -> torch.Tensor:
    """Inverse-frequency weights so the rarer class does not get ignored by the loss."""
    counts = Counter(labels)
    total = len(labels)
    num_classes = len(LABEL2ID)
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_eval(model, loader: DataLoader, device: str) -> tuple[list[float], list[int]]:
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_probs = torch.softmax(outputs.logits, dim=-1)[:, LABEL2ID["speak"]]
            probs.extend(batch_probs.detach().cpu().tolist())
            labels.extend(batch_labels.detach().cpu().tolist())
    return probs, labels


def metrics_at(labels: list[int], probs: list[float], threshold: float) -> dict[str, float]:
    preds = [1 if p >= threshold else 0 for p in probs]
    precision = precision_score(labels, preds, pos_label=LABEL2ID["speak"], zero_division=0)
    recall = recall_score(labels, preds, pos_label=LABEL2ID["speak"], zero_division=0)
    return {"precision_speak": precision, "recall_speak": recall}


def export_onnx(out_dir: str) -> None:
    """Export the saved checkpoint to ONNX, then attempt dynamic int8 quantization."""
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    onnx_dir = f"{out_dir}-onnx"
    ort_model = ORTModelForSequenceClassification.from_pretrained(out_dir, export=True)
    ort_model.save_pretrained(onnx_dir)
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    tokenizer.save_pretrained(onnx_dir)
    print(f"fp32 onnx model written to {onnx_dir}")

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_dir = f"{out_dir}-onnx-int8"
        Path(int8_dir).mkdir(parents=True, exist_ok=True)
        quantize_dynamic(
            model_input=str(Path(onnx_dir) / "model.onnx"),
            model_output=str(Path(int8_dir) / "model.onnx"),
            weight_type=QuantType.QInt8,
        )
        for item in Path(onnx_dir).glob("*"):
            if item.name != "model.onnx":
                shutil.copy(item, Path(int8_dir) / item.name)
        print(f"int8 quantized model written to {int8_dir}")
    except Exception as exc:
        print(f"int8 quantization failed ({exc}), the fp32 onnx model at {onnx_dir} is still usable")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    paths = [p.strip() for p in args.train.split(",") if p.strip()]
    texts, labels, langs, tpls = load_rows(paths)
    n_synth = len(texts)
    if args.real:
        r_texts, r_labels, r_langs, r_tpls = load_rows([args.real])
        texts += r_texts
        labels += r_labels
        langs += r_langs
        tpls += r_tpls
        print(f"mixed in {len(r_texts)} real-call rows from {args.real}")
    print(f"loaded {len(texts)} unique samples from {len(paths)} files; langs: {dict(Counter(langs))}; templates: {len(set(tpls))}")

    train_idx, val_idx = grouped_split(tpls, labels, args.val_frac, args.seed)
    if args.real and args.real_repeat > 1:
        # Upsample real rows on the train side only; validation stays single-copy so its
        # metrics and the threshold sweep are not distorted by duplicates.
        extra = [i for i in train_idx if i >= n_synth] * (args.real_repeat - 1)
        train_idx = train_idx + extra
        n_real_val = sum(1 for i in val_idx if i >= n_synth)
        print(f"real rows in train {sum(1 for i in train_idx if i >= n_synth)} (upsampled x{args.real_repeat}), in val {n_real_val}")
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    val_langs = [langs[i] for i in val_idx]
    print(f"grouped split: {len(train_idx)} train rows, {len(val_idx)} val rows from held-out templates; val labels {dict(Counter(val_labels))}")

    if args.init_from:
        # Warm start from our own MLM-pretrained base: tokenizer and encoder come from
        # the pretrain dir, the classification head is fresh. The vocabulary is inherited
        # rather than retrained so the embedding table stays aligned with the tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(args.init_from)
        actual_vocab = len(tokenizer)
        print(f"tokenizer loaded from {args.init_from}: vocab {actual_vocab}")
        model = BertForSequenceClassification.from_pretrained(
            args.init_from,
            num_labels=2,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
        )
    else:
        tokenizer = train_tokenizer(train_texts, args.vocab_size, args.max_len)
        actual_vocab = tokenizer.backend_tokenizer.get_vocab_size()
        print(f"tokenizer trained: vocab {actual_vocab}")

        config = BertConfig(
            vocab_size=actual_vocab,
            hidden_size=args.hidden,
            num_hidden_layers=args.layers,
            num_attention_heads=args.heads,
            intermediate_size=args.intermediate,
            max_position_embeddings=args.max_len,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
            num_labels=2,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = BertForSequenceClassification(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model built from random init: {n_params / 1e6:.2f}M parameters")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    train_encodings = tokenizer(train_texts, truncation=True, padding="max_length", max_length=args.max_len)
    val_encodings = tokenizer(val_texts, truncation=True, padding="max_length", max_length=args.max_len)
    train_loader = DataLoader(TurnDataset(train_encodings, train_labels), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TurnDataset(val_encodings, val_labels), batch_size=args.batch, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train_labels, device))

    epoch_reports = []
    best_pr_auc = -1.0
    best_probs: list[float] = []
    best_labels: list[int] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, batch_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        val_probs, val_labels_out = run_eval(model, val_loader, device)
        val_preds = [1 if p >= 0.5 else 0 for p in val_probs]
        accuracy = sum(int(p == l) for p, l in zip(val_preds, val_labels_out)) / len(val_labels_out)
        at_half = metrics_at(val_labels_out, val_probs, 0.5)
        pr_auc = average_precision_score(val_labels_out, val_probs)

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_accuracy={accuracy:.4f} "
            f"val_precision_speak={at_half['precision_speak']:.4f} val_recall_speak={at_half['recall_speak']:.4f} "
            f"val_pr_auc={pr_auc:.4f}"
        )

        epoch_reports.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": accuracy,
                "val_precision_speak": at_half["precision_speak"],
                "val_recall_speak": at_half["recall_speak"],
                "val_pr_auc": pr_auc,
            }
        )

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_probs = val_probs
            best_labels = val_labels_out
            Path(args.out).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)

    # Per-language validation PR-AUC for the best checkpoint's stored probabilities.
    per_lang = {}
    for lang in sorted(set(val_langs)):
        lang_probs = [p for p, lg in zip(best_probs, val_langs) if lg == lang]
        lang_labels = [l for l, lg in zip(best_labels, val_langs) if lg == lang]
        if len(set(lang_labels)) == 2:
            per_lang[lang] = {"n": len(lang_labels), "pr_auc": average_precision_score(lang_labels, lang_probs)}

    # Cost-optimal operating threshold on the grouped validation curve, cost = 5*FPR + FNR.
    n_wait = sum(1 for l in best_labels if l == 0) or 1
    n_speak = sum(1 for l in best_labels if l == 1) or 1
    best_thr, best_cost = 0.5, float("inf")
    for i in range(1, 99):
        thr = i / 100
        fpr = sum(1 for p, l in zip(best_probs, best_labels) if p >= thr and l == 0) / n_wait
        fnr = sum(1 for p, l in zip(best_probs, best_labels) if p < thr and l == 1) / n_speak
        cost = 5 * fpr + fnr
        if cost < best_cost or (cost == best_cost and thr < best_thr):
            best_cost, best_thr = cost, thr
    thr_payload = {
        "threshold": best_thr,
        "cost_ratio": 5,
        "method": "cost-optimal on grouped validation curve, cost = 5*FPR + FNR (class-conditional rates, imbalance-invariant)",
        "selection_source": "synthetic-val (template-grouped); re-pick on data/dev_set.json when the human dev set lands",
        "expected_val_cost": best_cost,
    }
    with open(Path(args.out) / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(thr_payload, f, indent=2)
    print(f"cost-optimal operating threshold: {best_thr} (val cost {best_cost})")

    op_metrics = metrics_at(best_labels, best_probs, best_thr)
    report = {
        "args": vars(args),
        "n_samples": len(texts),
        "n_params": n_params,
        "vocab_size": actual_vocab,
        "epochs": epoch_reports,
        "best_val_pr_auc": best_pr_auc,
        "val_pr_auc_per_lang": per_lang,
        "operating_threshold": best_thr,
        "operating_precision_speak": op_metrics["precision_speak"],
        "operating_recall_speak": op_metrics["recall_speak"],
    }
    with open(Path(args.out) / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"best checkpoint saved to {args.out} (val_pr_auc={best_pr_auc:.4f})")

    export_onnx(args.out)

    for suffix in ("-onnx", "-onnx-int8"):
        target_dir = Path(f"{args.out}{suffix}")
        if target_dir.exists():
            shutil.copy(Path(args.out) / "threshold.json", target_dir / "threshold.json")


if __name__ == "__main__":
    main()
