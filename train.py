"""Fine-tune a small encoder for end-of-turn detection (binary: wait vs speak)."""

import argparse
import gc
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from common import LABEL2ID, build_input, load_jsonl
from pick_threshold import sweep

FINAL_THRESHOLD = 0.833


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the end-of-turn detector.")
    parser.add_argument("--train", type=str, default="data/train.jsonl", required=True, help="Path to training jsonl file")
    parser.add_argument("--extra", type=str, default="", help="Optional extra jsonl file folded into training (e.g. real-call OOD data); rows carry context/text/label like --train")
    parser.add_argument("--extra-upsample", type=int, default=4, help="Train-side duplication factor for rows sourced from --extra, applied after the split, never on validation")
    parser.add_argument("--base", type=str, default="distilbert-base-uncased", help="Base encoder checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="models/eot-distilbert")
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


def load_rows(path: str) -> list[dict]:
    """Load a jsonl file's rows, filtered to those carrying a recognized label.

    Rows keep every source field untouched (context/text/label at minimum; tpl, call, cls,
    variant, lang pass through when present), so callers can build group keys and origin tags
    from them before ever converting to model-input strings.
    """
    rows = load_jsonl(path)
    return [r for r in rows if r.get("label") in LABEL2ID]


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


def score_dev_set(model, tokenizer, device: str, max_len: int, dev_path: str | Path) -> tuple[list[float], list[int]]:
    """Score P(speak) via build_input for every dev_set.json sample whose label is speak or wait."""
    with open(dev_path, encoding="utf-8") as f:
        dev = json.load(f)
    samples = [s for s in dev.get("samples", []) if s.get("label") in LABEL2ID]
    dev_texts = [build_input(s.get("context", ""), s["text"]) for s in samples]
    dev_labels = [LABEL2ID[s["label"]] for s in samples]
    dev_encodings = tokenizer(dev_texts, truncation=True, padding="max_length", max_length=max_len)
    dev_loader = DataLoader(TurnDataset(dev_encodings, dev_labels), batch_size=32, shuffle=False)
    return run_eval(model, dev_loader, device)


def score_texts(model, tokenizer, device: str, max_len: int, contexts: list[str], texts: list[str]) -> list[float]:
    """Score P(speak) for raw context/text pairs with the given model, batched, no labels required.

    Used for guardrail constraint rows, which carry a required decision rather than a ground-truth
    label to average over.
    """
    model.eval()
    inputs = [build_input(c, t) for c, t in zip(contexts, texts)]
    probs: list[float] = []
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            chunk = inputs[start : start + batch_size]
            encoded = tokenizer(chunk, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            batch_probs = torch.softmax(outputs.logits, dim=-1)[:, LABEL2ID["speak"]]
            probs.extend(batch_probs.detach().cpu().tolist())
    return probs


def load_tier1_constraint_rows(gold_path: str | Path, dev_path: str | Path, regressions_path: str | Path) -> list[dict]:
    """Guardrail rows enforcing the published tier-1 gates from EVALS.md: every F-class gold
    card (never interrupt a readout) and gold H5 (announced continuation holds) must predict
    wait; the dev H-class judgment card dH4 (the same announced-continuation gate) must predict
    wait; and every probe-found case in regressions.jsonl must predict its recorded label.
    """
    rows: list[dict] = []

    gold_path = Path(gold_path)
    if gold_path.exists():
        gold = json.load(open(gold_path, encoding="utf-8"))
        for s in gold.get("samples", []):
            if s.get("cls") == "F" or s.get("id") == "H5":
                rows.append({"context": s.get("context", ""), "text": s["text"], "label": "wait"})

    dev_path = Path(dev_path)
    if dev_path.exists():
        dev = json.load(open(dev_path, encoding="utf-8"))
        for s in dev.get("samples", []):
            if s.get("id") == "dH4":
                rows.append({"context": s.get("context", ""), "text": s["text"], "label": "wait"})

    regressions_path = Path(regressions_path)
    if regressions_path.exists():
        for r in load_jsonl(regressions_path):
            rows.append({"context": r.get("context", ""), "text": r["text"], "label": r["label"]})

    return rows


def export_onnx(out_dir: str) -> None:
    """Export the saved checkpoint to ONNX, then attempt dynamic int8 quantization."""
    from optimum.onnxruntime import ORTModelForSequenceClassification

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

    main_rows = load_rows(args.train)
    for r in main_rows:
        r["_origin"] = "main"

    extra_rows: list[dict] = []
    if args.extra:
        extra_rows = load_rows(args.extra)
        before_policy_filter = len(extra_rows)
        extra_rows = [
            r
            for r in extra_rows
            if not (r.get("label") == "speak" and ("one more thing" in r["text"].lower() or "hold that thought" in r["text"].lower()))
        ]
        print(f"policy-filtered {before_policy_filter - len(extra_rows)} vendor-behavior rows from extra")
        for r in extra_rows:
            r["_origin"] = "extra"

    all_rows = main_rows + extra_rows

    # Group-aware split: a group never straddles train/val, so every rendering of one synthetic
    # template ("tpl") or every turn from one real call ("call") lands on a single side. This
    # replaces the old row-level stratified split; stratification is now approximate rather than
    # exact, since a whole group carries whatever label mix it happens to have, not a 50/50 draw.
    group_keys = [r.get("call") or r.get("tpl") or f"__row{i}" for i, r in enumerate(all_rows)]
    group_sizes = Counter(group_keys)
    unique_groups = list(dict.fromkeys(group_keys))
    split_rng = random.Random(args.seed)
    split_rng.shuffle(unique_groups)

    target_val_n = args.val_frac * len(all_rows)
    val_groups: set[str] = set()
    val_n = 0
    for g in unique_groups:
        if val_n >= target_val_n:
            break
        val_groups.add(g)
        val_n += group_sizes[g]

    train_rows = [r for r, k in zip(all_rows, group_keys) if k not in val_groups]
    val_rows = [r for r, k in zip(all_rows, group_keys) if k in val_groups]

    # Upsample --extra rows on the train side only, so real-call OOD data pulls its weight in
    # gradient updates without ever duplicating a row into the held-out validation split.
    if args.extra:
        upsampled: list[dict] = []
        for r in train_rows:
            if r.get("_origin") == "extra":
                upsampled.extend([r] * args.extra_upsample)
            else:
                upsampled.append(r)
        train_rows = upsampled

    train_texts = [build_input(r.get("context", ""), r["text"]) for r in train_rows]
    train_labels = [LABEL2ID[r["label"]] for r in train_rows]
    val_texts = [build_input(r.get("context", ""), r["text"]) for r in val_rows]
    val_labels = [LABEL2ID[r["label"]] for r in val_rows]

    print(
        f"loaded {len(main_rows)} rows from {args.train}"
        + (f" + {len(extra_rows)} rows from {args.extra} (upsampled x{args.extra_upsample} on train side)" if args.extra else "")
        + f"; groups: {len(unique_groups)}; train rows: {len(train_rows)}; val rows: {len(val_rows)}"
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2)
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

    final_at_threshold = metrics_at(best_labels, best_probs, FINAL_THRESHOLD)
    report = {
        "args": vars(args),
        "epochs": epoch_reports,
        "best_val_pr_auc": best_pr_auc,
        "final_val_precision_at_threshold": final_at_threshold["precision_speak"],
        "final_val_recall_at_threshold": final_at_threshold["recall_speak"],
        "final_threshold": FINAL_THRESHOLD,
    }
    with open(Path(args.out) / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"best checkpoint saved to {args.out} (val_pr_auc={best_pr_auc:.4f})")

    dev_set_path = Path("data/dev_set.json")
    if dev_set_path.exists():
        best_model = AutoModelForSequenceClassification.from_pretrained(args.out)
        best_model.to(device)
        best_tokenizer = AutoTokenizer.from_pretrained(args.out)
        dev_probs, dev_labels = score_dev_set(best_model, best_tokenizer, device, args.max_len, dev_set_path)

        guardrail_rows = load_tier1_constraint_rows("data/gold_set.json", dev_set_path, "data/regressions.jsonl")
        guardrail_probs = score_texts(
            best_model, best_tokenizer, device, args.max_len,
            [r["context"] for r in guardrail_rows], [r["text"] for r in guardrail_rows],
        )
        constraints = list(zip(guardrail_probs, [r["label"] for r in guardrail_rows]))

        result = sweep(dev_probs, dev_labels, cost_ratio=5, constraints=constraints)
        best_thr, best_cost = result["threshold"], result["cost"]
        thr_payload = {
            "threshold": best_thr,
            "cost_ratio": 5,
            "method": "cost-optimal with tier-1 guardrail constraints",
            "selection_source": "data/dev_set.json",
            "dev_n": len(dev_labels),
            "admissible_count": result["admissible_count"],
            "constraints_note": (
                "constraints enforce the published tier-1 gates from EVALS.md: every F-class gold card "
                "(never interrupt a readout), gold H5 and dev dH4 (announced continuation holds), and "
                "every regressions.jsonl case must land on its required decision"
            ),
        }
        if result["tier1_unsatisfiable"]:
            thr_payload["tier1_unsatisfiable"] = True
        print(
            f"cost-optimal operating threshold from dev set (n={len(dev_labels)}) with "
            f"{len(constraints)} guardrail constraints: {best_thr} (dev cost {best_cost:.4f}, "
            f"admissible {result['admissible_count']}, tier1_unsatisfiable={result['tier1_unsatisfiable']})"
        )
        del best_model, best_tokenizer, dev_probs, dev_labels
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    else:
        result = sweep(best_probs, best_labels, cost_ratio=5)
        best_thr, best_cost = result["threshold"], result["cost"]
        thr_payload = {
            "threshold": best_thr,
            "cost_ratio": 5,
            "method": "cost-optimal on validation curve, cost = 5*FPR + FNR (class-conditional rates, imbalance-invariant)",
            "expected_val_cost": best_cost,
            "selection_source": "validation split fallback (data/dev_set.json not found)",
        }
        print(f"cost-optimal operating threshold from validation fallback: {best_thr} (val cost {best_cost:.4f})")
    with open(Path(args.out) / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(thr_payload, f, indent=2)

    op_metrics = metrics_at(best_labels, best_probs, best_thr)
    report["operating_threshold"] = best_thr
    report["operating_precision_speak"] = op_metrics["precision_speak"]
    report["operating_recall_speak"] = op_metrics["recall_speak"]
    with open(Path(args.out) / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    export_onnx(args.out)

    for suffix in ("-onnx", "-onnx-int8"):
        target_dir = Path(f"{args.out}{suffix}")
        if target_dir.exists():
            shutil.copy(Path(args.out) / "threshold.json", target_dir / "threshold.json")


if __name__ == "__main__":
    main()
