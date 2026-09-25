"""Evaluate an exported end-of-turn detector against the frozen gold set or a jsonl file."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

import common
from common import LABEL2ID, build_input, load_gold, load_jsonl

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.833, 0.9, 0.95]
BOUNDARY_BAND = (0.4, 0.9)
PREFIX_BATCH_SIZE = 64
EARLY_COMMIT_FRACTION = 0.6
CALIBRATION_BINS = 10
SHORT_UTTERANCE_WORDS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the end-of-turn detector.")
    parser.add_argument("--model", type=str, default="models/eot-distilbert-onnx-int8", help="Dir with model.onnx and tokenizer files")
    parser.add_argument("--data", type=str, default="data/gold_set.json", help="Path to gold_set.json or a jsonl file")
    parser.add_argument("--report", type=str, default="eval_report.json")
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def load_session_and_tokenizer(model_dir: str):
    import onnxruntime as ort

    session = ort.InferenceSession(str(Path(model_dir) / "model.onnx"), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return session, tokenizer


def predict_speak_probs(session, tokenizer, contexts: list[str], texts: list[str]) -> list[float]:
    input_names = {i.name for i in session.get_inputs()}
    probs = []
    for context, text in zip(contexts, texts):
        model_input = build_input(context, text)
        encoded = tokenizer(model_input, truncation=True, padding="max_length", max_length=128, return_tensors="np")
        onnx_inputs = {name: value for name, value in encoded.items() if name in input_names}
        logits = session.run(None, onnx_inputs)[0]
        probs.append(float(softmax(logits)[0][LABEL2ID["speak"]]))
    return probs


def predict_speak_probs_batched(session, tokenizer, model_inputs: list[str], batch_size: int = PREFIX_BATCH_SIZE) -> list[float]:
    """Same as predict_speak_probs but takes already-built model input strings and runs them batched in chunks."""
    input_names = {i.name for i in session.get_inputs()}
    probs: list[float] = []
    for start in range(0, len(model_inputs), batch_size):
        chunk = model_inputs[start : start + batch_size]
        encoded = tokenizer(chunk, truncation=True, padding="max_length", max_length=128, return_tensors="np")
        onnx_inputs = {name: value for name, value in encoded.items() if name in input_names}
        logits = session.run(None, onnx_inputs)[0]
        chunk_probs = softmax(logits)[:, LABEL2ID["speak"]]
        probs.extend(float(p) for p in chunk_probs)
    return probs


def rate_of(labels: list[int], preds: list[int], true_value: int, pred_value: int) -> float:
    subset = [p for label, p in zip(labels, preds) if label == true_value]
    if not subset:
        return 0.0
    return sum(1 for p in subset if p == pred_value) / len(subset)


def sweep_thresholds(labels: list[int], probs: list[float], extra_thresholds: list[float] | None = None) -> list[dict]:
    thresholds = list(THRESHOLDS)
    for extra in extra_thresholds or []:
        if extra not in thresholds:
            thresholds.append(extra)
    thresholds.sort()
    rows = []
    for threshold in thresholds:
        preds = [1 if p >= threshold else 0 for p in probs]
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(labels, preds, pos_label=LABEL2ID["speak"], zero_division=0),
                "recall": recall_score(labels, preds, pos_label=LABEL2ID["speak"], zero_division=0),
                "f1": f1_score(labels, preds, pos_label=LABEL2ID["speak"], zero_division=0),
                "false_speak_rate": rate_of(labels, preds, LABEL2ID["wait"], LABEL2ID["speak"]),
                "false_wait_rate": rate_of(labels, preds, LABEL2ID["speak"], LABEL2ID["wait"]),
            }
        )
    return rows


def per_class_accuracy(samples: list[dict], probs: list[float], labels: list[int], threshold: float) -> dict[str, float]:
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for sample, prob, label in zip(samples, probs, labels):
        cls = sample.get("cls", "?")
        pred = 1 if prob >= threshold else 0
        total[cls] += 1
        if pred == label:
            correct[cls] += 1
    return {cls: correct[cls] / total[cls] for cls in sorted(total)}


def expected_calibration_error(labels: list[int], probs: list[float], num_bins: int = CALIBRATION_BINS) -> float | None:
    """Equal-width-bin ECE over P(speak): sum over bins of (n_bin/N) * abs(mean_conf_bin - frac_true_speak_bin)."""
    n = len(probs)
    if n == 0:
        return None
    edges = [i / num_bins for i in range(num_bins + 1)]
    ece = 0.0
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == num_bins - 1:
            bin_items = [(p, l) for p, l in zip(probs, labels) if lo <= p <= hi]
        else:
            bin_items = [(p, l) for p, l in zip(probs, labels) if lo <= p < hi]
        if not bin_items:
            continue
        bin_probs = [p for p, _ in bin_items]
        bin_labels = [l for _, l in bin_items]
        mean_conf = sum(bin_probs) / len(bin_probs)
        frac_true_speak = sum(bin_labels) / len(bin_labels)
        ece += (len(bin_items) / n) * abs(mean_conf - frac_true_speak)
    return ece


def prefix_stability_report(session, tokenizer, hard: list[dict], threshold: float) -> dict:
    """Replay each gold hard sample word by word (streaming ASR simulation) and measure decision stability."""
    flat_inputs: list[str] = []
    spans: list[tuple[int, int]] = []  # (start index into flat_inputs, number of prefixes) per sample
    words_per_sample: list[list[str]] = []

    for sample in hard:
        words = sample["text"].split()
        words_per_sample.append(words)
        spans.append((len(flat_inputs), len(words)))
        context = sample.get("context", "")
        for k in range(1, len(words) + 1):
            flat_inputs.append(build_input(context, " ".join(words[:k])))

    flat_probs = predict_speak_probs_batched(session, tokenizer, flat_inputs) if flat_inputs else []

    flips_per_utterance: list[int] = []
    early_commit_flags: list[bool] = []
    commit_miss_flags: list[bool] = []
    would_interrupt_flags: list[bool] = []

    for sample, words, (start, count) in zip(hard, words_per_sample, spans):
        if count == 0:
            continue
        probs = flat_probs[start : start + count]
        decisions = [1 if p >= threshold else 0 for p in probs]
        flips = sum(1 for a, b in zip(decisions, decisions[1:]) if a != b)
        flips_per_utterance.append(flips)

        first_commit = next((k for k, p in enumerate(probs, start=1) if p >= threshold), None)
        if sample.get("label") == "speak":
            commit_miss_flags.append(first_commit is None)
            early_commit_flags.append(first_commit is not None and first_commit < EARLY_COMMIT_FRACTION * count)
        elif sample.get("label") == "wait":
            would_interrupt_flags.append(first_commit is not None)

    def rate(flags: list[bool]) -> float | None:
        return (sum(1 for f in flags if f) / len(flags)) if flags else None

    return {
        "n_samples": len(hard),
        "mean_flips": (sum(flips_per_utterance) / len(flips_per_utterance)) if flips_per_utterance else None,
        "max_flips": max(flips_per_utterance) if flips_per_utterance else None,
        "early_commit_rate_true_speak": rate(early_commit_flags),
        "commit_miss_rate_true_speak": rate(commit_miss_flags),
        "prefix_crossing_rate_true_wait": rate(would_interrupt_flags),
    }


def recall_on_mask(labels: list[int], probs: list[float], threshold: float, mask: list[bool]) -> float | None:
    sub_labels = [l for l, m in zip(labels, mask) if m]
    if not sub_labels:
        return None
    sub_probs = [p for p, m in zip(probs, mask) if m]
    preds = [1 if p >= threshold else 0 for p in sub_probs]
    return recall_score(sub_labels, preds, pos_label=LABEL2ID["speak"], zero_division=0)


def compute_slices(hard: list[dict], hard_labels: list[int], hard_probs: list[float], threshold: float) -> dict:
    has_context = [bool(s.get("context", "").strip()) for s in hard]
    no_context = [not m for m in has_context]
    short_utterance = [len(s["text"].split()) <= SHORT_UTTERANCE_WORDS for s in hard]

    return {
        "recall_overall": recall_on_mask(hard_labels, hard_probs, threshold, [True] * len(hard_labels)),
        "recall_with_context": recall_on_mask(hard_labels, hard_probs, threshold, has_context),
        "recall_without_context": recall_on_mask(hard_labels, hard_probs, threshold, no_context),
        "recall_short_utterance": recall_on_mask(hard_labels, hard_probs, threshold, short_utterance),
        "short_utterance_max_words": SHORT_UTTERANCE_WORDS,
    }


def print_sweep_report(section: dict) -> None:
    print(f"n={section['n']} pr_auc={section['pr_auc']}")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'false_speak':>12} {'false_wait':>11}")
    for row in section["threshold_sweep"]:
        print(
            f"{row['threshold']:>10} {row['precision']:>10.4f} {row['recall']:>10.4f} "
            f"{row['f1']:>10.4f} {row['false_speak_rate']:>12.4f} {row['false_wait_rate']:>11.4f}"
        )
    if "per_class_accuracy_at_threshold" in section:
        print(f"per-class accuracy at threshold {section['threshold_for_per_class']}:")
        for cls, acc in section["per_class_accuracy_at_threshold"].items():
            print(f"  {cls}: {acc:.4f}")
    if "ece" in section:
        print(f"ECE ({CALIBRATION_BINS} bins) on gold hard set: {section['ece']}")


def print_boundary_report(section: dict) -> None:
    print(
        f"boundary n={section['n']} mean_p_speak={section['mean_p_speak']} "
        f"count_in_band={section['count_in_band']} band={section['band']}"
    )


def print_prefix_stability_report(section: dict) -> None:
    print(
        f"prefix stability n={section['n_samples']} mean_flips={section['mean_flips']} max_flips={section['max_flips']} "
        f"early_commit_rate(true speak)={section['early_commit_rate_true_speak']} "
        f"commit_miss_rate(true speak)={section['commit_miss_rate_true_speak']} "
        f"prefix_crossing_rate(true wait)={section['prefix_crossing_rate_true_wait']}"
    )


def print_slices_report(section: dict) -> None:
    print(
        f"slices at operating threshold: recall_overall={section['recall_overall']} "
        f"recall_with_context={section['recall_with_context']} recall_without_context={section['recall_without_context']} "
        f"recall_short_utterance(<={section['short_utterance_max_words']}w)={section['recall_short_utterance']}"
    )


def evaluate_gold(session, tokenizer, path: str, threshold: float) -> dict:
    gold = load_gold(path)
    samples = gold.get("samples", [])
    hard = [s for s in samples if s.get("label") in LABEL2ID]
    boundary = [s for s in samples if s.get("label") == "unsure"]

    hard_probs = predict_speak_probs(session, tokenizer, [s.get("context", "") for s in hard], [s["text"] for s in hard])
    hard_labels = [LABEL2ID[s["label"]] for s in hard]

    hard_section = {
        "n": len(hard),
        "pr_auc": average_precision_score(hard_labels, hard_probs) if hard_labels else None,
        "threshold_sweep": sweep_thresholds(hard_labels, hard_probs, extra_thresholds=[threshold]),
        "per_class_accuracy_at_threshold": per_class_accuracy(hard, hard_probs, hard_labels, threshold),
        "threshold_for_per_class": threshold,
        "ece": expected_calibration_error(hard_labels, hard_probs),
    }

    if boundary:
        boundary_probs = predict_speak_probs(
            session, tokenizer, [s.get("context", "") for s in boundary], [s["text"] for s in boundary]
        )
        mean_p_speak = sum(boundary_probs) / len(boundary_probs)
        count_in_band = sum(1 for p in boundary_probs if BOUNDARY_BAND[0] < p < BOUNDARY_BAND[1])
    else:
        mean_p_speak = None
        count_in_band = 0

    boundary_section = {
        "n": len(boundary),
        "mean_p_speak": mean_p_speak,
        "count_in_band": count_in_band,
        "band": list(BOUNDARY_BAND),
    }

    prefix_section = prefix_stability_report(session, tokenizer, hard, threshold)
    slices_section = compute_slices(hard, hard_labels, hard_probs, threshold)

    print_sweep_report(hard_section)
    print_boundary_report(boundary_section)
    print_prefix_stability_report(prefix_section)
    print_slices_report(slices_section)

    return {
        "hard": hard_section,
        "boundary": boundary_section,
        "prefix_stability": prefix_section,
        "slices": slices_section,
    }


def evaluate_jsonl(session, tokenizer, path: str, threshold: float) -> dict:
    rows = [r for r in load_jsonl(path) if r.get("label") in LABEL2ID]
    probs = predict_speak_probs(session, tokenizer, [r.get("context", "") for r in rows], [r["text"] for r in rows])
    labels = [LABEL2ID[r["label"]] for r in rows]

    section = {
        "n": len(rows),
        "pr_auc": average_precision_score(labels, probs) if labels else None,
        "threshold_sweep": sweep_thresholds(labels, probs, extra_thresholds=[threshold]),
    }
    print_sweep_report(section)
    return {"jsonl": section}


def main() -> None:
    args = parse_args()
    session, tokenizer = load_session_and_tokenizer(args.model)

    threshold = common.load_threshold(args.model)
    threshold_path = Path(args.model) / "threshold.json"
    threshold_source = str(threshold_path) if threshold_path.exists() else "gold-set default (no threshold.json in model dir)"
    print(f"operating threshold: {threshold} (source: {threshold_source})")

    is_gold = args.data.endswith(".json") and not args.data.endswith(".jsonl")
    report: dict = {
        "model": args.model,
        "data": args.data,
        "operating_threshold": threshold,
        "operating_threshold_source": threshold_source,
    }
    if is_gold:
        report.update(evaluate_gold(session, tokenizer, args.data, threshold))
    else:
        report.update(evaluate_jsonl(session, tokenizer, args.data, threshold))

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"report written to {args.report}")


if __name__ == "__main__":
    main()