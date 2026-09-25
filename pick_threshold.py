"""Shared cost-optimal threshold selection, with optional tier-1 guardrail constraints.

One implementation of the sweep, used two ways: imported (`sweep`) by train.py so the
in-training selection and this CLI can never drift apart, and run standalone against an
already-exported model dir plus a dev-set-shaped labels file.

The guardrail mechanism exists because the plain cost sweep can collapse the threshold to
protect false-speaks in aggregate while breaking a specific tier-1 gate from EVALS.md (an
announced-continuation card scoring high enough that no ordinary threshold keeps it a wait).
Constraints let the caller pin down cards that must land on a required decision; the sweep then
optimizes cost only over thresholds that keep every pinned card correct, and says plainly when
no such threshold exists rather than silently picking one that violates a guardrail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import LABEL2ID, build_input, load_jsonl


def sweep(probs: list[float], labels: list[int], cost_ratio: float = 5, constraints: list[tuple[float, str]] | None = None) -> dict:
    """Sweep thresholds i/100 for i in 1..99, minimizing cost = cost_ratio*FPR + FNR
    (class-conditional rates, imbalance-invariant); ties prefer the lower threshold.

    constraints, when given, is a list of (prob, required_decision) pairs where
    required_decision is "speak" or "wait". A threshold t is admissible only if every pair
    satisfies (prob >= t) == (required == "speak"). The min-cost admissible threshold wins; if
    no threshold is admissible, the unconstrained optimum is returned with tier1_unsatisfiable
    set True so the caller can fail loud instead of silently shipping a broken guardrail.

    Returns dict(threshold, cost, admissible_count, tier1_unsatisfiable).
    """
    n_wait = sum(1 for l in labels if l == 0) or 1
    n_speak = sum(1 for l in labels if l == 1) or 1

    def cost_at(thr: float) -> float:
        fpr = sum(1 for p, l in zip(probs, labels) if p >= thr and l == 0) / n_wait
        fnr = sum(1 for p, l in zip(probs, labels) if p < thr and l == 1) / n_speak
        return cost_ratio * fpr + fnr

    unconstrained_thr, unconstrained_cost = 0.5, float("inf")
    for i in range(1, 100):
        thr = i / 100
        cost = cost_at(thr)
        if cost < unconstrained_cost or (cost == unconstrained_cost and thr < unconstrained_thr):
            unconstrained_cost, unconstrained_thr = cost, thr

    if not constraints:
        return {
            "threshold": unconstrained_thr,
            "cost": unconstrained_cost,
            "admissible_count": 99,
            "tier1_unsatisfiable": False,
        }

    admissible: list[float] = []
    for i in range(1, 100):
        thr = i / 100
        if all((prob >= thr) == (required == "speak") for prob, required in constraints):
            admissible.append(thr)

    if not admissible:
        return {
            "threshold": unconstrained_thr,
            "cost": unconstrained_cost,
            "admissible_count": 0,
            "tier1_unsatisfiable": True,
        }

    best_thr, best_cost = admissible[0], cost_at(admissible[0])
    for thr in admissible[1:]:
        cost = cost_at(thr)
        if cost < best_cost or (cost == best_cost and thr < best_thr):
            best_cost, best_thr = cost, thr

    return {
        "threshold": best_thr,
        "cost": best_cost,
        "admissible_count": len(admissible),
        "tier1_unsatisfiable": False,
    }


def load_labeled_rows(path: str) -> list[dict]:
    """Load rows carrying context/text/label from a gold/dev-set-shaped json file (top-level
    samples[]) or a jsonl file of rows, filtered to recognized labels.
    """
    p = Path(path)
    if p.suffix == ".json":
        data = json.load(open(p, encoding="utf-8"))
        rows = data.get("samples", [])
    else:
        rows = load_jsonl(p)
    return [r for r in rows if r.get("label") in LABEL2ID]


def score_probs(model_dir: str, rows: list[dict]) -> list[float]:
    """Score P(speak) for each row's context/text pair, one row per inference call.

    Supports an ONNX export dir (model.onnx present, scored via onnxruntime, the same path
    evaluate.py uses) and a torch checkpoint dir (scored via transformers), so the same CLI
    works against models/eot-distilbert and models/eot-distilbert-onnx-int8 alike.
    """
    # Scored one row per inference call (batch size 1), matching serve.py's request shape:
    # int8 dynamic quantization is batch-composition-dependent, so selection must score the way serving scores.
    model_path = Path(model_dir)
    model_inputs = [build_input(r.get("context", ""), r["text"]) for r in rows]

    if (model_path / "model.onnx").exists():
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer

        session = ort.InferenceSession(str(model_path / "model.onnx"), providers=["CPUExecutionProvider"])
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        input_names = {i.name for i in session.get_inputs()}
        probs: list[float] = []
        for model_input in model_inputs:
            encoded = tokenizer(model_input, truncation=True, padding="max_length", max_length=128, return_tensors="np")
            onnx_inputs = {name: value for name, value in encoded.items() if name in input_names}
            logits = session.run(None, onnx_inputs)[0]
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            exp = np.exp(shifted)
            softmax = exp / np.sum(exp, axis=-1, keepdims=True)
            probs.append(float(softmax[0, LABEL2ID["speak"]]))
        return probs

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    probs = []
    with torch.no_grad():
        for model_input in model_inputs:
            encoded = tokenizer(model_input, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            prob = torch.softmax(outputs.logits, dim=-1)[0, LABEL2ID["speak"]]
            probs.append(prob.detach().cpu().item())
    return probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick a cost-optimal decision threshold, optionally constrained by tier-1 guardrail cards.")
    parser.add_argument("--model-dir", type=str, required=True, help="Dir with model.onnx (fp32 or int8 export) or a torch checkpoint")
    parser.add_argument("--labels", type=str, required=True, help="Dev-set-shaped json (samples[] with context/text/label) or jsonl rows; the cost objective")
    parser.add_argument("--cost-ratio", type=float, default=5, help="Weight on false-speak vs false-wait in the cost function")
    parser.add_argument("--constraints", type=str, default="", help="Optional json or jsonl file of guardrail rows; each row's label is the required decision (speak or wait)")
    parser.add_argument("--out", type=str, default="", help="Where to write threshold.json; defaults to {model-dir}/threshold.json")
    parser.add_argument("--allow-unsatisfiable", action="store_true", help="Write the threshold even when no value satisfies the guardrails, and stamp the file as unsatisfiable. Off by default, because the fallback pick is indistinguishable from a clean one once written")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dev_rows = load_labeled_rows(args.labels)
    dev_probs = score_probs(args.model_dir, dev_rows)
    dev_labels = [LABEL2ID[r["label"]] for r in dev_rows]

    constraints: list[tuple[float, str]] | None = None
    if args.constraints:
        constraint_rows = load_labeled_rows(args.constraints)
        constraint_probs = score_probs(args.model_dir, constraint_rows)
        constraints = list(zip(constraint_probs, [r["label"] for r in constraint_rows]))

    result = sweep(dev_probs, dev_labels, cost_ratio=args.cost_ratio, constraints=constraints)

    method = "cost-optimal with tier-1 guardrail constraints" if constraints else "cost-optimal"
    payload = {
        "threshold": result["threshold"],
        "cost_ratio": args.cost_ratio,
        "method": method,
        "selection_source": args.labels,
        "dev_n": len(dev_labels),
        "admissible_count": result["admissible_count"],
    }
    if result["tier1_unsatisfiable"]:
        payload["tier1_unsatisfiable"] = True
        if not args.allow_unsatisfiable:
            # Refuse to write. When no threshold satisfies the guardrails, the sweep falls back to
            # the unconstrained pick, and writing that produced a threshold.json indistinguishable
            # from a clean one: serve.py reads the number and nothing else, make and CI saw exit 0,
            # and a model known to break the policy absolutes shipped quietly. The README promised
            # this failed loudly and it did not (adversary, 2026-08-29). Recording an unsatisfiable
            # lane on purpose is still possible, but it has to be asked for.
            sys.stderr.write(
                "REFUSING to write %s: no threshold satisfies the tier-1 guardrails in %s.\n"
                "The best unconstrained pick would be %s, which breaks at least one absolute.\n"
                "Fix the model or the constraints. To record this state deliberately, pass "
                "--allow-unsatisfiable.\n"
                % (args.out or Path(args.model_dir) / "threshold.json",
                   args.constraints, result["threshold"]))
            sys.exit(3)

    out_path = Path(args.out) if args.out else Path(args.model_dir) / "threshold.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"threshold {result['threshold']} cost {result['cost']:.4f} "
        f"admissible {result['admissible_count']} tier1_unsatisfiable={result['tier1_unsatisfiable']}"
    )
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()