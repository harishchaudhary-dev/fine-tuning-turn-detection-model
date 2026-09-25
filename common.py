"""Shared helpers for the end-of-turn detector: input formatting and data loading."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

LABEL2ID = {"wait": 0, "speak": 1}

DEFAULT_SPEAK_THRESHOLD = 0.8333


def build_input(context: str, text: str) -> str:
    """Build the model input string from conversation context and caller utterance."""

    context = context.strip()

    if context.lower().startswith("agent:"):
        context = context[6:].strip()

    if context:
        return f"agent: {context} caller: {text}"

    return f"caller: {text}"


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a jsonl file, one JSON object per line, skipping blank lines."""

    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def load_gold(
    path: Union[str, Path] = "data/gold_set.json",
) -> Dict[str, Any]:
    """Load the frozen gold set."""

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gold_threshold(
    path: Union[str, Path] = "data/gold_set.json",
) -> float:
    """Return the speak threshold recorded in the gold set."""

    try:
        gold = load_gold(path)

    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_SPEAK_THRESHOLD

    return gold.get(
        "speak_threshold",
        DEFAULT_SPEAK_THRESHOLD,
    )


def load_threshold(model_dir: Union[str, Path]) -> float:
    """Load model threshold from threshold.json."""

    p = Path(model_dir) / "threshold.json"

    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)["threshold"]

        except (json.JSONDecodeError, KeyError):
            pass

    return gold_threshold()