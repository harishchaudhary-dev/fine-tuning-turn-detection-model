"""English scale-up generator for the from-scratch lane.

Reuses the template BANKS from synth.py verbatim, because those templates are
the policy encoding and must not fork, but swaps in much larger slot pools
(generated spoken numbers, more cities, more times) so the dedup step stops
capping volume. At the default per-template count the base generator tops out
near 1.5k unique rows; the scratch model needs an order of magnitude more, and
volume has to come from slot diversity, not from new templates that would
silently drift the policy.

Writes data/train_scaled.jsonl and never touches data/train.jsonl. Pure code,
seeded, regenerates byte-identically. Same augmentation pipeline and gold-set
collision guard as synth.py.
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from pathlib import Path

import synth  # templates and agent-context banks are the single source of policy

CITIES = [
    "Fontana", "Barstow", "Phoenix", "Denver", "Reno", "Ontario", "Stockton", "Laredo",
    "Memphis", "Dallas", "Atlanta", "Fresno", "El Paso", "Tucson", "Bakersfield", "Amarillo",
    "Joliet", "Columbus", "Charlotte", "Savannah", "Jacksonville", "Kansas City", "Oklahoma City",
    "Albuquerque", "Salt Lake", "Boise", "Portland", "Spokane", "Cheyenne", "Omaha",
    "Little Rock", "Shreveport", "Nashville", "Louisville", "Indianapolis", "Toledo",
    "Harrisburg", "Allentown", "Macon", "Mobile",
]
FACILITIES = [
    "the warehouse", "the receiver", "the shipper", "the yard", "dock four", "the cross-dock",
    "the DC", "gate two", "the produce terminal", "the cold storage", "door seventeen", "the drop lot",
]
EQUIP = ["the reefer", "the dry van", "the flatbed", "trailer six two", "the step deck", "the box truck", "trailer forty one"]
DOCS = ["the BOL", "the rate con", "the lumper receipt", "the PO number", "the detention form", "the seal number", "the delivery receipt"]
HIGHWAYS = ["the ten", "the fifteen", "the forty", "the five", "the eight", "the twenty", "the seventy", "the ninety four", "the eighty"]

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TENS_WORDS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def spoken_digits(rng: random.Random, n_lo: int = 3, n_hi: int = 7) -> str:
    """A digit string read aloud the way callers read them, with occasional grouping pauses."""
    n = rng.randint(n_lo, n_hi)
    words = [ONES[rng.randrange(10)] for _ in range(n)]
    if n >= 5 and rng.random() < 0.5:
        cut = rng.randint(2, n - 2)
        return " ".join(words[:cut]) + ", " + " ".join(words[cut:])
    return " ".join(words)


def spoken_teen_or_tens(rng: random.Random, n: int) -> str:
    teens = {14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"}
    if n in teens:
        return teens[n]
    tens = TENS_WORDS[(n // 10) - 2]
    if n % 10 == 0:
        return tens
    return f"{tens} {ONES[n % 10]}"


def spoken_rate(rng: random.Random) -> str:
    """A freight rate said aloud: "eighteen fifty", "two thousand two hundred", "nineteen seventy five"."""
    if rng.random() < 0.5:
        hundreds = rng.randint(14, 32)
        tail = rng.choice(["hundred", "fifty", "twenty five", "seventy five", "ninety"])
        return f"{spoken_teen_or_tens(rng, hundreds)} {tail}"
    thousands = rng.choice(["two thousand", "three thousand", "twenty five hundred", "eighteen hundred"])
    tail = rng.choice(["", " one hundred", " two hundred", " four fifty", " even"])
    return (thousands + tail).strip()


def spoken_time(rng: random.Random) -> str:
    style = rng.random()
    if style < 0.35:
        hour = rng.choice(["seven", "eight", "nine", "ten", "eleven", "noon", "one", "two", "three", "four", "five", "six"])
        if hour == "noon":
            return "noon"
        mins = rng.choice(["", " fifteen", " thirty", " forty five"])
        ampm = rng.choice([" am", " pm"])
        return f"{hour}{mins}{ampm}"
    if style < 0.6:
        return f"{rng.choice(DAYS)} {rng.choice(['morning', 'afternoon', 'first thing', 'by noon', 'end of day'])}"
    return rng.choice(["tomorrow morning", "tonight", "first thing tomorrow", "later today", "within the hour", "in about an hour"])


def fill(t: str, rng: random.Random) -> str:
    return t.format(
        city=rng.choice(CITIES), facility=rng.choice(FACILITIES), load=spoken_digits(rng, 3, 5),
        rate=spoken_rate(rng), time=spoken_time(rng), equip=rng.choice(EQUIP),
        doc=rng.choice(DOCS), hwy=rng.choice(HIGHWAYS), digits=spoken_digits(rng),
        f1=rng.choice(synth.FILLERS), f2=rng.choice(synth.FILLERS),
    )


def ctx(kind: str | None, rng: random.Random) -> str:
    if kind is None:
        return ""
    return fill(rng.choice(synth.AGENT_CTX[kind]), rng)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/train_scaled.jsonl")
    ap.add_argument("--per-template", type=int, default=60, help="filled instances per template")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gold", default="data/gold_set.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []

    # tpl is a stable template id (crc32 of the template string). Augmented rows inherit
    # their parent's tpl, so a grouped train/val split keeps every rendering of one
    # template on one side and slot-fill leakage cannot inflate validation.
    for cls, label, ctx_kind, templates in synth.BANKS:
        for t in templates:
            tpl = f"en{zlib.crc32(t.encode())}"
            for _ in range(args.per_template):
                text = fill(t, rng)
                context = ctx(ctx_kind, rng)
                rows.append({"context": context, "text": text, "label": label, "cls": cls, "variant": "clean", "lang": "en", "tpl": tpl})

    # Truncation augmentation: complete speak utterances re-emitted cut short, labeled wait.
    speak_rows = [r for r in rows if r["label"] == "speak" and r["cls"] in ("A", "B", "H")]
    for r in speak_rows:
        cut = synth.truncate(r["text"], rng)
        if cut:
            rows.append({"context": r["context"], "text": cut, "label": "wait", "cls": "T", "variant": "clean", "lang": "en", "tpl": r["tpl"]})

    # Context-dropout augmentation: every contexted sample also emitted bare.
    for r in list(rows):
        if r["context"]:
            rows.append({**r, "context": "", "variant": r["variant"] + "+noctx"})

    # ASR-style variant of everything.
    for r in list(rows):
        t = synth.asr_variant(r["text"])
        if t != r["text"]:
            rows.append({**r, "text": t, "variant": "asr"})

    # Dedup and drop any exact-text collision with the frozen gold set.
    gold = json.load(open(args.gold))
    gold_texts = {s["text"].strip().lower() for s in gold["samples"]}
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for r in rows:
        key = (r["context"], r["text"])
        if key in seen or r["text"].strip().lower() in gold_texts:
            continue
        seen.add(key)
        out.append(r)
    rng.shuffle(out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_label = Counter(r["label"] for r in out)
    by_cls = Counter(r["cls"] for r in out)
    print(f"wrote {len(out)} samples to {args.out}")
    print("labels:", dict(by_label))
    print("classes:", dict(sorted(by_cls.items())))


if __name__ == "__main__":
    main()
