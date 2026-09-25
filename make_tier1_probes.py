"""Rebuild the twelve tier-1 guardrail rows from the committed data.

These are the constraints `pick_threshold.py --constraints` enforces: the absolutes of POLICY.md,
which no operating point is allowed to violate on the served artifact. The file they live in was
written to /tmp during the v7 to v9 work and the machine cleaned it up, so the twelve rows existed
nowhere by 2026-08-29. They were never arbitrary, though. Every one is a row of a committed file,
selected by a rule, so the set is derived here rather than remembered:

  every F-class card in the gold set          announced continuation, must hold        4
  gold card H5                                the hold-that-thought case               1
  dev card dH4                                the card int8 quantization moved         1
  every regression row, with its own label    each one traced to a live miss           6

Run `make tier1` to regenerate. The output is committed so a reviewer can rerun the constrained
pick without owning this machine.
"""
import json

OUT = "data/tier1_probes.jsonl"


def main() -> None:
    gold = json.load(open("data/gold_set.json", encoding="utf-8"))["samples"]
    dev = json.load(open("data/dev_set.json", encoding="utf-8"))["samples"]
    rows = []

    for s in gold:
        if s.get("cls") == "F" and s.get("label") == "wait":
            rows.append((s["id"], s.get("context", ""), s["text"], "wait"))
    for s in gold:
        if s.get("id") == "H5":
            rows.append((s["id"], s.get("context", ""), s["text"], "wait"))
    for s in dev:
        if s.get("id") == "dH4":
            rows.append((s["id"], s.get("context", ""), s["text"], "wait"))
    with open("data/regressions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append((r.get("source", "regression"), r.get("context", ""), r["text"], r["label"]))

    if len(rows) != 12:
        raise SystemExit("expected 12 tier-1 rows, derived %d: %s"
                         % (len(rows), [r[0] for r in rows]))

    with open(OUT, "w", encoding="utf-8") as fh:
        for _id, ctx, text, label in rows:
            fh.write(json.dumps({"context": ctx, "text": text, "label": label},
                                ensure_ascii=False) + "\n")
    print("wrote %s (%d rows)" % (OUT, len(rows)))


if __name__ == "__main__":
    main()