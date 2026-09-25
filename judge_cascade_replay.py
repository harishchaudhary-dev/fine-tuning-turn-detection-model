"""Replay two judge-panel designs over the recorded votes from the blind labeling batch.

Design A (full panel): all three judges vote on every card, 2-of-3 majority.
Design B (cascade): GPT and Claude vote first; Gemini is summoned only when they split
or either abstains. Under simple majority the third vote can only change the outcome
on a split, so the two designs must produce identical labels; what differs is cost
and what gets measured. This script checks that equivalence empirically and reports
the cost, the hidden dissents the cascade never records, and judge calibration.

Inputs are all committed: data/judge_votes.json (raw labels from the three judges on
the 90-card blind batch, gold and fresh cards shuffled together), data/gold_set.json
(the frozen human referee), data/dev_set.json (the fresh cards with their final
majority labels). Run: python judge_cascade_replay.py
"""

import json


def majority(votes: list[str | None]) -> str | None:
    real = [v for v in votes if v in ("speak", "wait")]
    if not real:
        return None
    s = real.count("speak")
    w = real.count("wait")
    return "speak" if s > w else ("wait" if w > s else None)


def main() -> None:
    votes = json.load(open("data/judge_votes.json", encoding="utf-8"))
    gold = json.load(open("data/gold_set.json", encoding="utf-8"))
    dev = json.load(open("data/dev_set.json", encoding="utf-8"))

    gold_truth = {s["id"]: s["label"] for s in gold["samples"] if s.get("label") in ("speak", "wait")}
    boundary_ids = {s["id"] for s in gold["samples"] if s.get("label") not in ("speak", "wait")}
    dev_truth = {s["id"]: s["label"] for s in dev["samples"]}

    rows = []
    for cid, v in votes.items():
        all3 = majority([v["claude"], v["gemini"], v["gpt"]])
        pair_agree = v["gpt"] == v["claude"] and v["gpt"] in ("speak", "wait")
        cascade = v["gpt"] if pair_agree else all3
        rows.append({**v, "id": cid, "all3": all3, "cascade": cascade, "cost": 2 if pair_agree else 3, "pair_agree": pair_agree})

    n = len(rows)
    agree = sum(r["pair_agree"] for r in rows)
    cost_cascade = sum(r["cost"] for r in rows)
    mismatches = [r["id"] for r in rows if r["cascade"] != r["all3"]]
    unanimous = sum(1 for r in rows if r["claude"] == r["gemini"] == r["gpt"] and r["claude"] in ("speak", "wait"))
    hidden = [r for r in rows if r["pair_agree"] and r["gemini"] in ("speak", "wait") and r["gemini"] != r["gpt"]]

    print(f"cards: {n}   pair agreement (gpt+claude): {agree}/{n}")
    print(f"cost: cascade {cost_cascade} calls vs full panel {3 * n}   saved {100 * (3 * n - cost_cascade) / (3 * n):.0f}%")
    print(f"label equivalence: {n - len(mismatches)}/{n} identical   mismatches: {mismatches or 'none'}")
    print(f"unanimous 3-0: {unanimous}/{n}")
    for r in hidden:
        truth = gold_truth.get(r["id"]) or dev_truth.get(r["id"])
        verdict = "pair right" if truth == r["gpt"] else ("gemini right" if truth == r["gemini"] else "no truth")
        print(f"hidden dissent {r['id']}: pair={r['gpt']} gemini={r['gemini']} truth={truth} ({verdict})")

    for judge in ("claude", "gemini", "gpt"):
        unsure_ids = {r["id"] for r in rows if r[judge] == "unsure"}
        overlap = len(unsure_ids & boundary_ids)
        g_ok = sum(1 for r in rows if gold_truth.get(r["id"]) and r[judge] == gold_truth[r["id"]])
        d_ok = sum(1 for r in rows if dev_truth.get(r["id"]) and r[judge] == dev_truth[r["id"]])
        print(f"{judge}: gold {g_ok}/{len(gold_truth)}   fresh {d_ok}/{len(dev_truth)}   unsure {len(unsure_ids)} (on human-boundary cards: {overlap}/{len(boundary_ids)})")

    pair_wrong = [
        r["id"]
        for r in rows
        if r["pair_agree"] and (gold_truth.get(r["id"]) or dev_truth.get(r["id"])) and r["gpt"] != (gold_truth.get(r["id"]) or dev_truth.get(r["id"]))
    ]
    print(f"pair-agreed-but-wrong (the cascade's blind spot): {pair_wrong or 'zero cases'}")


if __name__ == "__main__":
    main()