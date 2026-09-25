"""Build the out-of-distribution eval slice from real production voice-agent calls.

The scratch model trains on synthetic templates and is graded on a gold set
drawn from the same taxonomy phrasing space, so nothing in the offline evals
would catch template memorization. This script builds the guard: real caller
turns from the author's own production voice stack (ElevenLabs conversational
agent on a real phone number), where turns label themselves. The point where
the caller actually stopped and the agent responded is a true "speak"; a
mid-turn prefix of that same utterance is a true "wait". No annotation step.

Production-call discrimination is exact, borrowed from the stack's own metrics
code: a real call was initiated through Twilio and ran longer than the 25s
watchdog pings; widget sessions and test-harness runs are excluded on the stub
before any detail fetch. Turn boundaries come from the vendor's turn-taking
system, so a caller who got talked over can appear as a short "complete" turn;
that noise is accepted and stated in the approach doc.

Privacy: the output file contains real personal call content. It is written
locally, kept out of git, and only aggregate metrics from it appear in the
build. Requires ELEVENLABS_API_KEY, and an agent id via ELEVENLABS_AGENT_ID
or --agent. Read-only against the API.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
from pathlib import Path


def _get(url: str, key: str) -> dict:
    req = urllib.request.Request(url, headers={"xi-api-key": key})
    return json.load(urllib.request.urlopen(req, timeout=25))


def normalize(text: str) -> str:
    """ASR-style form, matching the serving input contract: lowercase, no terminal punctuation."""
    t = text.lower().strip()
    while t and t[-1] in ".?!":
        t = t[:-1].rstrip()
    return t.replace("¿", "").replace("¡", "").strip()


# Turns where the vendor's turn-taker responded but OUR written policy says wait
# (announced continuation is a tier-1 guardrail class). Keeping these as "speak"
# labels would import the vendor's behavior against the policy, in training and,
# worse, in the referee. The correction RELABELS the full turn to wait rather than
# dropping it, marked policy_corrected, so n is preserved and each corrected row
# becomes a live probe of the exact place policy and vendor diverge. The rule is
# content-anchored and pre-registered by the written policy, applied symmetrically
# to every model. Counts are printed and reported wherever the slice is described.
POLICY_COLLISIONS = ("one more thing", "hold that thought")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", default=os.environ.get("ELEVENLABS_AGENT_ID", ""), help="ElevenLabs agent id")
    ap.add_argument("--max-calls", type=int, default=60, help="most recent production calls to scan")
    ap.add_argument("--max-rows", type=int, default=400)
    ap.add_argument("--min-duration", type=int, default=26, help="seconds; excludes the 20s watchdog pings")
    ap.add_argument("--min-words-prefix", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/ood_aj.jsonl")
    args = ap.parse_args()

    key = os.environ["ELEVENLABS_API_KEY"]
    if not args.agent:
        raise SystemExit("no agent id: set ELEVENLABS_AGENT_ID or pass --agent")

    rng = random.Random(args.seed)

    # Paginate the conversation stubs; filter to real calls before any detail fetch.
    stubs, cursor = [], None
    while len(stubs) < args.max_calls:
        url = f"https://api.elevenlabs.io/v1/convai/conversations?agent_id={args.agent}&page_size=100"
        if cursor:
            url += f"&cursor={cursor}"
        page = _get(url, key)
        batch = page.get("conversations") or []
        for c in batch:
            if c.get("conversation_initiation_source") != "twilio":
                continue
            if (c.get("call_duration_secs") or 0) < args.min_duration:
                continue
            stubs.append(c)
            if len(stubs) >= args.max_calls:
                break
        cursor = page.get("next_cursor")
        if not page.get("has_more") or not batch or not cursor:
            break

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    calls_used = 0
    relabeled_policy = 0
    for c in stubs:
        try:
            d = _get(f"https://api.elevenlabs.io/v1/convai/conversations/{c['conversation_id']}", key)
        except Exception:
            continue
        transcript = d.get("transcript") or []
        calls_used += 1
        call_id = c["conversation_id"][-12:]  # rides on every row so splits can group by call
        prev_agent = ""
        for t in transcript:
            role = t.get("role")
            msg = (t.get("message") or "").strip()
            if role == "agent":
                if msg:
                    prev_agent = msg
                continue
            if role != "user" or not msg or t.get("ignored_as_backchannel"):
                continue
            text = normalize(msg)
            if not text:
                continue
            context = normalize(prev_agent)
            if (context, text) not in seen:
                seen.add((context, text))
                if any(phrase in text for phrase in POLICY_COLLISIONS):
                    relabeled_policy += 1
                    rows.append({"context": context, "text": text, "label": "wait", "cls": "OOD", "variant": "real", "lang": "en", "call": call_id, "policy_corrected": True})
                else:
                    rows.append({"context": context, "text": text, "label": "speak", "cls": "OOD", "variant": "real", "lang": "en", "call": call_id})
            words = text.split()
            if len(words) >= args.min_words_prefix:
                # A cut right after a sentence-final mark leaves a grammatically complete
                # prefix that would be labeled wait; those are the most common false waits,
                # so sentence-boundary cut points are excluded from the candidate set.
                candidates = [i for i in range(2, len(words) - 1) if not words[i - 1].rstrip('"\'').endswith((".", "!", "?"))]
                n_cuts = 2 if len(words) >= 10 else 1
                for _ in range(n_cuts):
                    if not candidates:
                        break
                    cut = rng.choice(candidates)
                    prefix = " ".join(words[:cut])
                    if (context, prefix) not in seen:
                        seen.add((context, prefix))
                        rows.append({"context": context, "text": prefix, "label": "wait", "cls": "OODT", "variant": "real-prefix", "lang": "en", "call": call_id})

    rng.shuffle(rows)
    rows = rows[: args.max_rows]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_label = Counter(r["label"] for r in rows)
    print(f"scanned {len(stubs)} production calls, used {calls_used}")
    print(f"relabeled {relabeled_policy} vendor-policy collision turns to wait, marked policy_corrected (phrases: {POLICY_COLLISIONS})")
    print(f"wrote {len(rows)} rows to {args.out} (labels: {dict(by_label)})")
    print("this file holds real personal call content; it stays local and out of git, aggregates only in the doc")


if __name__ == "__main__":
    main()