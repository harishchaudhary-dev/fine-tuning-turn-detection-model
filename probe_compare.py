"""Probe two served int8 models side by side and render docs/probe-comparison.html.

Scores a hand-written probe set, every policy class plus off-template assistant
register, texting register, and Spanish, one row at a time on the served int8
artifacts exactly as serve.py does, each model at its own picked threshold.
The page shows P(turn complete) per model with the threshold tick, the policy's
expected decision, and flags any cell that disagrees with policy. Regenerate
after any retrain: .venv/bin/python probe_compare.py
"""

from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from common import LABEL2ID, build_input, load_threshold  # noqa: E402

MODELS = [
    ("Fine-tuned DistilBERT (ships, 66.96M)", REPO / "models/eot-distilbert-onnx-int8"),
    ("From-scratch, own pretrain (7.36M)", REPO / "models/eot-scratch-pre-onnx-int8"),
]

# (class, agent context, caller text, policy expectation)
PROBES = [
    ("A complete statement", "Dispatch, how can I help?", "I'm loaded and rolling out of Fontana now.", "speak"),
    ("A complete statement, ASR style", "", "i delivered to the receiver about an hour ago", "speak"),
    ("B complete question", "", "What time does the warehouse close?", "speak"),
    ("B complete question, ASR style", "", "do i need a lumper receipt for this one", "speak"),
    ("C ack", "The dock closes at three thirty today.", "Okay, got it.", "speak"),
    ("C casual closer", "Anything else I can help with?", "nah bye", "speak"),
    ("C texting register", "Your check call is set for noon.", "k thanks", "speak"),
    ("D mid-clause cutoff", "", "So after I deliver in Ontario I was planning to", "wait"),
    ("D mid-clause cutoff", "", "Can you tell the receiver that my ETA is now", "wait"),
    ("E disfluent trail", "", "Yeah so, um, the thing is, uh", "wait"),
    ("F mid-data readout", "Can I get your MC number?", "yeah it's four one five", "wait"),
    ("F mid-data readout", "Best callback number?", "my cell is three three zero, two", "wait"),
    ("G connector-final", "", "I can pick up Thursday morning, but", "wait"),
    ("G connector-final", "", "The rate works for me, although", "wait"),
    ("H complete, might continue", "Where are you now?", "Just passed Barstow.", "speak"),
    ("H announced continuation", "Anything else I can help with?", "Actually yeah, one more thing.", "wait"),
    ("H announced continuation", "Is there anything else?", "Wait, before you go, one more question.", "wait"),
    ("I hedge, attributed (unsure)", "", "The broker said the lumper was covered, supposedly.", "speak"),
    ("I hedge, owned (sure)", "", "The detention was approved, or something", "wait"),
    ("I handoff hedge", "", "That's all I need, I guess.", "speak"),
    ("I owned statement, no softener", "", "The rate came out to nineteen fifty.", "speak"),
    ("J self-interrupt", "", "Book me on the- wait, hold on", "wait"),
    ("J full retraction", "", "Can you check the- no, scratch that.", "speak"),
    ("K self-retrieval hold", "", "Hang on, let me grab the load number", "wait"),
    ("K narrated external interruption", "", "I've got another call coming in, hang on", "speak"),
    ("Off-template assistant register", "", "how many interviews do i have this week", "speak"),
    ("Off-template prefix", "it is currently fifty-six degrees with clear skies", "how about the weather in, like, seven", "wait"),
    ("Off-template prefix", "", "can you move my dentist appointment to", "wait"),
    ("Off-template ack", "I moved it to Thursday at ten.", "yeah that works for me thanks", "speak"),
    ("ES complete statement", "Habla dispatch, dígame.", "Ya entregué en la bodega hace como una hora.", "speak"),
    ("ES question, ASR style", "", "a qué hora cierra el yard", "speak"),
    ("ES mid-clause cutoff", "", "Ando como a veinte millas pero el tráfico en la diez está", "wait"),
    ("ES mid-data readout", "¿Me da su número de MC?", "sí, es siete uno cinco", "wait"),
    ("ES rate accepted", "El rate es mil novecientos, todo incluido.", "Órale, la agarro.", "speak"),
    ("ES announced continuation", "¿Alguna otra cosa?", "Ah sí, una cosa más.", "wait"),
    ("Boundary card (labeled unsure)", "Does the noon slot work?", "That should work, probably...", "unsure"),
]


def load(model_dir: Path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    names = {i.name for i in sess.get_inputs()}
    return sess, tok, names, load_threshold(model_dir)


def score(sess, tok, names, context: str, text: str) -> tuple[float, float]:
    enc = tok(build_input(context, text), truncation=True, padding="max_length", max_length=128, return_tensors="np")
    t0 = time.perf_counter()
    logits = sess.run(None, {k: v for k, v in enc.items() if k in names})[0][0]
    ms = (time.perf_counter() - t0) * 1000
    e = np.exp(logits - logits.max())
    return float((e / e.sum())[LABEL2ID["speak"]]), ms


def main() -> None:
    loaded = [(name, *load(d)) for name, d in MODELS]
    for _, sess, tok, names, _ in loaded:
        score(sess, tok, names, "", "warmup")

    rows = []
    for cls, ctx, text, expected in PROBES:
        cells = []
        for _, sess, tok, names, thr in loaded:
            p, ms = score(sess, tok, names, ctx, text)
            cells.append({"p": p, "ms": ms, "decision": "speak" if p >= thr else "wait", "thr": thr})
        rows.append({"cls": cls, "ctx": ctx, "text": text, "expected": expected, "cells": cells})

    n_graded = sum(1 for r in rows if r["expected"] != "unsure")
    agree = [sum(1 for r in rows if r["expected"] != "unsure" and r["cells"][i]["decision"] == r["expected"]) for i in range(len(loaded))]
    mean_ms = [float(np.mean([r["cells"][i]["ms"] for r in rows])) for i in range(len(loaded))]
    both_agree = sum(1 for r in rows if r["cells"][0]["decision"] == r["cells"][1]["decision"])

    def chip(d: str) -> str:
        return f'<span class="chip {d}">{d.upper()}</span>'

    def cell(c: dict, expected: str) -> str:
        bad = expected != "unsure" and c["decision"] != expected
        return (
            f'<td class="m {"bad" if bad else ""}">'
            f'<div class="bar"><div class="fill {c["decision"]}" style="width:{c["p"] * 100:.1f}%"></div><div class="mark" style="left:{c["thr"] * 100:.1f}%"></div></div>'
            f'<div class="line"><b>{c["p"]:.3f}</b> {chip(c["decision"])} <span class="ms">{c["ms"]:.1f} ms</span></div></td>'
        )

    trs = []
    for i, r in enumerate(rows, 1):
        ctx = f'<div class="ctx">agent: {html.escape(r["ctx"])}</div>' if r["ctx"] else ""
        expected_cell = chip(r["expected"]) if r["expected"] != "unsure" else '<span class="chip unsure">UNSURE</span>'
        trs.append(
            f'<tr><td class="n">{i}</td><td class="cls">{html.escape(r["cls"])}</td>'
            f'<td class="txt">{ctx}<div class="say">{html.escape(r["text"])}</div></td>'
            f'<td class="exp">{expected_cell}</td>' + "".join(cell(c, r["expected"]) for c in r["cells"]) + "</tr>"
        )

    heads = "".join(
        f'<th class="mh">{html.escape(name)}<div class="sub">threshold {thr:.2f} · {agree[i]}/{n_graded} match policy · mean {mean_ms[i]:.1f} ms</div></th>'
        for i, (name, _, _, _, thr) in enumerate(loaded)
    )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>EoT probe comparison</title>
<style>
:root {{ --ink:#1c1a17; --dim:#6b655c; --line:#e4dfd6; --speak:#2f7d4f; --wait:#b8741f; --bad:#c43d2f; }}
body {{ margin:0; padding:40px 48px; font: 17px/1.45 -apple-system, system-ui, Helvetica, Arial, sans-serif; color:var(--ink); background:#fff; }}
h1 {{ font-size:28px; margin:0 0 6px; }}
p.lead {{ color:var(--dim); margin:0 0 26px; max-width:90ch; }}
table {{ border-collapse:collapse; width:100%; }}
th {{ text-align:left; font-size:15px; padding:10px 12px; border-bottom:2px solid var(--ink); vertical-align:bottom; }}
th.mh {{ width:24%; }}
th .sub {{ font-weight:normal; color:var(--dim); font-size:14px; margin-top:4px; }}
td {{ padding:12px; border-bottom:1px solid var(--line); vertical-align:top; }}
td.n {{ color:var(--dim); width:2ch; }}
td.cls {{ color:var(--dim); font-size:15px; width:19ch; }}
td.txt .ctx {{ color:var(--dim); font-size:15px; margin-bottom:3px; }}
td.txt .say {{ font-size:18px; }}
td.exp {{ width:9ch; }}
td.m {{ width:24%; }}
td.m.bad {{ background:#fdf1ef; box-shadow: inset 4px 0 0 var(--bad); }}
.bar {{ position:relative; height:10px; background:#eee8df; border-radius:6px; margin:6px 0 8px; }}
.fill {{ height:100%; border-radius:6px; }}
.fill.speak {{ background:var(--speak); }} .fill.wait {{ background:var(--wait); }}
.mark {{ position:absolute; top:-4px; width:2px; height:18px; background:var(--ink); opacity:.5; }}
.chip {{ display:inline-block; font-size:12.5px; font-weight:700; letter-spacing:.06em; padding:3px 9px; border-radius:999px; color:#fff; }}
.chip.speak {{ background:var(--speak); }} .chip.wait {{ background:var(--wait); }} .chip.unsure {{ background:#8a8378; }}
.ms {{ color:var(--dim); font-size:14px; margin-left:6px; }}
.line b {{ font-variant-numeric: tabular-nums; }}
.foot {{ color:var(--dim); font-size:15px; margin-top:22px; max-width:100ch; }}
</style></head><body>
<h1>End-of-turn probes, two models side by side</h1>
<p class="lead">{len(rows)} hand-written probes across every policy class, off-template assistant register, texting register, and Spanish, scored one row at a time on the served int8 artifacts exactly as the API does. The bar is P(turn complete); the tick is each model's own picked threshold; a red-edged cell is a decision that disagrees with the written policy. The two models agree with each other on {both_agree} of {len(rows)}.</p>
<table><thead><tr><th>#</th><th>Class</th><th>What the caller said</th><th>Policy</th>{heads}</tr></thead>
<tbody>{"".join(trs)}</tbody></table>
<p class="foot">Thresholds come from the dev set under the twelve tier-1 guardrail constraints, picked on these same int8 files. Latency is model-only, single row, warmed, on a laptop. The unsure card is one of the seven boundary cards the labeler marked as genuinely ambiguous; a good model sits near its threshold there rather than being confident either way.</p>
</body></html>"""
    out = REPO / "docs/probe-comparison.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}")
    for i, (name, *_) in enumerate(loaded):
        print(f"{name}: {agree[i]}/{n_graded} match policy, mean {mean_ms[i]:.1f} ms")
    print(f"models agree with each other on {both_agree}/{len(rows)}")


if __name__ == "__main__":
    main()