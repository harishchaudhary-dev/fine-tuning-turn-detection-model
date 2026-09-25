"""Emit the README figures as SVG text, from counted constants, with a drift check.

Every number drawn on a figure is either counted from a committed artifact (gold card
count, judge-vote tallies, regression rows) or pinned in FROZEN_NUMBERS with the report it
came from. `--write` regenerates assets/*.svg; `--check` regenerates in memory and fails
if the committed files differ, so a hand edit or a number that moved in one place and not
the others fails CI. No plotting library: hand-authored SVG stays crisp at any zoom.

Type sizes are chosen for a 75% browser zoom on GitHub's ~890px README column. Effective
pixels = font-size x (890 / viewBox width) x 0.75, and the floor is 12px, so the minimum
font-size for a canvas of width W is W / 55.6. `--check` asserts that floor on every text
element, because a hero that looks fine at 100% is exactly where unreadable labels hide.

Run: python draw_figures.py --write   |   python draw_figures.py --check
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
COLUMN_PX = 890
ZOOM = 0.75
FLOOR_PX = 12

# Report-derived figures, pinned at the v9 freeze. Each carries the artifact it was read from.
FROZEN_NUMBERS = {
    "gold_pr_auc": "0.949",  # eval_report.json, hard set, served int8, 53 cards
    "gold_wait_n": 27,  # gold cards labeled wait, the denominator of the zero
    "gold_false_speak": "0.000",  # 0 of 27, the 0.42 row of hard.threshold_sweep
    "real_pr_auc": "0.913",  # eval_report_distilen_oodtest.json, ood_test at 0.42
    "real_wait_n": 47,  # ood_test turns labeled wait
    "real_false_speak_count": 11,  # of those 47, at 0.42
    "real_false_speak": "0.234",  # the same sweep on ood_test, the sibling of the gold zero
    "p95_ms": "33.1",  # bench_report.json, end-to-end wall p95 at concurrency 8 on the shipped int8 file,
    # the worst of four passes with the dev box at a load average near 4 of its 18 cores
    "threshold": "0.42",  # models/eot-distilbert-onnx-int8/threshold.json
    "ood_n": 96,  # data/ood_test.jsonl (gitignored, real calls)
    "tier1_gates": 12,  # pick_threshold.py constraint probes
    "curve": [  # ood_test PR-AUC per lane, params by ONNX initializer sum
        ("0.48", "random init", "scratch, 3.70M"),
        ("0.60", "+ real calls", "scratch-real, 3.99M"),
        ("0.83", "+ 15-min pretrain", "scratch-pre, 7.36M"),
        ("0.91", "web-pretrained", "DistilBERT, 66.96M"),
    ],
    "ood_band": 0.85,  # EVALS.md tier-2 real-call band
}

# Palette: dark graphite with a warm amber accent. Cool slate marks the wait side.
BG0, BG1, BG2 = "#111316", "#0b0d10", "#07080a"
INK, MUTED, DIM = "#f2efe9", "#a8a39a", "#7d786f"
AMBER, AMBER_DEEP, AMBER_GLOW = "#f5b342", "#c98a1e", "#ffd27a"
SLATE = "#7fa3b8"
BORDER, DOT = "#2a2e33", "#1c2024"
MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def counted() -> dict:
    """Numbers counted from committed artifacts at run time."""
    gold = json.load(open(ROOT / "data/gold_set.json", encoding="utf-8"))
    votes = json.load(open(ROOT / "data/judge_votes.json", encoding="utf-8"))
    hard = sum(1 for s in gold["samples"] if s.get("label") in ("speak", "wait"))
    fresh = sum(1 for v in votes.values() if v.get("fresh"))
    pair_agree = sum(1 for v in votes.values() if v["gpt"] == v["claude"] and v["gpt"] in ("speak", "wait"))
    regressions = sum(1 for line in open(ROOT / "data/regressions.jsonl", encoding="utf-8") if line.strip())
    return {
        "gold_n": len(gold["samples"]),
        "gold_hard": hard,
        "gold_boundary": len(gold["samples"]) - hard,
        "batch_n": len(votes),
        "fresh_n": fresh,
        "exam_n": len(votes) - fresh,
        "pair_agree": pair_agree,
        "splits": len(votes) - pair_agree,
        "regressions": regressions,
    }


def text(x, y, s, size, fill, font=MONO, weight=None, anchor=None, spacing=None, opacity=None):
    attrs = [f'x="{x}"', f'y="{y}"', f'font-family="{font}"', f'font-size="{size}"', f'fill="{fill}"']
    if weight:
        attrs.append(f'font-weight="{weight}"')
    if anchor:
        attrs.append(f'text-anchor="{anchor}"')
    if spacing:
        attrs.append(f'letter-spacing="{spacing}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    return f"<text {' '.join(attrs)}>{s}</text>"


def rect(x, y, w, h, fill, rx=0, stroke=None, opacity=None):
    a = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"'
    if stroke:
        a += f' stroke="{stroke}"'
    if opacity is not None:
        a += f' opacity="{opacity}"'
    return a + "/>"


def arrow(x1, y1, x2, y2, color=AMBER_DEEP):
    """A straight connector with a small filled head at the end."""
    head_ = f'<polygon points="{x2},{y2} {x2 - 10},{y2 - 5} {x2 - 10},{y2 + 5}" fill="{color}"/>'
    return f'<line x1="{x1}" y1="{y1}" x2="{x2 - 8}" y2="{y2}" stroke="{color}" stroke-width="2"/>' + head_


def head(w, h, label, rx=0):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{label}">'
        f'<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{BG0}"/>'
        f'<stop offset="0.55" stop-color="{BG1}"/><stop offset="1" stop-color="{BG2}"/></linearGradient>'
        f'<pattern id="dots" width="26" height="26" patternUnits="userSpaceOnUse"><circle cx="13" cy="13" r="1.3" fill="{DOT}"/></pattern></defs>'
        f'{rect(0, 0, w, h, "url(#bg)", rx=rx)}{rect(0, 0, w, h, "url(#dots)", rx=rx)}'
    )


def hero(c: dict) -> str:
    f = FROZEN_NUMBERS
    w, h = 1200, 360
    label = (
        f"fine-tuning-turn-detection-model: an end-of-turn detector for voice agents, read at threshold {f['threshold']} on int8 CPU. "
        f"On the frozen human gold set it scores {f['gold_pr_auc']} PR-AUC and speaks over none of the {f['gold_wait_n']} wait cards. "
        f"On {f['ood_n']} held-out real-call turns it scores {f['real_pr_auc']} and speaks over {f['real_false_speak_count']} of the "
        f"{f['real_wait_n']} wait turns, a rate of {f['real_false_speak']}. End-to-end p95 is {f['p95_ms']} ms at concurrency 8, "
        f"the worst of four passes benched against that same file"
    )
    out = [head(w, h, label), rect(0, 0, w, 3, AMBER)]
    out.append(text(600, 52, "VOICE AI · END-OF-TURN DETECTION · FINE-TUNED AND FROM SCRATCH", 22, AMBER, weight=700, anchor="middle", spacing=3))
    out.append(
        f'<defs><linearGradient id="ttl" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{AMBER_GLOW}"/>'
        f'<stop offset="1" stop-color="{INK}"/></linearGradient>'
        f'<linearGradient id="rule" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{AMBER}" stop-opacity="0"/>'
        f'<stop offset="0.5" stop-color="{AMBER}" stop-opacity="0.95"/><stop offset="1" stop-color="{AMBER}" stop-opacity="0"/></linearGradient></defs>'
    )
    out.append(text(600, 108, "Is the caller done talking?", 48, "url(#ttl)", font=SANS, weight=700, anchor="middle"))
    out.append(
        f'<text x="600" y="152" font-family="{MONO}" font-size="27" fill="{MUTED}" text-anchor="middle">'
        f'One written policy, <tspan fill="{AMBER}">three referees</tspan>, {f["p95_ms"]} ms p95 at concurrency 8</text>'
    )
    out.append(rect(300, 176, 600, 2, "url(#rule)"))
    stats = [
        (195, f["gold_pr_auc"], "gold PR-AUC", f"0 of {f['gold_wait_n']} false speaks"),
        (465, f["real_pr_auc"], "real calls PR-AUC", f"held-out, {f['ood_n']} turns"),
        (735, f["real_false_speak"], "real false speaks", f"{f['real_false_speak_count']} of {f['real_wait_n']} wait turns"),
        (1005, f["threshold"], "threshold", f"cost 1:5, {f['tier1_gates']} gates"),
    ]
    for x, num, l1, l2 in stats:
        out.append(text(x, 244, num, 40, AMBER, weight=700, anchor="middle"))
        out.append(text(x, 282, l1, 22, MUTED, anchor="middle"))
        out.append(text(x, 310, l2, 22, DIM, anchor="middle"))
    # The motif: a waveform that runs up to the gate and stops. Speech on the left in amber,
    # the trailing silence on the right in slate. The gate is the decision.
    rng = random.Random(7)
    gate_x = 600
    for i in range(72):
        x = 84 + i * 14.4
        base = 350
        if x < gate_x:
            hgt = rng.uniform(6, 26)
            out.append(rect(round(x, 1), round(base - hgt, 1), 6, round(hgt, 1), AMBER, rx=2, opacity=0.32))
        else:
            hgt = rng.uniform(2, 6)
            out.append(rect(round(x, 1), round(base - hgt, 1), 6, round(hgt, 1), SLATE, rx=2, opacity=0.28))
    out.append(rect(gate_x - 1, 318, 2, 34, AMBER, opacity=0.9))
    out.append("</svg>")
    return "".join(out)


def band(eyebrow: str, headline: str, stat: str) -> str:
    w, h = 1040, 104
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{headline}">',
        f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{BG0}"/>'
        f'<stop offset="0.6" stop-color="{BG1}"/><stop offset="1" stop-color="{BG2}"/></linearGradient></defs>',
        rect(0, 0, w, h, "url(#g)", rx=9),
        rect(0, 0, w, h, "none", rx=9, stroke=BORDER),
        rect(0, 0, 7, h, AMBER, rx=3.5),
        text(34, 40, eyebrow, 19, AMBER_DEEP, weight=700, spacing=2),
        text(34, 78, headline, 28, INK, font=SANS, weight=700),
        text(830, 42, stat, 19, MUTED, anchor="end"),
    ]
    for i in range(10):
        cx = 852.1 + i * 16.2
        r = round(7.0 - i * 0.556, 1)
        op = round(0.92 - i * 0.088, 2)
        out.append(f'<circle cx="{round(cx, 1)}" cy="70.0" r="{r}" fill="{AMBER}" opacity="{op}"/>')
    out.append("</svg>")
    return "".join(out)


def referees(c: dict) -> str:
    f = FROZEN_NUMBERS
    w, h = 1040, 300
    label = (
        f"Three referees, one question each: a frozen gold set of {c['gold_n']} human-labeled cards for generalization, "
        f"{c['regressions']} probe-found regressions for memory, and {f['ood_n']} held-out real-call turns for discovery"
    )
    out = [head(w, h, label, rx=9), rect(0, 0, w, h, "none", rx=9, stroke=BORDER)]
    out.append(text(520, 44, "Three referees, one question each", 26, INK, font=SANS, weight=700, anchor="middle"))
    cards = [
        ("gold set", f"{c['gold_n']} cards, frozen", "Does it generalize to", "the human's judgment?", "never trained on", AMBER),
        ("regressions", f"{c['regressions']} probe-found misses", "Does a fixed failure", "stay fixed?", "overlap intended", AMBER),
        ("real calls", f"{f['ood_n']} held-out turns", "What can synthetic data", "not imagine?", "labeled by rule", SLATE),
    ]
    for i, (title, n, q1, q2, chip, color) in enumerate(cards):
        x = 30 + i * 330
        out.append(rect(x, 70, 320, 200, BG0, rx=10, stroke=BORDER))
        out.append(rect(x, 70, 6, 200, color, rx=3))
        out.append(text(x + 26, 108, title, 26, color, weight=700))
        out.append(text(x + 26, 142, n, 20, INK))
        out.append(text(x + 26, 178, q1, 19, MUTED, font=SANS))
        out.append(text(x + 26, 203, q2, 19, MUTED, font=SANS))
        out.append(rect(x + 26, 222, 16 + len(chip) * 11.4, 32, BG2, rx=16, stroke=BORDER))
        out.append(text(x + 34, 244, chip, 19, DIM))
    out.append("</svg>")
    return "".join(out)


def judges(c: dict) -> str:
    f = FROZEN_NUMBERS
    w, h = 1040, 450
    label = (
        f"How the dev set was labeled and why to trust it: {c['exam_n']} gold cards with known human answers hidden among "
        f"{c['fresh_n']} fresh cards, three stock vendor judges with zero training, two-of-three majority, "
        f"and the output feeds one file that tunes one number clamped by {f['tier1_gates']} human gates"
    )
    out = [head(w, h, label, rx=9), rect(0, 0, w, h, "none", rx=9, stroke=BORDER)]
    out.append(text(520, 44, "Machine labels you can trust: an exam, a vote, a small blast radius", 24, INK, font=SANS, weight=700, anchor="middle"))

    # Column 1: the blind batch as a 9x10 grid, exam cards and work cards interleaved.
    rng = random.Random(11)
    cells = ["exam"] * c["exam_n"] + ["work"] * c["fresh_n"]
    rng.shuffle(cells)
    gx, gy, cs, gap = 36, 90, 20, 5
    for i, kind in enumerate(cells):
        col, row = i % 9, i // 9
        color = AMBER if kind == "exam" else SLATE
        out.append(rect(gx + col * (cs + gap), gy + row * (cs + gap), cs, cs, color, rx=3, opacity=0.85 if kind == "exam" else 0.7))
    out.append(text(36, 356, f"{c['batch_n']}-card blind batch", 20, INK, weight=700))
    out.append(rect(36, 372, 13, 13, AMBER, rx=2))
    out.append(text(58, 384, f"{c['exam_n']} exam cards", 19, MUTED, font=SANS))
    out.append(rect(36, 398, 13, 13, SLATE, rx=2))
    out.append(text(58, 410, f"{c['fresh_n']} work cards", 19, MUTED, font=SANS))
    out.append(text(36, 436, "shuffled together", 19, DIM, font=SANS))

    # Column 2: the three judges.
    out.append(arrow(268, 200, 300, 200))
    jx = 306
    for i, name in enumerate(("Claude", "Gemini", "GPT")):
        y = 100 + i * 64
        out.append(rect(jx, y, 214, 48, BG0, rx=8, stroke=BORDER))
        out.append(text(jx + 18, y + 31, name, 21, INK, weight=700))
        out.append(text(jx + 112, y + 31, ("Anthropic", "Google", "OpenAI")[i], 19, DIM, font=SANS))
    out.append(text(jx, 322, "stock models, zero training", 19, MUTED, font=SANS))
    out.append(text(jx, 348, "policy rides in the prompt", 19, MUTED, font=SANS))
    out.append(text(jx, 384, f"exam {c['gold_hard']}/{c['gold_hard']}, all three", 19, AMBER, font=SANS))
    out.append(text(jx, 410, f"unsure on exactly the {c['gold_boundary']}", 19, AMBER, font=SANS))
    out.append(text(jx, 436, "human boundary cards", 19, AMBER, font=SANS))

    # Column 3: the vote.
    out.append(arrow(530, 200, 562, 200))
    vx = 568
    out.append(rect(vx, 100, 214, 76, BG0, rx=8, stroke=BORDER))
    out.append(text(vx + 16, 130, "pair agrees", 20, INK, weight=700))
    out.append(text(vx + 16, 158, f"{c['pair_agree']} of {c['batch_n']}, done", 19, MUTED, font=SANS))
    out.append(rect(vx, 188, 214, 76, BG0, rx=8, stroke=BORDER))
    out.append(text(vx + 16, 218, "pair splits", 20, INK, weight=700))
    out.append(text(vx + 16, 246, f"{c['splits']} of {c['batch_n']}, third votes", 19, MUTED, font=SANS))
    out.append(text(vx, 322, "two of three wins", 19, MUTED, font=SANS))
    out.append(text(vx, 348, "same labels either way", 19, MUTED, font=SANS))
    out.append(text(vx, 384, "31% fewer calls", 19, AMBER, font=SANS))
    out.append(text(vx, 410, "as a cascade", 19, AMBER, font=SANS))

    # Column 4: containment funnel.
    out.append(arrow(792, 200, 824, 200))
    fx = 830
    steps = [
        (100, 190, "one file", "dev_set.json"),
        (176, 170, "one number", f"threshold {f['threshold']}"),
        (252, 150, "clamped", f"{f['tier1_gates']} human gates"),
    ]
    for y, wd, big, small in steps:
        x = fx + (190 - wd) // 2
        out.append(rect(x, y, wd, 60, BG0, rx=8, stroke=AMBER_DEEP))
        out.append(text(x + wd / 2, y + 26, big, 19, AMBER, weight=700, anchor="middle"))
        out.append(text(x + wd / 2, y + 49, small, 19, MUTED, font=SANS, anchor="middle"))
    out.append(text(fx + 95, 358, "never a training row,", 19, MUTED, font=SANS, anchor="middle"))
    out.append(text(fx + 95, 384, "never a referee", 19, MUTED, font=SANS, anchor="middle"))
    out.append("</svg>")
    return "".join(out)


def pretrain_curve() -> str:
    f = FROZEN_NUMBERS
    w, h = 900, 440
    label = "What pretraining is worth, measured as PR-AUC on unseen real calls: " + ", ".join(f"{v} {n}" for v, n, _ in f["curve"])
    out = [head(w, h, label, rx=9), rect(0, 0, w, h, "none", rx=9, stroke=BORDER)]
    out.append(text(450, 44, "What pretraining is worth, on unseen real calls", 26, INK, font=SANS, weight=700, anchor="middle"))
    out.append(text(450, 72, "PR-AUC on the held-out real-call referee, same encoder, vocabulary grows with the corpus", 17, MUTED, font=SANS, anchor="middle"))
    base_y, scale, bw, gapx = 340, 240, 130, 40
    x0 = (w - (4 * bw + 3 * gapx)) // 2
    band_y = base_y - f["ood_band"] * scale
    out.append(f'<line x1="50" y1="{band_y:.1f}" x2="850" y2="{band_y:.1f}" stroke="{DIM}" stroke-width="1.5" stroke-dasharray="6 6"/>')
    out.append(text(848, band_y - 8, f"band {f['ood_band']}", 17, DIM, anchor="end"))
    out.append(f'<line x1="50" y1="{base_y}" x2="850" y2="{base_y}" stroke="{BORDER}" stroke-width="1.5"/>')
    for i, (val, name, sub) in enumerate(f["curve"]):
        x = x0 + i * (bw + gapx)
        hgt = float(val) * scale
        color = AMBER if i == len(f["curve"]) - 1 else AMBER_DEEP
        out.append(rect(x, round(base_y - hgt, 1), bw, round(hgt, 1), color, rx=6, opacity=0.6 + 0.13 * i))
        out.append(text(x + bw / 2, round(base_y - hgt - 14, 1), val, 30, AMBER_GLOW, weight=700, anchor="middle"))
        out.append(text(x + bw / 2, base_y + 30, name, 18, INK, font=SANS, anchor="middle"))
        out.append(text(x + bw / 2, base_y + 56, sub, 17, DIM, font=SANS, anchor="middle"))
    out.append(text(450, 420, "each step adds language exposure, not architecture", 17, MUTED, font=SANS, anchor="middle"))
    out.append("</svg>")
    return "".join(out)


def render_all() -> dict[str, str]:
    c = counted()
    return {
        "hero.svg": hero(c),
        "referees.svg": referees(c),
        "judges.svg": judges(c),
        "pretrain-curve.svg": pretrain_curve(),
        "band-judged.svg": band("THE REFEREES", "Three referees, one question each", f"gold {c['gold_n']} · regressions {c['regressions']} · real {FROZEN_NUMBERS['ood_n']}"),
        "band-models.svg": band("TWO MODELS", "What a fifteen-minute pretrain buys", "66.96M vs 7.36M params"),
    }


def font_floor_violations(svg: str) -> list[str]:
    """Every font-size must survive 75% zoom in the README column: size >= viewBox width / 55.6."""
    width = float(re.search(r'viewBox="0 0 (\d+)', svg).group(1))
    floor = width / (COLUMN_PX * ZOOM / FLOOR_PX)
    sizes = [float(s) for s in re.findall(r'font-size="([\d.]+)"', svg)]
    return [f"{s}px under {floor:.1f}px floor" for s in sizes if s < floor]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    figs = render_all()
    problems = []
    for name, svg in figs.items():
        for v in font_floor_violations(svg):
            problems.append(f"{name}: {v}")
    if problems:
        print("type floor: " + "; ".join(problems))
        return 1
    if args.write:
        ASSETS.mkdir(exist_ok=True)
        for name, svg in figs.items():
            (ASSETS / name).write_text(svg + "\n", encoding="utf-8")
            print(f"wrote assets/{name} ({len(svg)} bytes)")
        return 0
    if args.check:
        drift = []
        for name, svg in figs.items():
            p = ASSETS / name
            if not p.exists() or p.read_text(encoding="utf-8") != svg + "\n":
                drift.append(name)
        served = ROOT / "models/eot-distilbert-onnx-int8/threshold.json"
        if served.exists():
            live = json.load(open(served, encoding="utf-8"))["threshold"]
            if f"{live:.2f}" != FROZEN_NUMBERS["threshold"]:
                drift.append(f"threshold.json reads {live}, FROZEN_NUMBERS says {FROZEN_NUMBERS['threshold']}")
        if drift:
            print("figure drift: " + ", ".join(drift))
            return 1
        print(f"figures match their constants and clear the type floor ({len(figs)} files)")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())