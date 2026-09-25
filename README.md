Turn Detection Model
An end-of-turn detector for voice agents. Given the agent's last line and the caller's words so far, it decides whether the caller is done talking or still mid-turn.

Fine-tuned DistilBERT, shipped as int8, served at a picked operating threshold of 0.42.

Gold PR-AUC (60-card frozen set)	0.949
Gold recall	0.654
Held-out real calls (96 turns)	0.913 PR-AUC
False-speak on real calls	11 / 47 wait turns (0.234)
False-speak on gold wait cards	0 / 27
End-to-end p95 latency (concurrency 8)	33.1 ms
The finding this repo exists to report: the model is near-perfect on the frozen gold set and meaningfully worse on real calls. That gap — 0.949 → 0.913 PR-AUC, 0 → 11 false speaks — is the headline result, not a footnote. Read it as "this is what generalization actually costs," not as a bug to be hidden.

Quickstart
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

make synth        # regenerate the English training set (seeded, byte-identical)
make tier1        # derive the twelve guardrail rows from the committed data
make train         # fine-tune the DistilBERT lane, export ONNX + int8
make threshold      # re-pick the operating point on the served int8 file, one row per call
make eval          # score a model against the frozen gold set
make serve         # FastAPI on :8000, with a live probe page at /
make bench         # async stress test, latency percentiles + throughput
make docker-build && make docker-run && make smoke   # int8 model in a container
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"context": "What is your MC number?", "text": "yeah it is four one five"}'
Optional from-scratch lane (no pretrained weights, tokenizer trained from the corpus):

make corpus        # one-time, builds the scratch lane's data (pulls in `datasets`)
make pretrain       # ~15 min, our own masked-language-model base
make scratch        # fine-tune that base into the 7.36M from-scratch model
What a clean clone actually gives you
The synthetic data rebuilds byte-for-byte, and the eight probe cases on the / page reproduce wait / speak / wait exactly as documented.
The numbers will differ — this is expected, not a bug. One clean retrain read 0.965 gold on its fp32 export and 0.966 on int8, and picked thresholds of 0.71 and 0.63 against the frozen 0.42. That's one set of weights scored through two execution paths, not run-to-run noise.
Real-call data never leaves the author's machine. make train from a clone rebuilds the pre-augmentation fine-tune, which reads 0.65 on real calls where the shipped artifact reads 0.91. If you're auditing this repo, that 0.26-point gap is the real-call augmentation working as intended, not something withheld.
Every number in this README describes the v9 freeze. Your box will train something adjacent to it, not identical.
Why this exists: five hard problems
#	Problem	Where this stack stands
1	A complete sentence is not a complete turn. "Anything else?" → "actually yeah, one more thing." is grammatically finished and conversationally wide open.	Answered. Announced continuation is its own policy class and a tier-1 constraint. That exact card scores 0.035 and holds.
2	The two failure modes cost differently. Talking over a caller and leaving the line hanging are not equally bad, so accuracy is the wrong objective.	Answered. A 1:5 cost ratio (false-speak : false-wait) picks the threshold — 0.42 — where the model never talks over any of the 27 gold wait cards, and talks over 11 of 47 on held-out real calls.
3	There is no ground truth, only a policy. An unwritten label set is one person's ear.	Answered. POLICY.md came first; 60 cards were blind-labeled against it, and three vendor judge models hit 53/53 agreement.
4	The model you measure is not the model you ship. Quantization moves scores near the threshold.	Answered, after two red iterations. One card read 0.26 on the fp32 checkpoint and 0.412 through the actual serving path. Threshold selection now scores one row at a time, the way serving does — not batched.
5	Text has no prosody. Falling pitch and a trailing vowel never reach a transcript.	Not answerable here. This is a ceiling on the input, not a model defect. See Roadmap.
Model comparison
Every probe below is scored on the served int8 artifact, one row at a time — the way the API actually scores a call. The full 36-probe page lives in the repo; 35 are graded and one is a boundary card the policy itself calls unsure.

Fine-tuned DistilBERT (66.96M)	From-scratch (7.36M)
Matches written policy (35 graded probes)	31/35	34/35
Mean model latency	17.8 ms	2.9 ms
Spanish probes (6)	0.770 mean, 3 wrong	All 6 correct
English probes	28/35 (tied)	28/35 (tied)
Both models share the same single English miss: an unpunctuated yes/no question. Latency is one run on one laptop and moves ~1 ms per regeneration — treat it as directional, not a benchmark claim.

The fine-tune ships despite losing the probe page, because probes are a fixed 36-card sample and the fine-tune leads where it matters: unseen real calls (0.91 vs. the from-scratch lane's numbers on the same slice). Pretraining data volume tracks directly with real-call PR-AUC: 0.48 random init → 0.60 with real calls added → 0.83 with a 15-minute pretrain → 0.91 web-pretrained.

How the operating point is chosen
Sweep every threshold on the judged dev cards; keep the lowest cost, counted as 5 false speaks : 1 false wait.
Discard any threshold that breaks one of twelve pinned tier-1 cards. If none survive, fail loud — don't silently relax a guardrail.
Score the artifact the way it ships: int8, one row per call, not batched. The winner is written to threshold.json; serve.py reads it at startup.
Threshold history (v1 → v9)
This is a build log, not nine independently checkable results — each run overwrote the previous report, and only v9 regenerates from the committed artifact. The last three rows (v7–v9) share one set of weights; only the measuring instrument changed.

Run	Threshold	Gold PR-AUC	Gold recall	False-speak	ECE	Lesson
v1	0.833	0.961	0.577	0.00	—	Textbook 5:1 bar assumes calibration; this model runs under-confident.
v2	0.61	0.964	0.808	0.00	0.114	Picking on the measured curve moved recall.
v3	0.87	0.970	0.654	0.00	0.067	Synthetic validation pushed the dial high; synthetic speech is easier than real speech.
v4	0.86	0.969	0.654	0.00	0.070	Switching to rates instead of counts didn't fix it — the synthetic set itself was the problem.
v5	0.81	0.958	0.731	0.00	0.092	A human typed "nah bye" and got wait. Casual speech joined the training set; regression file born.
v6	0.18	0.955	0.654	0.037	0.169	Judged dev cards replaced synthetic validation; the dial collapsed and "one more thing" got interrupted.
v7	0.27	0.949	0.654	0.00	0.160	Twelve cards became hard constraints. Green on fp32, red on the int8 that actually serves.
v8	0.40	0.949	0.654	0.00	0.160	Re-picked on int8: 11/12 pinned cards. The picker was batching where serving scores one at a time.
v9	0.42	0.949	0.654	0.00	0.160	Scored one card at a time, matching serving exactly: 12/12 pinned cards, real calls 0.913, recall 0.959.
The v6→v9 arc is the part worth reading closely if you're building something similar: a threshold that's correct on a batched fp32 checkpoint can be wrong on the int8 artifact that actually serves traffic. Score the thing you ship, at the granularity you ship it.

Labeling: how the dev set was judged
60 gold cards with known human answers were hidden among 30 fresh cards. Three stock vendor judge models, with no task-specific training, voted two-of-three majority. Judge output feeds exactly one file, which tunes exactly one number, clamped by twelve human-set gates — the judges pick a threshold, they don't touch the policy or the constraints.

Blind, with one caveat: the policy spec quotes a few boundary examples, so certification was partially open-book, not fully blind.
Full mechanics and raw votes: docs/judge-cascade-replay.md.
The policy, as classes
Sixty blind labels collapsed into eleven classes, each with a written rule. synth.py's template banks are the policy — the generator doesn't approximate the rules, it encodes them.

Class	Shape	Example	Decision
A	Complete statement	"Hey, I'm calling to confirm the pickup for load four seven two tomorrow morning."	Speak
B	Complete question	"What's the detention policy if I'm stuck at the dock past two hours?"	Speak
C	Bare acknowledgement	"Okay, got it."	Speak — complete, but never treated as a call-ender
D	Mid-clause cutoff	"Can you tell the receiver that my ETA is now…"	Wait
E	Disfluent trail	"Yeah so, um, the thing is, uh…"	Wait
F	Mid-data readout	"Yeah, it's seven one five…" (after "Can I get your MC number?")	Wait — absolute, however long the pause runs
G	Connector-final	"I can pick up Thursday morning, but…"	Wait
H	Complete, then maybe more	"Yeah, I can make it." speaks; "Actually yeah, one more thing." holds	Speak, unless continuation is announced
I	Trailing hedge	"That's all I need, I guess…"	Speak by default; a decorative softener on an owned claim holds
J	Self-interrupt / restart	"Can you- actually, you know what…" holds; "I need the- no, scratch that…" speaks	Wait; a full retraction may earn a brief acknowledgement
K	Explicit hold	"Hang on, let me grab the load number…" holds; "Hold on, the receiver is waving at me…" speaks	Self-retrieval holds silently; a narrated outside interruption gets a courtesy ack
Class I is the one I'd defend on a whiteboard. "The broker said it was covered, supposedly…" should speak; "The detention was approved, or something…" should wait. The rule is ownership, not hedging: an attributed claim is a question in disguise, an owned claim with a softener is just a statement, and attribution markers are surface features a model can actually learn to detect.

Known weak spots (judgment calls, not edge cases)
Reported-speech hedges (class I): 1/5 scored hedge cards correct (0.20). Class I has 8 gold cards total, 3 of which the policy itself marks unsure.
Explicit holds (class K): sits at ~50%.
The bar for every judgment class is 0.60 recall; EVALS.md tracks both classes until they clear it. These are flagged here deliberately rather than smoothed over — don't ship a claim this repo can't back with a number.
Data augmentation (why the training set isn't just the policy rules verbatim)
Complete utterances are cut off mid-sentence and relabeled wait — the exact shape a live ASR partial arrives in.
Everything ships lowercased, punctuation stripped, so nothing can cheat off a period.
Contexted rows are also emitted bare, so the model works with or without the agent's last line.
From v6 onward, real-call rows join training at 4× weight, grouped by call so no call leaks across the train/referee split.
Dataset counts: 1,586 training rows · 60 gold cards · 30 judged dev cards · 6 regression cards · 400 real turns from 59 calls, split by call into 304/40 (train) and 96/19 (referee).

Serving
POST /predict — body {context, text} — returns {p_complete, decision, threshold, model_latency_ms}. ONNX Runtime, dynamic int8, CPU-only. The Docker image serves only the int8 artifact — there's no fp32 fallback in production.

Both rows below bench the same shipped int8 file, wall-clock time via bench.py. Two box states are quoted on purpose, because they disagree — the disagreement is about machine load, not model behavior.

Box state	C1 req/s (p95)	C8 req/s (p95)	Model p50 @ C8	Model p95 @ C8
Idle (~4/18 cores loaded)	55 req/s, 20.7 ms	312 req/s, 33.1 ms	22.4 ms	30.4 ms
Mid training load	28 req/s, 42.5 ms	170 req/s, 57.9 ms	39.5 ms	48.2 ms
The headline 33.1 ms is the worst of four passes on the idle row. Even the degraded 57.9 ms — the same file, benched while a training job holds the box — clears a 100 ms budget with margin.

Referees
Three independent checks, each answering a different question:

Frozen 60-card gold set — generalization against the written policy.
6 probe-found regressions — memory; does a fix stay fixed.
96 held-out real-call turns — discovery; what actually breaks in production.
Real-call files, and every report/model directory except the two committed ones, stay out of git — no real-call number regenerates from a clone. The gold, regression, pinned-card, threshold, and judge numbers all do regenerate, for both the shipped int8 lane and the committed 7.36M from-scratch lane.

The threshold itself is not a hand-picked constant — it's a dial the measured cost curve turns, using a written cost ratio, scored on the exact artifact that ships.

Repo map
synth.py                  policy-driven English generator (template banks ARE the policy)
synth_scale.py             same templates, larger slot pools, ~10x volume
synth_es.py                Spanish banks under the same policy, Spanglish register included
ood_from_elevenlabs.py     real-call eval slice builder (self-labeling turns; local only)
train.py                   fine-tune lane (DistilBERT or any HF encoder via --base)
train_scratch.py           from-scratch lane: byte-level BPE tokenizer + small encoder
pretrain_scratch.py        masked-language-model pretraining for the from-scratch lane
fetch_pretrain_corpus.py   license-clean bilingual Wikipedia slices for scratch pretrain
evaluate.py                gold-set and jsonl evaluation: sweeps, classes, calibration, stability
pick_threshold.py          dev-set threshold selection under tier-1 guardrails, scored single-row on the served artifact
serve.py                   FastAPI serving over ONNX int8, plus the live probe page
bench.py                   async stress harness, stepped concurrency
probe_compare.py           side-by-side probe page for two served models (docs/probe-comparison.html)
judge_cascade_replay.py    replays the dev-set labeling panel two ways over recorded votes, checks labels match
draw_figures.py            emits every figure from counted constants; --check fails CI on drift or a font under the 75% floor
labeling-booth.html        the calibration booth the gold set was labeled in
assets/                    figures and the eight probe clips
data/                      gold set (frozen), generated training sets, judge votes, dataset card
docs/                      approach doc, judge replay, probe page, video script
Depth documents, next to the code:

docs/approach.md — the full write-up
POLICY.md — the label rules
EVALS.md — live tracking of the two judgment classes below bar
iterations.md — the v1–v9 history in full
data/README.md — dataset card
Honest limitations
Text has no prosody. Falling pitch, trailing vowels, and breath patterns never reach a transcript. This is the model's real ceiling, not something more training data fixes — see Roadmap for the audio-path options under consideration.
Reported-speech hedges and explicit holds are still below the 0.60 bar (see Known weak spots above). Don't treat this repo as claiming those classes are solved.
Real-call numbers are not independently reproducible from a clone by design — the calling data is private. Treat the gold-set and regression numbers as the reproducible ground truth, and the real-call numbers as reported, audited results.
Latency figures are single-machine, wall-clock, and will vary with hardware — treat them as directional.
Roadmap
Prosody is the acknowledged gap. The next iteration is expected to bring in an audio signal alongside text — three transcriber-route options are under evaluation; see docs/approach.md for the current state of that comparison.
