# End-of-Turn Detector

> A production-oriented end-of-turn detection service for voice agents, built around a fine-tuned DistilBERT classifier, ONNX INT8 inference, and FastAPI.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU-005CED?style=flat-square)](https://onnxruntime.ai/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square)](LICENSE)

---

## What this project does

Voice agents need to answer one deceptively difficult question:

> **Has the caller finished speaking, or should the system keep listening?**

This project implements that decision as a text classification problem.

Given:

- the agent's previous utterance/context
- the caller's current transcript

the service predicts:

```text
SPEAK

or

WAIT

The model is a fine-tuned DistilBERT classifier exported to ONNX and quantized for CPU inference.

The production path is:

Caller Speech
     │
     ▼
   ASR
     │
     ▼
Caller Transcript
     │
     ├───────────────┐
     │               │
     ▼               ▼
Agent Context     Caller Text
     │               │
     └───────┬───────┘
             ▼
       Tokenization
             │
             ▼
    DistilBERT ONNX INT8
             │
             ▼
      Probability Score
             │
             ▼
       Threshold Gate
          /       \
         /         \
     SPEAK          WAIT

The goal is not simply to maximize classification accuracy.

The real engineering problem is deciding when the agent should take the conversational floor without talking over the caller.

Why end-of-turn detection is difficult

A transcript can look complete while the caller is still speaking.

Consider:

Agent:
"Is there anything else I can help you with?"

Caller:
"Actually yeah, one more thing..."

Grammatically, the sentence is complete.

Conversationally, the caller is not finished.

The detector therefore needs to reason about more than punctuation or sentence completion.

Important cases include:

Scenario	Example	Decision
Complete statement	"Yes, I can make it tomorrow."	SPEAK
Complete question	"Can you send me the confirmation?"	SPEAK
Acknowledgement	"Okay, got it."	SPEAK
Mid-clause	"I wanted to ask about..."	WAIT
Disfluent continuation	"Yeah so, um, the thing is..."	WAIT
Data readout	"It's four one five..."	WAIT
Connector-final	"I can come Thursday, but..."	WAIT
Announced continuation	"Actually, one more thing..."	WAIT
Self-interruption	"Can you... actually, you know what..."	WAIT
Explicit hold	"Hang on, let me check..."	WAIT

This is why the system is designed around conversation policy, not only sentence completion.

Architecture
                       ┌─────────────────────────┐
                       │       Voice Agent       │
                       └────────────┬────────────┘
                                    │
                              Agent Context
                                    │
                                    ▼
┌───────────────┐          ┌──────────────────────┐
│      ASR      │─────────▶│   FastAPI /predict   │
└───────┬───────┘          └──────────┬───────────┘
        │                             │
        │ Caller text                 ▼
        │                    ┌────────────────────┐
        └───────────────────▶│ Hugging Face       │
                             │ Tokenizer          │
                             └─────────┬──────────┘
                                       │
                                       ▼
                             ┌────────────────────┐
                             │ DistilBERT ONNX    │
                             │ INT8 CPU           │
                             └─────────┬──────────┘
                                       │
                                       ▼
                             ┌────────────────────┐
                             │ Probability        │
                             │ + Threshold        │
                             └─────────┬──────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
                    SPEAK NOW                  KEEP LISTENING
Engineering goals

The project is designed around several production concerns.

1. Low-latency CPU inference

The model is exported to ONNX and quantized to INT8 so the inference service does not require a GPU.

2. Reproducible serving

The model evaluated during deployment should be the same artifact used by the API.

Training
   ↓
Checkpoint
   ↓
ONNX export
   ↓
INT8 quantization
   ↓
Serving artifact
   ↓
Evaluation
   ↓
Production
3. Explicit decision threshold

The API does not blindly use 0.5.

Instead:

p_complete >= threshold
        │
        ├── yes ──▶ SPEAK
        │
        └── no ───▶ WAIT

The threshold is configurable through:

EOT_THRESHOLD

If the environment variable is not supplied, the service loads the threshold associated with the model artifact.

4. Serving-path correctness

A common ML deployment mistake is evaluating one model representation and serving another.

For example:

FP32 checkpoint
      ↓
evaluation
      ↓
good score

while production actually runs:

INT8 ONNX
      ↓
different numerical behavior
      ↓
different score

This project explicitly evaluates the ONNX serving path.

Model

The current inference model is based on:

DistilBERT
    ↓
Fine-tuning
    ↓
ONNX export
    ↓
INT8 quantization
    ↓
ONNX Runtime

The model directory is expected at:

models/
└── eot-distilbert-onnx-int8/
    ├── model.onnx
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── config.json
    └── ...

The exact model architecture and tokenizer configuration are loaded from the model artifact rather than being hard-coded into the API.

Important implementation detail: ONNX tensor dtypes

One of the deployment issues addressed in this project is the mismatch between tokenizer output types and ONNX input types.

For example, a tokenizer may produce:

int32

while an ONNX graph expects:

int64

Passing the wrong tensor dtype can result in an inference failure such as:

Unexpected input data type.
Actual: tensor(int32)
Expected: tensor(int64)

The serving layer therefore inspects the ONNX input metadata and prepares tensors using the expected types.

Conceptually:

onnx_inputs = {
    name: value.astype(np.int64)
    for name, value in encoded.items()
    if name in INPUT_NAMES
}

This is important because the inference service must respect the deployed model contract rather than assuming that tokenizer output types automatically match the ONNX graph.

API
POST /predict

Predict whether the caller has completed their turn.

Request
{
  "context": "What is your MC number?",
  "text": "yeah it is four one five"
}
Response
{
  "p_complete": 0.12,
  "decision": "wait",
  "threshold": 0.42,
  "model_latency_ms": 8.7,
  "request_id": "9b4e7e7c-..."
}

The values above are illustrative. Do not interpret them as benchmark results.

Health endpoint
GET /healthz

Used by local development, container orchestration, and load balancers.

Example:

curl http://127.0.0.1:8000/healthz

Expected response:

{
  "status": "ok"
}
Interactive UI

The project includes a browser-based inference interface.

Start the service:

python -m uvicorn serve:app --reload --host 127.0.0.1 --port 8000

Then open:

http://127.0.0.1:8000/

The UI is designed as a lightweight inference laboratory rather than a generic form.

It provides:

Agent context input
Caller utterance input
Probability visualization
Threshold visualization
SPEAK NOW / KEEP LISTENING decision
Model latency
Request ID
Health status
Clear/reset controls
Keyboard shortcut for inference
Responsive layout

The frontend is intentionally kept separate from the inference implementation:

index.html
     │
     ▼
FastAPI
     │
     ▼
/predict
Project structure
fine-tuning-turn-detection-model/
│
├── serve.py
├── common.py
├── index.html
├── favicon.svg
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── eot-distilbert-onnx-int8/
│       ├── model.onnx
│       ├── config.json
│       ├── tokenizer.json
│       └── ...
│
├── tests/
│   ├── test_health.py
│   ├── test_predict.py
│   └── test_inputs.py
│
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   └── deployment.md
│
└── assets/
    ├── architecture.png
    └── screenshots/

Not every directory needs to exist on day one. The structure is intended to keep the repository organized as the project grows.

Local development
1. Clone
git clone https://github.com/YOUR_USERNAME/fine-tuning-turn-detection-model.git
cd fine-tuning-turn-detection-model
2. Create a virtual environment

Windows PowerShell:

python -m venv .venv

Activate:

.\.venv\Scripts\Activate.ps1

If PowerShell blocks activation:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Then:

.\.venv\Scripts\Activate.ps1
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
4. Start the API
python -m uvicorn serve:app --reload --host 127.0.0.1 --port 8000

Open:

http://127.0.0.1:8000/

API documentation:

http://127.0.0.1:8000/docs
Configuration

The service supports environment-based configuration.

Variable	Purpose	Default
EOT_MODEL_DIR	Model directory	models/eot-distilbert-onnx-int8
EOT_MAX_LEN	Maximum tokenizer length	128
EOT_THRESHOLD	Override decision threshold	Model threshold

Example:

$env:EOT_THRESHOLD="0.42"
$env:EOT_MAX_LEN="128"

Then:

python -m uvicorn serve:app --host 127.0.0.1 --port 8000
API example

PowerShell:

$body = @{
    context = "What is your MC number?"
    text = "yeah it is four one five"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body

Linux/macOS:

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "context": "What is your MC number?",
    "text": "yeah it is four one five"
  }'
Decision policy

The model output is converted into an operational decision.

                 p_complete
                     │
                     ▼
              ┌─────────────┐
              │  threshold  │
              └──────┬──────┘
                     │
          ┌──────────┴──────────┐
          │                     │
      p >= threshold       p < threshold
          │                     │
          ▼                     ▼
        SPEAK                  WAIT

This separation is intentional.

The neural network estimates:

P(completed turn)

The application policy determines:

What should the voice agent do?

Keeping these concerns separate makes threshold tuning and policy changes safer.

Evaluation strategy

A production turn detector should not be evaluated only with accuracy.

Important measurements include:

Precision

How often does the system correctly identify completed turns?

Recall

How many completed turns does the detector recover?

PR-AUC

Useful when the positive and negative classes are not perfectly balanced.

False speak rate

How often does the agent incorrectly interrupt the caller?

This is particularly important for voice applications.

False wait rate

How often does the agent unnecessarily continue listening after the caller has finished?

Latency

The detector must operate within the conversational latency budget.

Evaluation dataset design

The evaluation strategy should contain separate slices.

Training
   │
   ├── synthetic / policy-driven examples
   │
   └── development examples
            │
            ▼
      Threshold tuning
            │
            ▼
      Frozen evaluation
            │
            ▼
      Held-out real calls

The most important rule is:

Do not tune the threshold on the same examples used to report final performance.

A clean evaluation setup should separate:

TRAIN
DEV
FROZEN TEST
REAL-CALL HOLDOUT

This reduces the risk of reporting an optimistic score.

Threshold selection

The threshold is an operational parameter.

A threshold that is optimal for generic classification accuracy may not be optimal for a voice agent.

For example:

False SPEAK
    ↓
Agent interrupts caller
    ↓
Poor conversational experience

while:

False WAIT
    ↓
Agent waits too long
    ↓
Increased response latency

These errors can have different costs.

Therefore threshold selection should consider:

classification performance
+
false-speak cost
+
false-wait cost
+
regression cases
+
serving-path behavior

The final threshold should be selected against the actual deployed inference artifact.

Calibration and monitoring

Probability scores are not automatically calibrated.

A model returning:

0.80

does not necessarily mean:

80% probability

Therefore production monitoring should track:

probability distribution
threshold crossings
false speak rate
false wait rate
disagreement with downstream turn-taking
latency percentiles
model version
ASR version
language distribution
transcript length distribution

A useful production metric is the disagreement rate between the detector and the incumbent turn-taking system.

ASR
 │
 ▼
Current turn detector ─────┐
                           │
                           ▼
                     disagreement
                           │
                           ▼
                    human review
                           │
                           ▼
                    new evaluation
                           │
                           ▼
                       retraining
Observability

Every prediction receives a request ID.

Example:

X-Request-ID:
9b4e7e7c-...

The response also includes:

{
  "request_id": "9b4e7e7c-..."
}

This allows an individual prediction to be traced through:

Client
  ↓
FastAPI
  ↓
Tokenizer
  ↓
ONNX Runtime
  ↓
Decision

Production logging should capture metadata rather than raw caller content whenever possible.

Security and privacy

Voice-agent transcripts can contain sensitive information.

The service therefore should follow these principles:

Do not log raw caller transcripts by default

Avoid:

INFO caller_text="My SSN is..."

Prefer:

INFO request_id=... latency_ms=... decision=wait
Do not commit private recordings

Keep:

data/raw/
data/private/
data/real_calls/

out of Git.

Do not commit secrets

Keep:

.env
.env.*
secrets/

out of Git.

Minimize stored inference data

If prediction logs are required for debugging, use retention limits and redact sensitive information.

Performance

The project is designed for CPU inference.

The serving stack is:

FastAPI
   ↓
Tokenizer
   ↓
ONNX Runtime
   ↓
INT8 DistilBERT
   ↓
CPU

Benchmarking should report at least:

Concurrency
Requests/sec
p50
p95
p99
Model latency
End-to-end latency
CPU utilization
Memory usage

Example benchmark table:

Concurrency	Throughput	p50	p95	p99
1	TBD	TBD	TBD	TBD
4	TBD	TBD	TBD	TBD
8	TBD	TBD	TBD	TBD
16	TBD	TBD	TBD	TBD

Replace TBD with measurements from your own benchmark before publishing performance claims.

Why ONNX INT8?

The project uses ONNX Runtime because the deployment target is CPU inference.

Advantages include:

portable inference artifact
optimized graph execution
CPU execution
reduced model size
INT8 quantization
predictable serving environment
separation between training and inference

The production artifact is therefore:

PyTorch / Transformers
        │
        ▼
     ONNX
        │
        ▼
     INT8
        │
        ▼
 ONNX Runtime
        │
        ▼
     FastAPI

The API does not need the full training stack to execute the model.

Failure modes

A senior production system should explicitly identify failure modes.

1. Transcript truncation

Long context may exceed the model's maximum sequence length.

Mitigation:

MAX_LEN

is explicitly configurable.

2. ASR errors

The detector operates on transcripts.

If ASR produces:

"yeah I can make it tomorrow"

instead of:

"yeah I can make it tomorrow but..."

the detector receives incorrect evidence.

This is an input limitation rather than purely a model limitation.

3. Prosody loss

Text-only models cannot directly observe:

pitch
duration
pause length
speaking rate
final-word lengthening
intonation

For example, these two transcripts can look identical:

"you need the receipt"

but the audio may distinguish:

statement

from:

question

A future version can combine text and audio features.

4. Domain shift

A model trained on one conversational domain may behave differently on:

customer support
logistics
healthcare
finance
sales
multilingual calls

Production evaluation should therefore include domain-specific holdouts.

Future multimodal architecture

The natural evolution of the project is a multimodal end-of-turn detector.

                    Caller Audio
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
        ASR Transcript          Audio Features
             │                       │
             ▼                       ▼
       Text Encoder             Audio Encoder
             │                       │
             └──────────┬────────────┘
                        ▼
                   Fusion Layer
                        │
                        ▼
                 Turn Probability
                        │
                        ▼
                 Policy Threshold
                   /          \
               SPEAK          WAIT

Potential audio features:

pause duration
final syllable duration
pitch contour
speaking rate
energy
voice activity
word timing

This addresses one of the fundamental limitations of text-only end-of-turn detection.

Production deployment

A production deployment can evolve into:

                    Load Balancer
                         │
                         ▼
                  FastAPI Service
                         │
                ┌────────┴────────┐
                ▼                 ▼
           ONNX Runtime       Metrics
                │                 │
                ▼                 ▼
            CPU Nodes       Prometheus
                                  │
                                  ▼
                              Grafana

Containerized deployment:

Docker
  │
  ▼
FastAPI
  │
  ▼
ONNX Runtime
  │
  ▼
INT8 model

For larger deployments:

Kubernetes
    │
    ├── API pods
    ├── autoscaling
    ├── health probes
    ├── rolling deployments
    └── model versioning
Model versioning

Every production model should have an explicit version.

Example:

models/
└── eot-distilbert-onnx-int8/
    ├── model.onnx
    ├── config.json
    ├── tokenizer.json
    ├── threshold.json
    └── metadata.json

Recommended metadata:

{
  "model_name": "eot-distilbert",
  "format": "onnx",
  "quantization": "int8",
  "max_length": 128,
  "threshold": null,
  "created_at": null,
  "dataset_version": null
}

Populate these fields with real values before publishing.

Testing strategy

The project should maintain tests at multiple levels.

Unit tests

Test:

request validation
probability calculation
threshold behavior
tensor dtype conversion
configuration loading
API tests

Test:

POST /predict
GET /healthz
GET /
Model contract tests

Verify:

model exists
tokenizer loads
ONNX inputs match tokenizer
ONNX outputs exist
Regression tests

Maintain a small set of conversational examples representing known failure modes.

For example:

"Actually, one more thing..."
"Hang on, let me check..."
"Yeah, I can make it tomorrow."
"Can you send that again?"

Every model update should run these before deployment.

CI/CD direction

A production CI pipeline should look like:

Pull Request
     │
     ▼
Lint
     │
     ▼
Unit Tests
     │
     ▼
API Tests
     │
     ▼
Model Contract Test
     │
     ▼
Regression Tests
     │
     ▼
Benchmark
     │
     ▼
Docker Build
     │
     ▼
Security Scan
     │
     ▼
Deploy

The important principle is:

A model change should be treated as a software release, not only as an experiment.

Reproducibility

A model result is only useful if another engineer can reproduce the environment.

Pin:

Python version
dependencies
model artifact
tokenizer
threshold
dataset version
evaluation configuration

Recommended environment:

Python 3.9.x

The project currently targets Python 3.9 compatibility.

Development commands

Start the server:

python -m uvicorn serve:app --reload --host 127.0.0.1 --port 8000

Run without reload:

python -m uvicorn serve:app --host 0.0.0.0 --port 8000

Check the API:

curl http://127.0.0.1:8000/healthz

Open Swagger:

http://127.0.0.1:8000/docs

Open the application:

http://127.0.0.1:8000/
Known limitations

This version is intentionally text-only.

Current limitations include:

No direct audio/prosody input
ASR errors propagate into the detector
Threshold requires evaluation against representative data
Model quality depends on training/evaluation data
Multilingual performance requires dedicated evaluation
CPU benchmark numbers are hardware dependent
Production drift monitoring is not yet included
No automatic model registry integration yet

These are engineering constraints, not hidden assumptions.

Roadmap
Phase 1 — Current
 DistilBERT classifier
 ONNX export
 INT8 CPU inference
 FastAPI service
 Health endpoint
 Request IDs
 Browser inference UI
 Configurable threshold
 ONNX input dtype handling
Phase 2 — Evaluation
 Frozen evaluation dataset
 Precision / recall
 PR-AUC
 Calibration analysis
 False-speak analysis
 False-wait analysis
 Threshold sweep
 Regression suite
Phase 3 — Production
 Docker image
 CI/CD
 Prometheus metrics
 Grafana dashboard
 Load testing
 Model versioning
 Automated regression checks
Phase 4 — Multimodal
 Word timings
 Pause features
 Prosody features
 Audio encoder
 Text/audio fusion
 Streaming inference
What this project demonstrates

This project is intentionally more than:

train model → save model → call predict()

It demonstrates the complete ML engineering path:

Problem definition
       ↓
Conversation policy
       ↓
Dataset design
       ↓
Model fine-tuning
       ↓
Model export
       ↓
Quantization
       ↓
Inference contract
       ↓
FastAPI serving
       ↓
Threshold selection
       ↓
Evaluation
       ↓
Regression testing
       ↓
Observability
       ↓
Deployment

The key engineering lesson is:

The model is only one component of a production ML system.

The difficult part is making the model's behavior measurable, reproducible, observable, and safe enough to operate inside a real-time conversational system.

Research questions

This project can be extended around several practical research questions:

How much does agent context improve end-of-turn detection?
How much performance is lost through INT8 quantization?
How sensitive is the operating threshold to domain shift?
How does ASR punctuation affect classification?
Can word timing features improve text-only detection?
Can audio prosody resolve ambiguous transcript cases?
How should false-speak and false-wait costs be optimized?
Can disagreement mining provide an efficient human-labeling strategy?
How does performance change across languages?
Can the detector operate reliably in streaming conditions?
Engineering principles
Measure the artifact that ships

Do not evaluate FP32 and deploy INT8 without checking the difference.

Separate probability from policy

The model estimates a probability.

The application decides what action to take.

Keep evaluation frozen

Do not continuously modify the test set until the metrics look good.

Treat regressions as first-class data

Every discovered failure should have a path into the regression suite.

Optimize the complete system

A fast model is not enough if:

tokenization
+
network
+
queueing
+
inference
+
serialization

still violates the latency budget.

Make failure visible

Production systems should expose:

request ID
model version
latency
decision
threshold
health

rather than silently failing.

License

This project is intended to be released under the Apache-2.0 license.

Add the license file to the repository before publishing:

LICENSE
Author

Harishchandra Chaudhary

Software Engineer | Python | FastAPI | React | Machine Learning | AI Engineering

GitHub:

https://github.com/HarishchandraChaudhary

LinkedIn:

https://www.linkedin.com/in/harishchaudhary-dev/

Topics
python
machine-learning
deep-learning
nlp
transformers
distilbert
onnx
onnxruntime
int8
quantization
fastapi
voice-ai
conversational-ai
end-of-turn-detection
mlops
model-serving
Status

This repository is an engineering project focused on production-oriented end-of-turn detection for voice-agent systems.

The benchmark numbers in this README should be populated only from reproducible measurements of the current model artifact and serving environment.
