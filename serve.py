"""Production-oriented FastAPI serving for the end-of-turn detector.

Responsibilities:
- Load the ONNX model and tokenizer once at startup.
- Validate model inputs.
- Convert tokenizer tensors to the dtypes expected by ONNX Runtime.
- Expose /predict, /healthz and a lightweight browser UI.
- Keep inference synchronous and CPU-efficient.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from common import LABEL2ID, build_input, load_threshold


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------



logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("eot-detector")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = Path(
    os.environ.get(
        "EOT_MODEL_DIR",
        str(BASE_DIR / "models" / "eot-distilbert-onnx-int8"),
    )
)

MODEL_PATH = MODEL_DIR / "model.onnx"

MAX_LEN = int(os.environ.get("EOT_MAX_LEN", "128"))

# IMPORTANT:
# Do not write:
#
#   os.environ.get("EOT_THRESHOLD", load_threshold(...))
#
# because Python evaluates load_threshold(...) before os.environ.get().
#
threshold_env = os.environ.get("EOT_THRESHOLD")

if threshold_env is not None:
    try:
        THRESHOLD = float(threshold_env)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid EOT_THRESHOLD value: {threshold_env!r}"
        ) from exc
else:
    THRESHOLD = float(load_threshold(MODEL_DIR))


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="End-of-Turn Detector",
    description="ONNX Runtime inference API for end-of-turn detection.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def create_session() -> ort.InferenceSession:
    """Create and configure the ONNX Runtime inference session."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {MODEL_PATH}"
        )

    session_options = ort.SessionOptions()

    # Keep CPU inference predictable for a lightweight API service.
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    logger.info("Loading ONNX model: %s", MODEL_PATH)

    inference_session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    logger.info(
        "ONNX providers: %s",
        inference_session.get_providers(),
    )

    return inference_session


try:
    session = create_session()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

except Exception:
    logger.exception("Failed to initialize model")
    raise


# ---------------------------------------------------------------------------
# ONNX input metadata
# ---------------------------------------------------------------------------

SESSION_INPUTS = {
    item.name: item
    for item in session.get_inputs()
}

INPUT_NAMES = set(SESSION_INPUTS.keys())


logger.info(
    "Model inputs: %s",
    {
        name: {
            "type": item.type,
            "shape": item.shape,
        }
        for name, item in SESSION_INPUTS.items()
    },
)

logger.info(
    "Configuration | model=%s | threshold=%.4f | max_len=%d",
    MODEL_DIR,
    THRESHOLD,
    MAX_LEN,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request payload for end-of-turn prediction."""

    context: str = Field(
        default="",
        description="Previous agent context.",
        max_length=4000,
    )

    text: str = Field(
        ...,
        description="Current caller utterance.",
        min_length=1,
        max_length=4000,
    )


class PredictResponse(BaseModel):
    """Prediction response."""

    p_complete: float
    decision: str
    threshold: float
    model_latency_ms: float
    request_id: str


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""

    logits = np.asarray(logits, dtype=np.float32)

    shifted = logits - np.max(
        logits,
        axis=-1,
        keepdims=True,
    )

    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values,
        axis=-1,
        keepdims=True,
    )


def prepare_onnx_inputs(
    encoded: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Prepare tokenizer outputs for ONNX Runtime.

    The current ONNX model expects integer inputs as int64.
    Transformers can return int32 numpy arrays, which causes:

        INVALID_ARGUMENT:
        Actual tensor(int32), expected tensor(int64)

    Therefore we explicitly cast according to the ONNX model metadata.
    """

    onnx_inputs: Dict[str, np.ndarray] = {}

    for name, value in encoded.items():

        # Ignore tokenizer outputs that the ONNX model does not consume.
        if name not in INPUT_NAMES:
            continue

        model_input = SESSION_INPUTS[name]

        expected_type = model_input.type

        if expected_type == "tensor(int64)":
            value = value.astype(np.int64, copy=False)

        elif expected_type == "tensor(int32)":
            value = value.astype(np.int32, copy=False)

        elif expected_type == "tensor(float)":
            value = value.astype(np.float32, copy=False)

        else:
            raise RuntimeError(
                f"Unsupported ONNX input type "
                f"for '{name}': {expected_type}"
            )

        onnx_inputs[name] = value

    # Verify that every model input is available.
    missing_inputs = INPUT_NAMES - set(onnx_inputs.keys())

    if missing_inputs:
        raise RuntimeError(
            "Missing ONNX inputs: "
            + ", ".join(sorted(missing_inputs))
        )

    return onnx_inputs


def validate_probability(value: float) -> float:
    """Keep probability numerically safe."""

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
):
    """Attach a request ID and log request duration."""

    request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4()),
    )

    request.state.request_id = request_id

    start = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        logger.exception(
            "Request failed | request_id=%s | method=%s | path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise

    duration_ms = (
        time.perf_counter() - start
    ) * 1000.0

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "%s %s -> %s | %.2f ms | request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
    )

    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/healthz",
    tags=["system"],
)
def healthz() -> dict:
    """Liveness/readiness endpoint."""

    return {
        "status": "ok",
        "service": "eot-detector",
        "model_dir": str(MODEL_DIR),
        "model_exists": MODEL_PATH.exists(),
        "threshold": THRESHOLD,
        "max_len": MAX_LEN,
        "providers": session.get_providers(),
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
)
def predict(
    request: PredictRequest,
) -> PredictResponse:
    """Run end-of-turn inference."""

    request_id = getattr(
        request,
        "request_id",
        None,
    )

    model_input = build_input(
        request.context,
        request.text,
    )

    start = time.perf_counter()

    try:
        # ---------------------------------------------------------------
        # Tokenization
        # ---------------------------------------------------------------

        encoded = tokenizer(
            model_input,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="np",
        )

        # ---------------------------------------------------------------
        # ONNX input preparation
        # ---------------------------------------------------------------

        onnx_inputs = prepare_onnx_inputs(encoded)

        logger.debug(
            "ONNX inputs: %s",
            {
                name: {
                    "dtype": str(value.dtype),
                    "shape": value.shape,
                }
                for name, value in onnx_inputs.items()
            },
        )

        # ---------------------------------------------------------------
        # Model inference
        # ---------------------------------------------------------------

        outputs = session.run(
            None,
            onnx_inputs,
        )

        if not outputs:
            raise RuntimeError(
                "ONNX model returned no outputs."
            )

        logits = np.asarray(
            outputs[0],
            dtype=np.float32,
        )

        if logits.ndim != 2:
            raise RuntimeError(
                f"Unexpected logits shape: {logits.shape}"
            )

        probabilities = softmax(logits)

        speak_index = LABEL2ID["speak"]

        if speak_index >= probabilities.shape[1]:
            raise RuntimeError(
                f"Label index {speak_index} is outside "
                f"model output shape {probabilities.shape}"
            )

        p_complete = validate_probability(
            float(probabilities[0][speak_index])
        )

        decision = (
            "speak"
            if p_complete >= THRESHOLD
            else "wait"
        )

        model_latency_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return PredictResponse(
            p_complete=p_complete,
            decision=decision,
            threshold=THRESHOLD,
            model_latency_ms=round(
                model_latency_ms,
                3,
            ),
            request_id=request_id or "",
        )

    except Exception as exc:

        logger.exception(
            "Inference failed | request_id=%s | error=%s",
            request_id,
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "model_inference_failed",
                "message": str(exc),
                "request_id": request_id,
            },
        ) from exc


# ---------------------------------------------------------------------------
# Browser UI
# ---------------------------------------------------------------------------

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>End-of-Turn Live Probe</title>

<style>

:root {
  --bg: #1D1512;
  --card: #2E241C;
  --line: #4A3B2E;
  --cream: #F2E9DB;
  --dim: #B5A48E;
  --faint: #8A7A66;
  --gold: #C9A45C;
  --gold-ink: #241A10;
  --slate: #5F7A92;
  --rust: #B85C42;
}

* {
  box-sizing: border-box;
}

html,
body {
  height: 100%;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--cream);
  font-family:
    system-ui,
    -apple-system,
    "Segoe UI",
    Roboto,
    Helvetica,
    Arial,
    sans-serif;

  display: flex;
  justify-content: center;
  padding: 64px 20px;
}

.wrap {
  width: 100%;
  max-width: 540px;
}

h1 {
  font-size: 26px;
  font-weight: 650;
  letter-spacing: 0.01em;
  margin: 0 0 8px;
}

.sub {
  font-size: 14px;
  color: var(--dim);
  line-height: 1.5;
  margin: 0 0 32px;
  max-width: 48ch;
}

.field {
  margin-bottom: 16px;
}

label {
  display: block;
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--faint);
  margin-bottom: 6px;
}

input[type="text"] {
  width: 100%;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  color: var(--cream);
  font-family: inherit;
  font-size: 16px;
  padding: 13px 14px;
}

input[type="text"]::placeholder {
  color: var(--faint);
}

input[type="text"]:focus-visible {
  outline: 2px solid var(--gold);
  outline-offset: 1px;
  border-color: var(--gold);
}

#text {
  font-size: 18px;
}

.result {
  margin-top: 40px;
}

.pct {
  font-size: 56px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  line-height: 1;
  color: var(--cream);
  transition: color 0.15s ease;
}

.bar {
  position: relative;
  margin-top: 20px;
  height: 14px;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: visible;
}

.fill {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 0%;
  background: var(--slate);
  border-radius: 7px;
  transition:
    width 0.12s ease,
    background 0.15s ease;
}

.marker {
  position: absolute;
  top: -4px;
  bottom: -4px;
  width: 2px;
  background: var(--cream);
  opacity: 0.6;
  left: 0%;
  display: none;
}

.row {
  margin-top: 20px;
}

.chip {
  display: inline-block;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.08em;
  padding: 9px 18px;
  border-radius: 999px;
  border: 1px solid var(--line);
  color: var(--faint);
  background: transparent;

  transition:
    background 0.15s ease,
    color 0.15s ease,
    border-color 0.15s ease;
}

.captions {
  margin-top: 18px;
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 12.5px;
  color: var(--faint);
  font-variant-numeric: tabular-nums;
  flex-wrap: wrap;
}

</style>
</head>

<body>

<main class="wrap">

  <h1>End-of-Turn Live Probe</h1>

  <p class="sub">
    Type as the caller would speak it.
    P(turn complete) refreshes word by word
    against the live ONNX model.
  </p>

  <div class="field">

    <label for="ctx">
      Agent context
    </label>

    <input
      id="ctx"
      type="text"
      placeholder="Agent's last line (optional)"
      autocomplete="off"
    >

  </div>

  <div class="field">

    <label for="text">
      Caller
    </label>

    <input
      id="text"
      type="text"
      placeholder="Type what the caller says..."
      autocomplete="off"
      autofocus
    >

  </div>

  <section class="result">

    <div
      class="pct"
      id="pct"
    >
      ...
    </div>

    <div
      class="bar"
      id="bar"
    >

      <div
        class="fill"
        id="fill"
      ></div>

      <div
        class="marker"
        id="marker"
      ></div>

    </div>

    <div class="row">

      <span
        class="chip"
        id="chip"
      >
        IDLE
      </span>

    </div>

    <div class="captions">

      <span id="threshCaption">
        threshold pending
      </span>

      <span id="latencyCaption">
        model: pending
      </span>

    </div>

  </section>

</main>

<script>

(function () {

  var ctxInput =
    document.getElementById("ctx");

  var textInput =
    document.getElementById("text");

  var pctEl =
    document.getElementById("pct");

  var fillEl =
    document.getElementById("fill");

  var markerEl =
    document.getElementById("marker");

  var chipEl =
    document.getElementById("chip");

  var threshEl =
    document.getElementById("threshCaption");

  var latEl =
    document.getElementById("latencyCaption");

  var GOLD = "#C9A45C";
  var SLATE = "#5F7A92";
  var INK = "#241A10";
  var FAINT = "#8A7A66";
  var LINE = "#4A3B2E";
  var RUST = "#B85C42";
  var CREAM = "#F2E9DB";

  var debounceTimer = null;

  var requestSeq = 0;

  var knownThreshold = null;

  var lastScoredKey = null;

  var WORD_BOUNDARY =
    /[\\s.,!?;:]$/;


  function setThresholdCaption() {

    if (knownThreshold === null) {

      threshEl.textContent =
        "threshold pending";

      markerEl.style.display =
        "none";

      return;
    }

    threshEl.textContent =
      "threshold " +
      knownThreshold.toFixed(2);

    markerEl.style.left =
      (knownThreshold * 100).toFixed(2) +
      "%";

    markerEl.style.display =
      "block";
  }


  function renderIdle() {

    pctEl.textContent =
      "...";

    pctEl.style.color =
      CREAM;

    fillEl.style.width =
      "0%";

    fillEl.style.background =
      SLATE;

    chipEl.textContent =
      "IDLE";

    chipEl.style.background =
      "transparent";

    chipEl.style.color =
      FAINT;

    chipEl.style.borderColor =
      LINE;

    latEl.textContent =
      "model: pending";

    setThresholdCaption();
  }


  function renderLoading() {

    chipEl.textContent =
      "ANALYZING";

    chipEl.style.background =
      "transparent";

    chipEl.style.color =
      GOLD;

    chipEl.style.borderColor =
      GOLD;

    latEl.textContent =
      "model: running...";
  }


  function renderResult(data) {

    var pct =
      Math.max(
        0,
        Math.min(
          100,
          data.p_complete * 100
        )
      );

    var speak =
      data.decision === "speak";

    var accent =
      speak ? GOLD : SLATE;

    pctEl.textContent =
      pct.toFixed(1) + "%";

    pctEl.style.color =
      accent;

    fillEl.style.width =
      pct.toFixed(2) + "%";

    fillEl.style.background =
      accent;

    chipEl.textContent =
      speak
        ? "SPEAK"
        : "KEEP LISTENING";

    chipEl.style.background =
      accent;

    chipEl.style.color =
      INK;

    chipEl.style.borderColor =
      accent;

    latEl.textContent =
      "model: " +
      data.model_latency_ms.toFixed(1) +
      " ms";

    knownThreshold =
      data.threshold;

    setThresholdCaption();
  }


  function renderError() {

    chipEl.textContent =
      "ERROR";

    chipEl.style.background =
      "transparent";

    chipEl.style.color =
      RUST;

    chipEl.style.borderColor =
      RUST;

    pctEl.textContent =
      "...";

    pctEl.style.color =
      CREAM;

    fillEl.style.width =
      "0%";

    latEl.textContent =
      "model: unavailable";
  }


  function currentKey() {

    var text =
      textInput.value;

    if (text.trim() === "") {
      return null;
    }

    return (
      ctxInput.value.trim() +
      "\\u0000" +
      text.trim()
    );
  }


  async function runCheck() {

    var text =
      textInput.value;

    var key =
      currentKey();

    if (key === null) {

      lastScoredKey = null;

      renderIdle();

      return;
    }

    if (key === lastScoredKey) {
      return;
    }

    lastScoredKey = key;

    var seq =
      ++requestSeq;

    renderLoading();

    try {

      var response =
        await fetch(
          "/predict",
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json"
            },

            body: JSON.stringify({
              context:
                ctxInput.value,

              text:
                text
            })
          }
        );

      if (!response.ok) {
        throw new Error(
          "HTTP " +
          response.status
        );
      }

      var data =
        await response.json();

      if (
        seq === requestSeq &&
        key === currentKey()
      ) {

        renderResult(data);
      }

    } catch (error) {

      console.error(
        "Prediction failed:",
        error
      );

      if (
        seq === requestSeq &&
        key === currentKey()
      ) {

        lastScoredKey = null;

        renderError();
      }
    }
  }


  function scheduleCheck(event) {

    clearTimeout(
      debounceTimer
    );

    if (
      currentKey() === null ||
      (
        event &&
        event.data &&
        WORD_BOUNDARY.test(
          event.data
        )
      )
    ) {

      runCheck();

      return;
    }

    debounceTimer =
      setTimeout(
        runCheck,
        120
      );
  }


  ctxInput.addEventListener(
    "input",
    scheduleCheck
  );

  textInput.addEventListener(
    "input",
    scheduleCheck
  );

  renderIdle();

})();

</script>

</body>
</html>
"""


@app.get(
    "/",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def index() -> HTMLResponse:
    """Serve the standalone browser probe."""

    return HTMLResponse(
        content=INDEX_HTML
    )