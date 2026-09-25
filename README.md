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
