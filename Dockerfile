FROM python:3.12-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY serve.py common.py ./
COPY data/gold_set.json data/gold_set.json
COPY models/eot-distilbert-onnx-int8 models/eot-distilbert-onnx-int8

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]