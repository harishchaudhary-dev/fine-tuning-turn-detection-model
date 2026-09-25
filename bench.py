"""Load-test client for the end-of-turn detector /predict endpoint."""

import argparse
import asyncio
import json
import time

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the end-of-turn detector /predict endpoint.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/predict")
    parser.add_argument("--n", type=int, default=2000, help="Total requests fired per concurrency level")
    parser.add_argument("--concurrency", type=str, default="1,8,32", help="Comma-separated concurrency levels")
    parser.add_argument("--data", type=str, default="data/calibration_samples.json")
    parser.add_argument("--report", type=str, default="bench_report.json")
    return parser.parse_args()


def load_payloads(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    return [{"context": s.get("context", ""), "text": s["text"]} for s in samples]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


async def fire_one(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    semaphore: asyncio.Semaphore,
    wall_times: list[float],
    model_times: list[float],
) -> None:
    async with semaphore:
        start = time.perf_counter()
        response = await client.post(url, json=payload)
        wall_times.append((time.perf_counter() - start) * 1000)
        response.raise_for_status()
        body = response.json()
        if "model_latency_ms" in body:
            model_times.append(body["model_latency_ms"])


async def run_level(url: str, payloads: list[dict], n: int, concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    wall_times: list[float] = []
    model_times: list[float] = []

    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        tasks = [fire_one(client, url, payloads[i % len(payloads)], semaphore, wall_times, model_times) for i in range(n)]
        await asyncio.gather(*tasks)
        total_seconds = time.perf_counter() - start

    return {
        "concurrency": concurrency,
        "n": n,
        "requests_per_sec": n / total_seconds if total_seconds > 0 else 0.0,
        "wall_p50_ms": percentile(wall_times, 50),
        "wall_p95_ms": percentile(wall_times, 95),
        "wall_p99_ms": percentile(wall_times, 99),
        "model_latency_p50_ms": percentile(model_times, 50),
        "model_latency_p95_ms": percentile(model_times, 95),
    }


def print_level_report(level: dict) -> None:
    print(
        f"concurrency={level['concurrency']} requests_per_sec={level['requests_per_sec']:.1f} "
        f"wall_p50={level['wall_p50_ms']:.2f}ms wall_p95={level['wall_p95_ms']:.2f}ms wall_p99={level['wall_p99_ms']:.2f}ms "
        f"model_p50={level['model_latency_p50_ms']:.2f}ms model_p95={level['model_latency_p95_ms']:.2f}ms"
    )


async def main_async() -> None:
    args = parse_args()
    payloads = load_payloads(args.data)
    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",") if c.strip()]

    results = []
    for concurrency in concurrency_levels:
        level = await run_level(args.url, payloads, args.n, concurrency)
        print_level_report(level)
        results.append(level)

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"url": args.url, "levels": results}, f, indent=2)
    print(f"report written to {args.report}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()