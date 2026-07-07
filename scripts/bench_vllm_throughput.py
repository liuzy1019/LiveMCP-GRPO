#!/usr/bin/env python3
"""Small OpenAI-compatible vLLM throughput benchmark."""

from __future__ import annotations

import argparse
import concurrent.futures
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class Result:
    ok: bool
    elapsed_s: float
    prompt_tokens: int
    completion_tokens: int
    error: str = ""


def _request(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> Result:
    client = OpenAI(base_url=endpoint, api_key="EMPTY", timeout=timeout_s)
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "ignore_eos": True,
                "min_tokens": max_tokens,
            },
        )
        elapsed = time.perf_counter() - start
        usage = response.usage
        return Result(
            ok=True,
            elapsed_s=elapsed,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
    except Exception as exc:  # pragma: no cover - diagnostic script
        return Result(
            ok=False,
            elapsed_s=time.perf_counter() - start,
            prompt_tokens=0,
            completion_tokens=0,
            error=repr(exc),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--prompt",
        default="Return a compact JSON object with fields action, thought, and answer. "
        "Make the answer concise but non-empty.",
    )
    args = parser.parse_args()

    endpoints = [e.rstrip("/") for e in args.endpoint]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                _request,
                endpoints[i % len(endpoints)],
                args.model,
                args.prompt,
                args.max_tokens,
                args.temperature,
                args.timeout_s,
            )
            for i in range(args.concurrency)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    wall = time.perf_counter() - started

    failures = [r for r in results if not r.ok]
    prompt_tokens = sum(r.prompt_tokens for r in results)
    completion_tokens = sum(r.completion_tokens for r in results)
    total_tokens = prompt_tokens + completion_tokens
    avg_latency = sum(r.elapsed_s for r in results) / max(len(results), 1)
    print(f"requests={len(results)} failures={len(failures)} endpoints={len(endpoints)}")
    print(f"wall_s={wall:.2f} avg_latency_s={avg_latency:.2f}")
    print(f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} total_tokens={total_tokens}")
    print(f"completion_tok_s={completion_tokens / wall:.2f}")
    print(f"total_tok_s={total_tokens / wall:.2f}")
    if failures:
        print(f"first_error={failures[0].error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
