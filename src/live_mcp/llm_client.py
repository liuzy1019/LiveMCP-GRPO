"""LLM inference client for teacher-guided data generation.

Supports three backends:
- local:  transformers pipeline on a single device (default: cuda:0)
- openai: OpenAI-compatible API (vLLM server or external)
- local_pool: multi-process data-parallel across GPUs (for batch generation)

Multi-GPU modes:
1. Tensor Parallel (TP) — vLLM server with --tensor-parallel-size N
   → use mode="openai", api_base="http://localhost:8000/v1"
2. Data Parallel (DP) — N independent model copies on N GPUs
   → use LLMClientPool(mode="local_dp", gpu_ids=[0,1,2,3])
3. Device assignment — pin local mode to a specific GPU
   → use LLMClient(mode="local", device=2)
"""

from __future__ import annotations

import json
import os
import re
import multiprocessing as mp
from typing import Any

from loguru import logger

from src.utils import extract_json

# Lazy imports to avoid hard dependency on model packages
_HAS_TRANSFORMERS = False
_HAS_TORCH = False
try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    pass

try:
    from transformers import pipeline  # noqa: F401
    _HAS_TRANSFORMERS = True
except (ImportError, OSError):
    pass


class LLMClient:
    """Lightweight LLM inference wrapper.

    Usage:
        # Single-GPU local
        client = LLMClient(mode="local", model_path="models/Qwen3-4B", device=0)

        # Multi-GPU with device_map="auto" (model parallelism for large models)
        client = LLMClient(mode="local", model_path="models/Qwen3-32B")

        # vLLM / OpenAI-compatible server
        client = LLMClient(mode="openai", model_path="Qwen3-4B",
                          api_base="http://localhost:8000/v1")
    """

    def __init__(
        self,
        mode: str = "local",
        model_path: str = "models/Qwen3-4B",
        api_base: str | None = None,
        api_key: str = "not-needed",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        device: int | str | None = None,
    ):
        self.mode = mode
        self.model_path = model_path
        self.api_base = api_base or os.environ.get("LLM_API_BASE", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "not-needed")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._pipe = None
        self._tokenizer = None
        self._client = None

        # Resolve device_map for local mode
        if device is not None:
            # Pin to specific GPU: e.g. device=2 → {"": "cuda:2"}
            if isinstance(device, int):
                self._device_map = {"": f"cuda:{device}"}
            elif isinstance(device, str) and device == "auto":
                self._device_map = "auto"
            else:
                self._device_map = {"": str(device)}
        else:
            self._device_map = "auto"

    def _ensure_pipe(self):
        if self.mode == "local" and self._pipe is None:
            if not _HAS_TRANSFORMERS:
                raise ImportError(
                    "transformers not installed. Use mode='openai' "
                    "or pip install transformers torch"
                )
            logger.info(f"Loading local model: {self.model_path} (device_map={self._device_map})")
            self._pipe = pipeline(
                "text-generation",
                model=self.model_path,
                trust_remote_code=True,
                device_map=self._device_map,
                torch_dtype="auto",
            )
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True,
            )
            logger.info("Model loaded")
        elif self.mode == "openai" and self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text from prompt (delegates to generate_chat for chat-template-aware generation)."""
        return self.generate_chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text from chat messages (applies chat template for local models)."""
        self._ensure_pipe()
        temp = temperature if temperature is not None else self.temperature
        mt = max_tokens if max_tokens is not None else self.max_tokens

        if self.mode == "local":
            if hasattr(self, '_tokenizer') and self._tokenizer.chat_template:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                prompt = "\n".join(m["content"] for m in messages)
            return self._generate_local(prompt, temp, mt)

        # OpenAI mode: pass messages directly
        extra_body = {}
        if "qwen" in self.model_path.lower() or "Qwen" in self.model_path:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        response = self._client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            temperature=temp,
            max_tokens=mt,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        return response.choices[0].message.content or ""

    def generate_json(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate and parse JSON response."""
        raw = self.generate_chat(
            [{"role": "user", "content": prompt}],
            temperature,
        )
        return extract_json(raw)

    def _generate_local(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Low-level local generation via transformers pipeline."""
        result = self._pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
            return_full_text=False,
        )
        from src.utils import strip_think_tags
        text = strip_think_tags(result[0]["generated_text"])
        return text


class LLMClientPool:
    """Data-parallel inference pool across multiple GPUs.

    Each GPU runs an independent model copy. Tasks are distributed
    via multiprocessing.Queue. Use when you have many tasks to generate
    and want to saturate all GPUs.

    Usage:
        pool = LLMClientPool(
            model_path="models/Qwen3-8B",
            gpu_ids=[0, 1, 2, 3],   # or "auto" to use CUDA_VISIBLE_DEVICES
        )
        results = pool.generate_batch(prompts=["prompt1", "prompt2", ...])
        pool.shutdown()
    """

    def __init__(
        self,
        model_path: str,
        gpu_ids: list[int] | str = "auto",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        if gpu_ids == "auto":
            n_gpus = int(os.environ.get("GPU_COUNT", torch.cuda.device_count()))
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible:
                gpu_ids = [int(x) for x in visible.split(",")]
            else:
                gpu_ids = list(range(n_gpus))

        if not gpu_ids:
            raise ValueError("No GPUs available for LLMClientPool")

        self.gpu_ids = gpu_ids
        logger.info(f"LLMClientPool: {len(gpu_ids)} GPUs → {gpu_ids}")

    def generate_batch(
        self,
        prompts: list[str],
        batch_size: int | None = None,
    ) -> list[str]:
        """Generate responses for a list of prompts using all GPUs in parallel.

        Args:
            prompts: List of prompt strings
            batch_size: Chunks per GPU (auto-computed if None)

        Returns:
            List of generated responses in the same order as prompts
        """
        if not prompts:
            return []

        n_gpus = len(self.gpu_ids)
        if batch_size is None:
            # Split evenly across GPUs
            batch_size = max(1, len(prompts) // n_gpus)

        # Split prompts into chunks for each GPU
        chunks = [prompts[i::n_gpus] for i in range(n_gpus)]

        # Launch one worker per GPU
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        result_queue = manager.Queue()

        workers = []
        for gpu_idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            p = ctx.Process(
                target=_gpu_worker,
                args=(
                    gpu_idx, self.gpu_ids[gpu_idx], self.model_path,
                    chunk, self.temperature, self.max_tokens, result_queue,
                    n_gpus,
                ),
            )
            p.start()
            workers.append(p)

        # Collect results
        results: dict[int, str] = {}
        for _ in range(len(prompts)):
            idx, text = result_queue.get()
            results[idx] = text

        for p in workers:
            p.join()

        # Reconstruct original order
        return [results[i] for i in range(len(prompts))]

    def shutdown(self):
        """No-op for now; workers are cleaned up after each generate_batch call."""
        pass


def _gpu_worker(
    gpu_idx: int,
    device_id: int,
    model_path: str,
    prompts: list[str],
    temperature: float,
    max_tokens: int,
    result_queue: mp.Queue,
    n_gpus: int,
) -> None:
    """Worker function that loads model on one GPU and processes prompts."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # Re-import after CUDA_VISIBLE_DEVICES is set
    import torch
    from transformers import pipeline, AutoTokenizer

    logger.info(f"[GPU {device_id}] Loading model: {model_path}")
    pipe = pipeline(
        "text-generation",
        model=model_path,
        trust_remote_code=True,
        device_map={"": "cuda:0"},  # CUDA_VISIBLE_DEVICES remaps → always cuda:0 in child
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    for i, prompt in enumerate(prompts):
        # prompts are interleaved: chunk[gpu_idx][i] = all_prompts[gpu_idx + i * n_gpus]
        global_idx = gpu_idx + i * n_gpus
        try:
            if tokenizer.chat_template:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                formatted = prompt

            result = pipe(
                formatted,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                return_full_text=False,
            )
            text = result[0]["generated_text"]
            logger.debug(f"[GPU {device_id}] {i}/{len(prompts)} done")
        except Exception as e:
            logger.error(f"[GPU {device_id}] Error on prompt {i}: {e}")
            text = ""

        result_queue.put((global_idx, text))

    logger.info(f"[GPU {device_id}] Done ({len(prompts)} prompts)")
