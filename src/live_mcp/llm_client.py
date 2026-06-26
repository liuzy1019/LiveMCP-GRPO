"""LLM inference client for teacher-guided data generation.

Supports two backends:
- local: transformers pipeline (offline, no server needed)
- openai: OpenAI-compatible API (vLLM server or external)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from loguru import logger

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
except ImportError:
    pass


class LLMClient:
    """Lightweight LLM inference wrapper.

    Usage:
        client = LLMClient(mode="local", model_path="models/Qwen3-4B")
        # or
        client = LLMClient(mode="openai", model_path="Qwen3-4B",
                          api_base="http://localhost:8000/v1")
        response = client.generate(prompt)
    """

    def __init__(
        self,
        mode: str = "local",
        model_path: str = "models/Qwen3-4B",
        api_base: str | None = None,
        api_key: str = "not-needed",
        temperature: float = 0.7,
        max_tokens: int = 1024,
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

    def _ensure_pipe(self):
        if self.mode == "local" and self._pipe is None:
            if not _HAS_TRANSFORMERS:
                raise ImportError(
                    "transformers not installed. Use mode='openai' "
                    "or pip install transformers torch"
                )
            logger.info(f"Loading local model: {self.model_path}")
            self._pipe = pipeline(
                "text-generation",
                model=self.model_path,
                trust_remote_code=True,
                device_map="auto",
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
        response = self._client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            temperature=temp,
            max_tokens=mt,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
        return _extract_json(raw)

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
        text = result[0]["generated_text"]
        # Strip Qwen3 <think>...</think> blocks (defense-in-depth)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "<think>" in text:
            after_think = re.sub(r"<think>[\s\S]+", "", text)
            text = after_think.strip() if after_think.strip() else ""
        text = text.strip()
        return text

    def _generate_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Low-level OpenAI API generation (kept for backward compat)."""
        response = self._client.chat.completions.create(
            model=self.model_path,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from text, handling markdown fences and think tags."""
    # Strip Qwen3 <think>...</think> blocks
    # 1. Remove closed <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 2. If <think> is unclosed, strip from <think> to end
    if "<think>" in text:
        after_think = re.sub(r"<think>[\s\S]+", "", text)
        text = after_think.strip() if after_think.strip() else ""
    text = text.strip()

    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON object boundaries (greedy: last { to first })
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from: {text[:200]}...")
