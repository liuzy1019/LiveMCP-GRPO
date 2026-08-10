"""LLM inference client for teacher-guided data generation.

Supports three backends:
- local:  transformers pipeline on a single device (default: cuda:0)
- openai: OpenAI-compatible API (vLLM server or external)

Multi-GPU modes:
1. Tensor Parallel (TP) — vLLM server with --tensor-parallel-size N
   → use mode="openai", api_base="http://localhost:8000/v1"
2. Device assignment — pin local mode to a specific GPU
   → use LLMClient(mode="local", device=2)
"""

from __future__ import annotations

import os
import threading
import re
from typing import Any

from loguru import logger

from src.utils import extract_json


_CONTEXT_LENGTH_ERROR_RE = re.compile(
    r"maximum context length is\s+(?P<context>\d+)\s+tokens.*?"
    r"requested\s+(?P<output>\d+)\s+output tokens.*?"
    r"prompt contains at least\s+(?P<input>\d+)\s+input tokens",
    re.IGNORECASE | re.DOTALL,
)
_MIN_CONTEXT_RETRY_TOKENS = 64


def _remaining_context_output_budget(
    error: BaseException,
    requested_max_tokens: int,
) -> int | None:
    """Return a safe one-shot decode budget for an explicit context error.

    vLLM reports the configured context window and tokenized input size in its
    400 response.  Preserve the complete Teacher input and shrink only the
    requested JSON decode budget.  Very small residual windows remain a hard
    failure because a truncated action envelope is not useful training data.
    """
    match = _CONTEXT_LENGTH_ERROR_RE.search(str(error))
    if match is None:
        return None
    context_tokens = int(match.group("context"))
    input_tokens = int(match.group("input"))
    remaining = context_tokens - input_tokens
    if not (_MIN_CONTEXT_RETRY_TOKENS <= remaining < requested_max_tokens):
        return None
    return remaining

# Lazy imports to avoid hard dependency on model packages
_HAS_TRANSFORMERS = False
try:
    from transformers import pipeline  # noqa: F401
    _HAS_TRANSFORMERS = True
except (ImportError, OSError):
    pass


class LLMClient:
    """Lightweight LLM inference wrapper.

    Usage:
        # Single-GPU local
        client = LLMClient(
            mode="local", model_path="models/Google/Gemma-4-31B-it", device=0,
        )

        # Multi-GPU with device_map="auto" (model parallelism for the Teacher)
        client = LLMClient(
            mode="local", model_path="models/Google/Gemma-4-31B-it",
        )

        # vLLM / OpenAI-compatible server
        client = LLMClient(mode="openai", model_path="Qwen3-4B-Instruct-2507",
                          api_base="http://localhost:8000/v1")
    """

    def __init__(
        self,
        mode: str = "local",
        model_path: str = "models/Google/Gemma-4-31B-it",
        contract_model_id: str | None = None,
        api_base: str | None = None,
        api_key: str = "not-needed",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout_s: float | None = None,
        device: int | str | None = None,
    ):
        self.mode = mode
        self.model_path = model_path
        # API serving aliases are transport details, not model provenance.
        # Keep a stable identity for cache contracts even when vLLM is called
        # through a shorter --served-model-name.
        self.contract_model_id = contract_model_id or model_path
        self.api_base = api_base or os.environ.get("LLM_API_BASE", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "not-needed")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = (
            float(timeout_s)
            if timeout_s is not None
            else float(os.environ.get("LLM_API_TIMEOUT_S", "180"))
        )
        self._pipe = None
        self._tokenizer = None
        self._client = None
        self._local_lock = threading.RLock()

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
            with self._local_lock:
                if self._pipe is not None:
                    return
                self._load_local_pipeline()
        elif self.mode == "openai" and self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout_s,
            )

    def _load_local_pipeline(self):
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
        json_mode: bool = False,
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
            with self._local_lock:
                return self._generate_local(prompt, temp, mt)

        # OpenAI mode: pass messages directly.
        # For Qwen3 models, request non-thinking mode.  Some serving stacks may
        # still emit <think> blocks, so callers also strip them before parsing.
        # The OpenAI SDK only accepts vLLM-specific request fields through
        # extra_body; the SDK merges this dict into the JSON request body.
        # Do not send a literal {"extra_body": ...} in raw HTTP requests.
        create_kwargs: dict = {}
        if json_mode:
            # vLLM's OpenAI-compatible server supports the standard JSON
            # response format. Semantics remain prompt-driven; this only keeps
            # the Teacher action envelope machine-readable.
            create_kwargs["response_format"] = {"type": "json_object"}
        if "qwen" in self.model_path.lower() or "Qwen" in self.model_path:
            create_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        request_kwargs = {
            "model": self.model_path,
            "messages": messages,
            "temperature": temp,
            "max_tokens": mt,
            **create_kwargs,
        }
        try:
            response = self._client.chat.completions.create(**request_kwargs)
        except Exception as error:
            adjusted_max_tokens = _remaining_context_output_budget(error, mt)
            if adjusted_max_tokens is None:
                raise
            logger.warning(
                "Teacher request reached the model context boundary; "
                "preserving the full input and retrying once with max_tokens={} "
                "instead of {}",
                adjusted_max_tokens,
                mt,
            )
            request_kwargs["max_tokens"] = adjusted_max_tokens
            response = self._client.chat.completions.create(**request_kwargs)
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
            json_mode=True,
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
