from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import OPENAI_RESPONSE_FORMAT


@dataclass
class LLMResponse:
    text: str
    backend: str
    model: str
    cached: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None


class JsonlCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict[str, Any]] = {}

        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = item.get("key")
                    if key:
                        self._index[str(key)] = item

    @staticmethod
    def make_key(payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        return self._index.get(key)

    def put(
        self,
        key: str,
        payload: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        item = {
            "key": key,
            "payload": payload,
            "response": response,
            "created_at": time.time(),
        }
        self._index[key] = item
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


class LLMClient:
    """Small OpenAI/Anthropic client with a local JSONL cache."""

    def __init__(
        self,
        backend: str,
        model: str,
        cache_path: str | Path = "outputs/llm_cache.jsonl",
        timeout: int = 120,
    ) -> None:
        self.backend = backend.lower()
        self.model = model
        self.timeout = timeout
        self.cache = JsonlCache(cache_path)

        if self.backend == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            from openai import OpenAI

            self.openai_client = OpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=5,
            )
        elif self.backend == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1200,
        temperature: float = 0.55,
        bypass_cache: bool = False,
    ) -> LLMResponse:
        payload = {
            "backend": self.backend,
            "model": self.model,
            "system": system_prompt,
            "user": user_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "schema_version": "six_turn_v1",
        }
        key = self.cache.make_key(payload)

        if not bypass_cache:
            cached = self.cache.get(key)
            if cached:
                response = cached.get("response", {})
                return LLMResponse(
                    text=str(response.get("text", "")),
                    backend=self.backend,
                    model=self.model,
                    cached=True,
                    input_tokens=response.get("input_tokens"),
                    output_tokens=response.get("output_tokens"),
                )

        if self.backend == "openai":
            response = self._generate_openai(
                system_prompt, user_prompt, max_tokens, temperature
            )
        else:
            response = self._generate_anthropic(
                system_prompt, user_prompt, max_tokens, temperature
            )

        self.cache.put(key, payload, response)
        return LLMResponse(
            text=str(response.get("text", "")),
            backend=self.backend,
            model=self.model,
            cached=False,
            input_tokens=response.get("input_tokens"),
            output_tokens=response.get("output_tokens"),
        )

    def _generate_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        current_limit = max(max_tokens, 1200)

        for attempt in range(2):
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=current_limit,
                temperature=temperature,
                response_format=OPENAI_RESPONSE_FORMAT,
            )
            if not response.choices:
                raise RuntimeError("OpenAI returned no completion choices.")

            choice = response.choices[0]
            if choice.finish_reason != "length":
                refusal = getattr(choice.message, "refusal", None)
                if refusal:
                    raise RuntimeError(f"OpenAI refused the request: {refusal}")
                text = (choice.message.content or "").strip()
                if not text:
                    raise RuntimeError("OpenAI returned an empty response.")
                usage = response.usage
                return {
                    "text": text,
                    "input_tokens": usage.prompt_tokens if usage else None,
                    "output_tokens": usage.completion_tokens if usage else None,
                }

            if attempt == 0:
                current_limit = min(current_limit * 2, 2400)

        raise RuntimeError("OpenAI output remained truncated after retrying.")

    def _generate_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        api_key = os.environ["ANTHROPIC_API_KEY"]
        current_limit = max(max_tokens, 1200)

        for attempt in range(2):
            body = {
                "model": self.model,
                "max_tokens": current_limit,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            request = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "content-type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout
                ) as response:
                    raw = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(
                    f"Anthropic request failed: {exc.code} {detail}"
                ) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Anthropic connection failed: {exc}") from exc

            if raw.get("stop_reason") == "max_tokens" and attempt == 0:
                current_limit = min(current_limit * 2, 2400)
                continue

            text = "\n".join(
                str(item.get("text", ""))
                for item in raw.get("content", [])
                if item.get("type") == "text"
            ).strip()
            if not text:
                raise RuntimeError("Anthropic returned an empty response.")
            usage = raw.get("usage") or {}
            return {
                "text": text,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            }

        raise RuntimeError("Anthropic output remained truncated after retrying.")

