"""Google Gemini & LangChain Gemini client wrapper with retry, rate-limit backoff, and Pydantic validation."""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Callable, Type, TypeVar

from pydantic import BaseModel

from core.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, get_gemini_api_key

logger = logging.getLogger("researchiq.llm")
T = TypeVar("T", bound=BaseModel)

MAX_PROMPT_CHARS = 30000
DEFAULT_MAX_RETRIES = 3


def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in (
            "429",
            "rate limit",
            "resource_exhausted",
            "quota",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "503",
            "502",
            "500",
            "connection",
        )
    )


def _retry_with_backoff(fn: Callable[[], Any], max_retries: int = DEFAULT_MAX_RETRIES) -> Any:
    """Execute an API call with exponential backoff and jitter on transient/rate-limit errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries - 1 or not _is_retryable_error(exc):
                raise
            delay = (1.5 * (2**attempt)) + random.uniform(0.0, 0.5)
            logger.warning("Gemini API transient error (attempt %d/%d); retrying in %.2fs", attempt + 1, max_retries, delay)
            time.sleep(delay)
    if last_exc:
        raise last_exc


class GeminiClient:
    """Unified wrapper over google-genai and langchain-google-genai. Never exposes API keys."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self._api_key = get_gemini_api_key(api_key)
        self.model_name = model_name or DEFAULT_LLM_MODEL
        self.last_error: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key and self._api_key != "your_gemini_api_key_here")

    def __repr__(self) -> str:
        return f"GeminiClient(model_name={self.model_name!r}, configured={self.is_configured})"

    @staticmethod
    def _truncate_text(text: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n\n[... middle truncated to stay within context budget ...]\n\n" + text[-half:]

    def get_langchain_llm(self, temperature: float = 0.2):
        """Return a ChatGoogleGenerativeAI instance."""
        if not self.is_configured:
            raise RuntimeError("Gemini API key is not configured.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self._api_key,
            temperature=temperature,
        )

    def get_genai_client(self):
        """Return a native google-genai Client."""
        if not self.is_configured:
            raise RuntimeError("Gemini API key is not configured.")
        from google import genai

        return genai.Client(api_key=self._api_key)

    def get_langchain_embeddings(self):
        """Return Gemini Embeddings via LangChain."""
        if not self.is_configured:
            raise RuntimeError("Gemini API key is not configured.")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=f"models/{DEFAULT_EMBEDDING_MODEL}"
            if not DEFAULT_EMBEDDING_MODEL.startswith("models/")
            else DEFAULT_EMBEDDING_MODEL,
            google_api_key=self._api_key,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Gemini Embeddings API with batching and backoff."""
        if not texts:
            return []
        if not self.is_configured:
            raise RuntimeError("Gemini API key is not configured.")
        client = self.get_genai_client()
        vectors: list[list[float]] = []
        batch_size = 32
        self.last_error = ""
        try:
            for i in range(0, len(texts), batch_size):
                batch = [t[:8000] if t and t.strip() else " " for t in texts[i : i + batch_size]]
                resp = _retry_with_backoff(
                    lambda b=batch: client.models.embed_content(
                        model=DEFAULT_EMBEDDING_MODEL,
                        contents=b,
                    )
                )
                for emb in resp.embeddings:
                    vectors.append(list(emb.values))
            return vectors
        except Exception as exc:
            self.last_error = f"Embedding request failed ({type(exc).__name__})"
            logger.warning("Gemini embedding failure: %s", type(exc).__name__)
            raise

    def generate_text(self, prompt: str, system_instruction: str = "", temperature: float = 0.3) -> str:
        """Generate plain text or markdown using LangChain Gemini integration with retry."""
        if not self.is_configured:
            raise RuntimeError("Gemini API key is not configured.")
        from langchain_core.messages import HumanMessage, SystemMessage

        safe_prompt = self._truncate_text(prompt)
        llm = self.get_langchain_llm(temperature=temperature)
        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=safe_prompt))

        self.last_error = ""
        try:
            response = _retry_with_backoff(lambda: llm.invoke(messages))
            content = response.content
            if isinstance(content, list):
                return "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
            return str(content)
        except Exception as exc:
            self.last_error = f"LLM text generation failed ({type(exc).__name__})"
            logger.warning("Gemini text generation failure: %s", type(exc).__name__)
            raise

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_instruction: str = "",
        temperature: float = 0.15,
    ) -> T:
        """Generate a validated Pydantic object via Gemini structured output with fallback and retry."""
        if not self.is_configured:
            raise RuntimeError("Gemini API key is not configured.")
        safe_prompt = self._truncate_text(prompt)
        self.last_error = ""

        # Path 1: LangChain structured output
        try:
            llm = self.get_langchain_llm(temperature=temperature)
            structured_llm = llm.with_structured_output(schema)
            full_prompt = f"{system_instruction}\n\n{safe_prompt}" if system_instruction else safe_prompt
            result = _retry_with_backoff(lambda: structured_llm.invoke(full_prompt))
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
        except Exception as lc_exc:
            logger.debug("LangChain structured output fallback triggered: %s", type(lc_exc).__name__)

        # Path 2: Native google-genai JSON schema response
        try:
            client = self.get_genai_client()
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
                system_instruction=system_instruction or None,
            )
            resp = _retry_with_backoff(
                lambda: client.models.generate_content(
                    model=self.model_name,
                    contents=safe_prompt,
                    config=config,
                )
            )
            raw_text = (resp.text or "{}").strip()
            # Strip accidental markdown code fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                if raw_text.lower().startswith("json"):
                    raw_text = raw_text[4:].strip()
            data: Any = json.loads(raw_text)
            return schema.model_validate(data)
        except Exception as exc:
            self.last_error = f"Structured generation failed for {schema.__name__} ({type(exc).__name__})"
            logger.warning("Structured output failure for %s: %s", schema.__name__, type(exc).__name__)
            raise RuntimeError(self.last_error) from exc


def get_llm_client(session_api_key: str | None = None) -> GeminiClient:
    """Factory for GeminiClient respecting session override without mutating process env."""
    return GeminiClient(api_key=session_api_key)
