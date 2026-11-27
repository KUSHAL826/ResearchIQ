"""Google Gemini & LangChain Gemini client wrapper with structured Pydantic output."""

from __future__ import annotations

import json
import os
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from core.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, get_gemini_api_key

T = TypeVar("T", bound=BaseModel)


class GeminiClient:
    """Unified wrapper over google-genai and langchain-google-genai."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self.api_key = get_gemini_api_key(api_key)
        self.model_name = model_name or DEFAULT_LLM_MODEL

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_gemini_api_key_here")

    def get_langchain_llm(self, temperature: float = 0.2):
        """Return a ChatGoogleGenerativeAI instance."""
        if not self.is_configured:
            raise RuntimeError("Gemini API key is not configured.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=temperature,
        )

    def get_genai_client(self):
        """Return a native google-genai Client."""
        if not self.is_configured:
            raise RuntimeError("Gemini API key is not configured.")
        from google import genai

        return genai.Client(api_key=self.api_key)

    def get_langchain_embeddings(self):
        """Return Gemini Embeddings via LangChain."""
        if not self.is_configured:
            raise RuntimeError("Gemini API key is not configured.")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=f"models/{DEFAULT_EMBEDDING_MODEL}"
            if not DEFAULT_EMBEDDING_MODEL.startswith("models/")
            else DEFAULT_EMBEDDING_MODEL,
            google_api_key=self.api_key,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Gemini Embeddings API."""
        if not texts:
            return []
        if not self.is_configured:
            raise RuntimeError("Gemini API key is not configured.")
        client = self.get_genai_client()
        vectors: list[list[float]] = []
        # Batch in groups of 32 to stay well within payload limits
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = [t[:8000] if t.strip() else " " for t in texts[i : i + batch_size]]
            resp = client.models.embed_content(
                model=DEFAULT_EMBEDDING_MODEL,
                contents=batch,
            )
            for emb in resp.embeddings:
                vectors.append(list(emb.values))
        return vectors

    def generate_text(self, prompt: str, system_instruction: str = "", temperature: float = 0.3) -> str:
        """Generate plain text or markdown using LangChain Gemini integration."""
        if not self.is_configured:
            raise RuntimeError("Gemini API key is not configured.")
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = self.get_langchain_llm(temperature=temperature)
        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=prompt))
        response = llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            return "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        return str(content)

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_instruction: str = "",
        temperature: float = 0.15,
    ) -> T:
        """Generate a validated Pydantic object via Gemini structured output."""
        if not self.is_configured:
            raise RuntimeError("Gemini API key is not configured.")
        try:
            llm = self.get_langchain_llm(temperature=temperature)
            structured_llm = llm.with_structured_output(schema)
            full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
            result = structured_llm.invoke(full_prompt)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
        except Exception:
            # Fallback to google-genai native JSON schema output
            client = self.get_genai_client()
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
                system_instruction=system_instruction or None,
            )
            resp = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            raw_text = (resp.text or "{}").strip()
            data: Any = json.loads(raw_text)
            return schema.model_validate(data)

        raise RuntimeError(f"Failed to produce structured output for {schema.__name__}")


def get_llm_client(session_api_key: str | None = None) -> GeminiClient:
    """Factory for GeminiClient respecting session override and environment."""
    if session_api_key:
        os.environ.setdefault("GOOGLE_API_KEY", session_api_key)
    return GeminiClient(api_key=session_api_key)
