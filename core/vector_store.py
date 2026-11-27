"""Session-scoped temporary FAISS vector store with Gemini Embeddings and transparent fallback."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Tuple

import faiss
import numpy as np

from core.llm_client import GeminiClient
from core.schemas import DocumentChunk


class SessionVectorStore:
    """In-memory temporary FAISS index scoped to a single research project."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[DocumentChunk] = []
        self.embedding_mode: str = "uninitialized"  # "gemini" | "lexical_fallback"
        self.dimension: int = 0

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype("float32")

    @staticmethod
    def _lexical_embed(texts: list[str], dim: int = 384) -> np.ndarray:
        """Deterministic feature-hashing vector fallback when Gemini API key is not set."""
        mat = np.zeros((len(texts), dim), dtype="float32")
        for row_idx, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9]{2,}", text.lower())
            if not tokens:
                continue
            for tok in tokens:
                digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % dim
                sign = 1.0 if (digest[4] % 2 == 0) else -1.0
                mat[row_idx, bucket] += sign
            # Add character trigrams for partial fuzzy match
            for i in range(max(0, len(text) - 2)):
                tri = text[i : i + 3].lower()
                if tri.strip():
                    digest = hashlib.blake2b(tri.encode("utf-8"), digest_size=4).digest()
                    bucket = int.from_bytes(digest, "little") % dim
                    mat[row_idx, bucket] += 0.35 / math.sqrt(max(1, len(tokens)))
        return SessionVectorStore._normalize(mat)

    def build_from_chunks(
        self,
        chunks: list[DocumentChunk],
        llm_client: GeminiClient | None = None,
    ) -> Tuple[int, str]:
        """Build or rebuild the FAISS index from document chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.index = None
            self.embedding_mode = "empty"
            self.dimension = 0
            return 0, self.embedding_mode

        texts = [f"[{c.section} | p.{c.page_start}-{c.page_end}] {c.text}" for c in self.chunks]

        vectors_np: np.ndarray | None = None
        if llm_client and llm_client.is_configured:
            try:
                raw_vecs = llm_client.embed_texts(texts)
                if raw_vecs:
                    vectors_np = self._normalize(np.array(raw_vecs, dtype="float32"))
                    self.embedding_mode = "gemini"
            except Exception:
                vectors_np = None

        if vectors_np is None:
            vectors_np = self._lexical_embed(texts)
            self.embedding_mode = "lexical_fallback"

        self.dimension = int(vectors_np.shape[1])
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors_np)
        return len(self.chunks), self.embedding_mode

    def search(
        self,
        query: str,
        top_k: int = 5,
        llm_client: GeminiClient | None = None,
        paper_id_filter: str | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Semantic similarity search returning chunks with cosine scores."""
        if not self.index or not self.chunks or not query.strip():
            return []

        q_vec: np.ndarray | None = None
        if self.embedding_mode == "gemini" and llm_client and llm_client.is_configured:
            try:
                raw = llm_client.embed_texts([query])
                if raw and len(raw[0]) == self.dimension:
                    q_vec = self._normalize(np.array(raw, dtype="float32"))
            except Exception:
                q_vec = None

        if q_vec is None:
            if self.embedding_mode == "gemini":
                # Query couldn't use Gemini; fallback keyword overlap scoring over stored chunks
                return self._keyword_score(query, top_k=top_k, paper_id_filter=paper_id_filter)
            q_vec = self._lexical_embed([query], dim=self.dimension)

        search_k = min(len(self.chunks), top_k * 3 if paper_id_filter else top_k)
        scores, indices = self.index.search(q_vec, search_k)
        results: list[tuple[DocumentChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if paper_id_filter and chunk.paper_id != paper_id_filter:
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break
        return results

    def _keyword_score(
        self, query: str, top_k: int = 5, paper_id_filter: str | None = None
    ) -> list[tuple[DocumentChunk, float]]:
        q_tokens = set(re.findall(r"[a-z0-9]{2,}", query.lower()))
        if not q_tokens:
            return []
        scored: list[tuple[DocumentChunk, float]] = []
        for chunk in self.chunks:
            if paper_id_filter and chunk.paper_id != paper_id_filter:
                continue
            c_tokens = set(re.findall(r"[a-z0-9]{2,}", chunk.text.lower()))
            overlap = len(q_tokens & c_tokens)
            if overlap > 0:
                score = overlap / math.sqrt(len(q_tokens) * max(1, len(c_tokens)))
                scored.append((chunk, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
