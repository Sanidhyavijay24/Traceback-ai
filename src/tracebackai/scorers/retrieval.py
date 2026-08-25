"""
Traceback AI - Retrieval Step Scorer.

Evaluates relevance between retrieval query input and returned document chunks
using sentence-transformers embeddings with a pure-Python BM25 term-overlap fallback.
"""

import logging
import math
import re
from typing import Any, Optional

from tracebackai.models import Step
from tracebackai.scorers.base import BaseScorer

logger = logging.getLogger(__name__)

WEAK_RETRIEVAL_THRESHOLD = 0.55

_MODEL_INSTANCE: Any = None
_MODEL_LOAD_FAILED = False
_WARNED_FALLBACK = False

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or", "is",
    "are", "was", "were", "be", "been", "being", "with", "as", "by", "what",
    "which", "who", "whom", "this", "that", "these", "those", "how", "why",
}


def _get_sentence_transformer() -> Any:
    """Lazy-load sentence-transformers model instance."""
    global _MODEL_INSTANCE, _MODEL_LOAD_FAILED, _WARNED_FALLBACK
    if _MODEL_LOAD_FAILED:
        return None
    if _MODEL_INSTANCE is None:
        try:
            from sentence_transformers import SentenceTransformer

            _MODEL_INSTANCE = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _MODEL_LOAD_FAILED = True
            if not _WARNED_FALLBACK:
                logger.warning(
                    "sentence-transformers not available or failed to load. "
                    "Falling back to BM25 term overlap. Install with: pip install traceback-ai[semantic]"
                )
                _WARNED_FALLBACK = True
            return None
    return _MODEL_INSTANCE


def _extract_query(input_data: Any) -> Optional[str]:
    """Extract search query string from input payload."""
    if input_data is None:
        return None
    if isinstance(input_data, str):
        return input_data.strip() or None
    if isinstance(input_data, dict):
        for key in ("query", "q", "question", "prompt", "text", "search_term"):
            val = input_data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        kwargs = input_data.get("kwargs", {})
        if isinstance(kwargs, dict):
            for key in ("query", "q", "question", "prompt", "text"):
                val = kwargs.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        args = input_data.get("args", [])
        if isinstance(args, list) and args:
            for item in args:
                if isinstance(item, str) and item.strip():
                    return item.strip()
        for val in input_data.values():
            if isinstance(val, str) and val.strip():
                return val.strip()
    if isinstance(input_data, (list, tuple)):
        for item in input_data:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return str(input_data).strip() or None


def _extract_chunks(output_data: Any) -> list[str]:
    """Extract document chunk strings from output payload."""
    if output_data is None:
        return []
    if isinstance(output_data, str):
        return [output_data] if output_data.strip() else []
    if isinstance(output_data, (list, tuple, set)):
        chunks: list[str] = []
        for item in output_data:
            if isinstance(item, str):
                if item.strip():
                    chunks.append(item.strip())
            elif isinstance(item, dict):
                for key in ("text", "content", "chunk", "page_content", "document", "snippet"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip():
                        chunks.append(val.strip())
                        break
                else:
                    chunks.append(str(item))
            else:
                chunks.append(str(item))
        return chunks
    if isinstance(output_data, dict):
        for key in ("chunks", "documents", "results", "data", "matches"):
            val = output_data.get(key)
            if isinstance(val, list):
                return _extract_chunks(val)
        for key in ("text", "content", "snippet"):
            val = output_data.get(key)
            if isinstance(val, str) and val.strip():
                return [val.strip()]
        return [str(output_data)]
    return [str(output_data)]


def _tokenize(text: str) -> list[str]:
    """Split text into normalized lowercase tokens."""
    return re.findall(r"\w+", text.lower())


def _stem(word: str) -> str:
    """Basic rule-based suffix normalization for robust keyword overlap."""
    w = word.lower()
    for suffix in ("ing", "tions", "tion", "ted", "ed", "ers", "er", "al", "ment", "es", "s"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[:-len(suffix)]
    return w


def _bm25_similarity(query: str, chunks: list[str]) -> float:
    """Compute normalized semantic overlap score between query and chunks."""
    all_query_tokens = _tokenize(query)
    if not all_query_tokens or not chunks:
        return 0.0

    meaningful_query_tokens = [t for t in all_query_tokens if t not in STOP_WORDS]
    if not meaningful_query_tokens:
        meaningful_query_tokens = all_query_tokens

    unique_query_set = set(meaningful_query_tokens)
    stemmed_query_set = {_stem(t) for t in unique_query_set}

    chunk_scores: list[float] = []
    for chunk in chunks:
        chunk_tokens = _tokenize(chunk)
        if not chunk_tokens:
            chunk_scores.append(0.0)
            continue

        chunk_token_set = set(chunk_tokens)
        stemmed_chunk_set = {_stem(t) for t in chunk_tokens}

        matched_tokens = unique_query_set.intersection(chunk_token_set)
        stem_matches = stemmed_query_set.intersection(stemmed_chunk_set)

        coverage = max(len(matched_tokens) / len(unique_query_set), len(stem_matches) / len(stemmed_query_set))

        # Term frequency boost
        freq_matches = sum(1 for t in chunk_tokens if _stem(t) in stemmed_query_set)
        density_boost = min(0.2, freq_matches / max(10, len(chunk_tokens)))

        score = (0.8 * coverage) + density_boost
        chunk_scores.append(min(1.0, score))

    if not chunk_scores:
        return 0.0

    chunk_scores.sort(reverse=True)
    top_scores = chunk_scores[:3]
    mean_top = sum(top_scores) / len(top_scores)
    return max(0.0, min(1.0, mean_top))


class RetrievalScorer(BaseScorer):
    """Scorer for retrieval steps computing query-to-chunk relevance."""

    step_type: str = "retrieval"

    def score(self, step: Step) -> float:
        """Compute retrieval score in [0.0, 1.0] from step input and output."""
        query = _extract_query(step.input)
        chunks = _extract_chunks(step.output)

        if not query or not chunks:
            step.metadata["retrieval_chunks_count"] = len(chunks)
            step.metadata["retrieval_score_method"] = "empty"
            return 0.0

        model = _get_sentence_transformer()
        if model is not None:
            try:
                import numpy as np

                query_emb = model.encode([query], normalize_embeddings=True)[0]
                chunk_embs = model.encode(chunks, normalize_embeddings=True)

                sims = [float(np.dot(query_emb, c_emb)) for c_emb in chunk_embs]
                sims.sort(reverse=True)
                top_sims = sims[:3]
                mean_sim = sum(top_sims) / len(top_sims)
                clamped_score = max(0.0, min(1.0, mean_sim))

                step.metadata["top_similarity"] = round(top_sims[0], 4) if top_sims else 0.0
                step.metadata["mean_top3_similarity"] = round(mean_sim, 4)
                step.metadata["retrieval_chunks_count"] = len(chunks)
                step.metadata["retrieval_score_method"] = "sentence_transformers"
                return round(clamped_score, 4)
            except Exception as e:
                logger.debug(f"SentenceTransformer encoding failed: {e}. Using BM25 fallback.")

        # Fallback to term overlap
        bm25_score = _bm25_similarity(query, chunks)
        step.metadata["retrieval_chunks_count"] = len(chunks)
        step.metadata["retrieval_score_method"] = "bm25_fallback"
        return round(bm25_score, 4)
