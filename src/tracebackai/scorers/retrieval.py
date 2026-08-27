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

# Thresholds for weak retrieval detection based on scoring methodology
SEMANTIC_RETRIEVAL_THRESHOLD = 0.55
BM25_RETRIEVAL_THRESHOLD = 0.33
WEAK_RETRIEVAL_THRESHOLD = 0.55  # Maintained for backward compatibility

_MODEL_INSTANCE: Any = None
_MODEL_LOAD_FAILED = False
_WARNED_FALLBACK = False

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or", "is",
    "are", "was", "were", "be", "been", "being", "with", "as", "by", "what",
    "which", "who", "whom", "this", "that", "these", "those", "how", "why",
    "do", "does", "did", "have", "has", "had", "can", "could", "should", "would",
    "from", "it", "its", "they", "them", "their", "will", "shall",
}


def get_retrieval_threshold(step_or_method: Any) -> float:
    """Return the appropriate weak retrieval threshold based on scoring method."""
    method = None
    if isinstance(step_or_method, str):
        method = step_or_method
    elif hasattr(step_or_method, "metadata") and isinstance(step_or_method.metadata, dict):
        method = step_or_method.metadata.get("retrieval_score_method")

    if method == "bm25_fallback":
        return BM25_RETRIEVAL_THRESHOLD
    return SEMANTIC_RETRIEVAL_THRESHOLD


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
    """
    Rule-based suffix normalization for robust keyword overlap.
    Handles singular/plural symmetry for -e/-es final words, y/ies, and inflections.
    """
    w = word.lower().strip()
    if len(w) <= 2:
        return w

    # Step 1: Plurals and 'y'/'ies'
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("es") and len(w) > 4:
        if w.endswith(("ses", "xes", "zes", "ches", "shes")):
            w = w[:-2]
        else:
            w = w[:-1]  # databases -> database, caches -> cache, phases -> phase
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]

    # Step 2: Suffixes
    for suffix in ("ing", "tions", "tion", "ted", "ed", "ers", "er", "al", "ment", "ments", "ity", "ities"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            w = w[:-len(suffix)]
            break

    # Step 3: Normalize 'y' and trailing 'e' for invariant base form
    if w.endswith("y") and len(w) > 3:
        w = w[:-1]  # strategy -> strateg, query -> quer
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]  # database -> databas, cache -> cach, phase -> phas, service -> servic

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

    # In IR/RAG, the top matching chunk is primary; blend top-1 with remaining top chunks
    if len(top_scores) == 1:
        raw_score = top_scores[0]
    else:
        rest_mean = sum(top_scores[1:]) / len(top_scores[1:])
        raw_score = (0.7 * top_scores[0]) + (0.3 * rest_mean)

    # Calibrate raw keyword overlap to standard [0.0, 1.0] health quality scale
    if raw_score <= 0.0:
        calibrated = 0.0
    elif raw_score < 0.25:
        calibrated = raw_score * 2.0  # [0.0, 0.25) -> [0.0, 0.50)
    else:
        calibrated = min(1.0, 0.50 + (raw_score - 0.25) * 1.5)  # [0.25, 0.60+] -> [0.50, 1.0]

    return max(0.0, min(1.0, round(calibrated, 4)))


class RetrievalScorer(BaseScorer):
    """Scorer for retrieval steps computing query-to-chunk relevance."""

    step_type: str = "retrieval"

    def __init__(self, force_method: Optional[str] = None) -> None:
        self.force_method = force_method

    def score(self, step: Step) -> float:
        """Compute retrieval score in [0.0, 1.0] from step input and output."""
        query = _extract_query(step.input)
        chunks = _extract_chunks(step.output)

        if not query or not chunks:
            step.metadata["retrieval_chunks_count"] = len(chunks)
            step.metadata["retrieval_score_method"] = "empty"
            return 0.0

        if self.force_method != "bm25_fallback":
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
