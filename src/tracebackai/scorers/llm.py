"""
Traceback AI - LLM Step Scorer.

Computes a composite health score based on response completeness,
refusal detection, and multi-sample consistency.
"""

import re
from typing import Any, Sequence

from tracebackai.models import Step
from tracebackai.scorers.base import BaseScorer
from tracebackai.token_utils import count_tokens

MIN_HEALTHY_OUTPUT_TOKENS = 20

REFUSAL_PATTERNS = [
    r"\bi cannot\b",
    r"\bi am unable\b",
    r"\bi'm unable\b",
    r"\bi don't have access\b",
    r"\bi do not have access\b",
    r"\bas an ai\b",
    r"\bas a language model\b",
    r"\bi apologize, but i cannot\b",
    r"\bsorry, but i cannot\b",
    r"\bsorry, but i can't\b",
    r"\bcannot fulfill\b",
    r"\bi am not able to\b",
]

_REFUSAL_REGEXES = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]


def _longest_common_subsequence(tokens_a: list[str], tokens_b: list[str]) -> int:
    """Compute length of longest common subsequence between two token lists."""
    m, n = len(tokens_a), len(tokens_b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if tokens_a[i - 1] == tokens_b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _compute_rouge_l(text_a: str, text_b: str) -> float:
    """Compute ROUGE-L F1 score between two text strings."""
    tokens_a = re.findall(r"\w+", text_a.lower())
    tokens_b = re.findall(r"\w+", text_b.lower())
    if not tokens_a or not tokens_b:
        return 0.0
    lcs = _longest_common_subsequence(tokens_a, tokens_b)
    precision = lcs / len(tokens_b)
    recall = lcs / len(tokens_a)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _check_refusal(text: str) -> bool:
    """Check if output contains common refusal indicators."""
    if not text:
        return False
    for regex in _REFUSAL_REGEXES:
        if regex.search(text):
            return True
    return False


class LLMScorer(BaseScorer):
    """Scorer for LLM generation steps evaluating completeness, refusals, and consistency."""

    step_type: str = "llm"

    def __init__(self, min_healthy_tokens: int = MIN_HEALTHY_OUTPUT_TOKENS) -> None:
        self.min_healthy_tokens = min_healthy_tokens

    def score(self, step: Step) -> float:
        """Compute composite LLM health score."""
        out_str = str(step.output or "")
        inp_str = str(step.input or "")

        output_tokens = count_tokens(out_str) or 0
        prompt_tokens = count_tokens(inp_str) or 0

        step.metadata["response_length_tokens"] = output_tokens
        step.metadata["prompt_length_tokens"] = prompt_tokens

        # 1. Refusal Detection (Immediate penalty if refusal found)
        is_refusal = _check_refusal(out_str)
        step.metadata["refusal_detected"] = is_refusal
        if is_refusal:
            return 0.0

        # 2. Response Completeness
        question_tokens = step.metadata.get("question_tokens")
        if question_tokens and isinstance(question_tokens, (int, float)) and question_tokens > 0:
            completeness = min(1.0, output_tokens / float(question_tokens))
        else:
            completeness = min(1.0, output_tokens / float(self.min_healthy_tokens))

        # 3. Self-Consistency
        n_samples = step.metadata.get("n_samples", 1)
        samples = step.metadata.get("samples")
        if n_samples > 1 and isinstance(samples, Sequence) and len(samples) > 1:
            pairwise_scores: list[float] = []
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    pairwise_scores.append(_compute_rouge_l(str(samples[i]), str(samples[j])))
            consistency_score = sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else 0.8
        else:
            consistency_score = 0.8

        composite = (completeness + 1.0 + consistency_score) / 3.0
        final_score = max(0.0, min(1.0, composite))
        return round(final_score, 4)
