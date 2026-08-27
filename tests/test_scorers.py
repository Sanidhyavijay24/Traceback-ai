"""
Tests for Traceback AI step scorers, registry, and score_trace integration.
"""

import pytest

from tracebackai.models import Step, Trace
from tracebackai.scorers import (
    BaseScorer,
    LLMScorer,
    RetrievalScorer,
    ScorerRegistry,
    ToolScorer,
)
from tracebackai.scorers.retrieval import _bm25_similarity
from tracebackai.scoring import score_trace
from tracebackai.store import Store
from tracebackai.tracer import trace


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Isolate SQLite database for every test."""
    db_file = tmp_path / "test_scorers.db"
    monkeypatch.setenv("TRACEBACK_DB_PATH", str(db_file))
    return db_file


# ---------------------------------------------------------------------------
# Retrieval Scorer Tests
# ---------------------------------------------------------------------------


def test_retrieval_scorer_relevant_chunks():
    """Verify high score for semantically relevant retrieval."""
    scorer = RetrievalScorer()
    step = Step(
        step_type="retrieval",
        input={"query": "what is retrieval augmented generation RAG?"},
        output=[
            "Retrieval-Augmented Generation (RAG) combines search with LLMs.",
            "RAG enhances generation by retrieving relevant context passages.",
        ],
    )
    score = scorer.score(step)
    assert 0.0 <= score <= 1.0
    assert score > 0.6
    assert step.metadata["retrieval_chunks_count"] == 2


def test_retrieval_scorer_irrelevant_chunks():
    """Verify low score for irrelevant chunks (< 0.4)."""
    scorer = RetrievalScorer()
    step = Step(
        step_type="retrieval",
        input={"query": "what is quantum computing theory?"},
        output=[
            "Baking sourdough bread requires flour, water, and wild yeast culture.",
            "The recipe for chocolate chip cookies calls for brown sugar and butter.",
        ],
    )
    score = scorer.score(step)
    assert 0.0 <= score <= 1.0
    assert score < 0.4


def test_retrieval_scorer_empty_input_or_output():
    """Verify zero score on empty query or empty chunks without crashing."""
    scorer = RetrievalScorer()
    step_no_output = Step(step_type="retrieval", input="some query", output=[])
    assert scorer.score(step_no_output) == 0.0

    step_no_query = Step(step_type="retrieval", input=None, output=["chunk1"])
    assert scorer.score(step_no_query) == 0.0


def test_retrieval_scorer_dict_chunks():
    """Verify extraction of text from list of dictionary chunks."""
    scorer = RetrievalScorer()
    step = Step(
        step_type="retrieval",
        input="python programming language",
        output=[
            {"text": "Python is a high-level interpreted programming language.", "id": 1},
            {"text": "Guido van Rossum created Python programming in 1991.", "id": 2},
        ],
    )
    score = scorer.score(step)
    assert score > 0.6


def test_retrieval_bm25_similarity():
    """Verify standalone BM25 term overlap calculation."""
    query = "vector database indexing"
    chunks = [
        "Vector database systems use approximate nearest neighbor indexing.",
        "Vector indexing stores high dimensional embeddings in database structures.",
    ]
    bm25_val = _bm25_similarity(query, chunks)
    assert 0.0 <= bm25_val <= 1.0
    assert bm25_val > 0.6


def test_stem_singular_plural_pairs():
    """Verify _stem normalizes singular and plural pairs to identical base stems."""
    from tracebackai.scorers.retrieval import _stem

    pairs = [
        ("database", "databases"),
        ("phase", "phases"),
        ("cache", "caches"),
        ("service", "services"),
        ("strategy", "strategies"),
        ("query", "queries"),
        ("index", "indexes"),
        ("table", "tables"),
        ("clause", "clauses"),
        ("device", "devices"),
    ]
    for singular, plural in pairs:
        stem_sing = _stem(singular)
        stem_plur = _stem(plural)
        assert stem_sing == stem_plur, (
            f"Stem mismatch for pair ({singular}, {plural}): '{stem_sing}' != '{stem_plur}'"
        )


def test_retrieval_bm25_database_indexing_repro():
    """
    Verify concrete repro query 'How do database indexing strategies work?' scores above
    BM25 threshold and is not marked as weak retrieval under the BM25 fallback path.
    """
    from tracebackai.scorers.retrieval import BM25_RETRIEVAL_THRESHOLD

    scorer = RetrievalScorer(force_method="bm25_fallback")
    query = "How do database indexing strategies work?"
    chunks = [
        "B-tree indexes are the default index type in most relational databases.",
        "A B-tree index speeds up lookups, range queries, and sorted retrieval.",
        "Composite indexes cover queries filtering on multiple columns together.",
    ]
    step = Step(step_type="retrieval", input=query, output=chunks)
    score = scorer.score(step)

    assert step.metadata["retrieval_score_method"] == "bm25_fallback"
    assert score >= BM25_RETRIEVAL_THRESHOLD, (
        f"BM25 retrieval score {score} fell below BM25 threshold {BM25_RETRIEVAL_THRESHOLD}"
    )


def test_get_retrieval_threshold():
    """Verify get_retrieval_threshold returns method-specific thresholds."""
    from tracebackai.scorers.retrieval import (
        BM25_RETRIEVAL_THRESHOLD,
        SEMANTIC_RETRIEVAL_THRESHOLD,
        get_retrieval_threshold,
    )

    bm25_step = Step(
        step_type="retrieval",
        input="q",
        output=["c"],
        metadata={"retrieval_score_method": "bm25_fallback"},
    )
    semantic_step = Step(
        step_type="retrieval",
        input="q",
        output=["c"],
        metadata={"retrieval_score_method": "sentence_transformers"},
    )

    assert get_retrieval_threshold(bm25_step) == BM25_RETRIEVAL_THRESHOLD
    assert get_retrieval_threshold("bm25_fallback") == BM25_RETRIEVAL_THRESHOLD
    assert get_retrieval_threshold(semantic_step) == SEMANTIC_RETRIEVAL_THRESHOLD
    assert get_retrieval_threshold("sentence_transformers") == SEMANTIC_RETRIEVAL_THRESHOLD
    assert get_retrieval_threshold(None) == SEMANTIC_RETRIEVAL_THRESHOLD


# ---------------------------------------------------------------------------
# LLM Scorer Tests
# ---------------------------------------------------------------------------


def test_llm_scorer_healthy_response():
    """Verify healthy comprehensive response receives high score (> 0.75)."""
    scorer = LLMScorer()
    step = Step(
        step_type="llm",
        input="Explain the concept of backpropagation in deep neural networks.",
        output=(
            "Backpropagation is an algorithm used in machine learning to compute the "
            "gradient of the loss function with respect to the weights of the network. "
            "It applies the chain rule of calculus backward through each layer."
        ),
    )
    score = scorer.score(step)
    assert score > 0.75
    assert step.metadata["refusal_detected"] is False
    assert step.metadata["response_length_tokens"] > 20


def test_llm_scorer_refusal_detection():
    """Verify refusal responses are detected and heavily penalized (< 0.2)."""
    scorer = LLMScorer()
    step = Step(
        step_type="llm",
        input="How do I bypass authentication?",
        output="I cannot fulfill this request. As an AI language model, I must adhere to safety policies.",
    )
    score = scorer.score(step)
    assert score < 0.2
    assert step.metadata["refusal_detected"] is True


def test_llm_scorer_short_response_penalized():
    """Verify unusually short output receives proportionally lower completeness."""
    scorer = LLMScorer(min_healthy_tokens=20)
    step = Step(
        step_type="llm",
        input="Give me a comprehensive overview of machine learning history.",
        output="It started long ago.",
    )
    score = scorer.score(step)
    assert score < 0.75


def test_llm_scorer_short_answer_with_huge_rag_prompt():
    """
    CRITICAL REGRESSION TEST: A concise factual answer (25+ tokens) to a 2000-token RAG prompt
    must score 1.0 on completeness sub-score and NOT be penalized by total input prompt size.
    """
    scorer = LLMScorer(min_healthy_tokens=20)
    huge_prompt = "Context: " + ("Large context chunk with detailed technical specification. " * 100)
    answer = "The capital of France is Paris, which is widely recognized as the country's most populous city and its primary political and cultural center."

    step = Step(
        step_type="llm",
        input=huge_prompt,
        output=answer,
    )
    score = scorer.score(step)
    assert score >= 0.85
    assert step.metadata["prompt_length_tokens"] > 500
    assert step.metadata["response_length_tokens"] >= 20


def test_llm_scorer_multi_sample_consistency():
    """Verify self-consistency evaluation across multiple samples."""
    scorer = LLMScorer()
    step = Step(
        step_type="llm",
        input="What is 2 + 2?",
        output="The result of calculating two plus two is four in standard arithmetic operations.",
        metadata={
            "n_samples": 3,
            "samples": [
                "The result of two plus two is four.",
                "The calculation of two plus two gives four.",
                "Adding two plus two results in four.",
            ],
        },
    )
    score = scorer.score(step)
    assert score > 0.70


# ---------------------------------------------------------------------------
# Tool Scorer Tests
# ---------------------------------------------------------------------------


def test_tool_scorer_valid_output(tmp_path):
    """Verify tool returning valid structured data scores high (> 0.8)."""
    db_file = tmp_path / "tool_test.db"
    store = Store(db_path=str(db_file))
    scorer = ToolScorer(store=store)

    step = Step(
        step_type="tool",
        name="fetch_weather",
        input={"city": "Tokyo"},
        output={"temperature_c": 18.5, "condition": "Sunny"},
        metadata={"expected_type": "dict"},
    )
    score = scorer.score(step)
    assert score >= 0.9


def test_tool_scorer_empty_output():
    """Verify empty/null tool output receives low score (0.2)."""
    scorer = ToolScorer()
    step_none = Step(step_type="tool", name="db_lookup", output=None)
    assert scorer.score(step_none) == 0.2

    step_empty = Step(step_type="tool", name="db_lookup", output={})
    assert scorer.score(step_empty) == 0.2


def test_tool_scorer_error_step():
    """Verify step with error scores 0.0."""
    scorer = ToolScorer()
    step = Step(
        step_type="tool",
        name="api_call",
        error="ConnectionTimeoutError: Server took too long to respond",
    )
    assert scorer.score(step) == 0.0


def test_tool_scorer_expected_type_mismatch():
    """Verify mismatch between expected_type and output penalizes score."""
    scorer = ToolScorer()
    step = Step(
        step_type="tool",
        name="parse_json",
        output="plain string instead of dict",
        metadata={"expected_type": "dict"},
    )
    score = scorer.score(step)
    assert score == 0.5


def test_tool_scorer_historical_error_rate(tmp_path):
    """Verify tool health score is penalized by historical error rate."""
    db_file = tmp_path / "tool_hist.db"
    store = Store(db_path=str(db_file))

    for _ in range(5):
        s_ok = Step(name="flaky_api", step_type="tool", output={"ok": True})
        t_ok = Trace(pipeline_name="p", steps=[s_ok])
        store.save_trace(t_ok)

        s_err = Step(name="flaky_api", step_type="tool", error="API 500 error")
        t_err = Trace(pipeline_name="p", steps=[s_err])
        store.save_trace(t_err)

    scorer = ToolScorer(store=store)
    step = Step(
        name="flaky_api",
        step_type="tool",
        output={"data": "success"},
    )
    score = scorer.score(step)
    assert score == pytest.approx(0.5, abs=0.05)
    assert step.metadata["historical_error_rate"] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# Registry & score_trace Tests
# ---------------------------------------------------------------------------


def test_scorer_registry():
    """Verify registry correctly dispatches by step_type."""
    registry = ScorerRegistry()
    assert isinstance(registry.get("retrieval"), RetrievalScorer)
    assert isinstance(registry.get("llm"), LLMScorer)
    assert isinstance(registry.get("tool"), ToolScorer)
    assert registry.get("generic") is None


def test_score_trace_mutates_steps():
    """Verify score_trace applies scores to typed steps while preserving unscored ones."""
    step_ret = Step(
        step_type="retrieval",
        input="python programming language",
        output=["Python programming documentation guide"],
    )
    step_llm = Step(
        step_type="llm",
        input="prompt",
        output="This is a fully complete and detailed answer to the prompt question with plenty of explanation.",
    )
    step_gen = Step(
        step_type="generic",
        name="format_text",
        input="raw",
        output="formatted",
    )

    trace_obj = Trace(pipeline_name="test_pipe", steps=[step_ret, step_llm, step_gen])
    score_trace(trace_obj)

    assert step_ret.score is not None
    assert step_llm.score is not None
    assert step_gen.score is None


def test_store_has_no_dependency_on_scorers():
    """Verify architectural requirement: store.py has no import from scorers/."""
    import tracebackai.store as store_mod

    for name in dir(store_mod):
        obj = getattr(store_mod, name)
        module_name = getattr(obj, "__module__", "")
        assert "tracebackai.scorers" not in module_name


def test_end_to_end_traced_pipeline_with_scoring():
    """Verify complete traced pipeline automatically calculates and saves step scores."""

    @trace(step_type="retrieval")
    def find_docs(q: str) -> list[str]:
        return ["RAG is a technique to supply retrieval context to an LLM pipeline."]

    @trace(step_type="llm")
    def generate_ans(prompt: str) -> str:
        return "RAG combines search with generative intelligence to produce grounded responses for user questions."

    @trace(step_type="tool")
    def save_metric(val: str) -> dict:
        return {"saved": True, "val": val}

    @trace(pipeline=True)
    def full_pipeline(q: str) -> str:
        docs = find_docs(q)
        ans = generate_ans(docs[0])
        save_metric(ans)
        return ans

    res = full_pipeline("what is RAG?")
    assert "RAG combines search" in res

    store = Store()
    runs = store.list_runs()
    assert len(runs) == 1

    loaded = store.load_trace(runs[0]["run_id"])
    steps_by_name = {s.name: s for s in loaded.steps}

    assert steps_by_name["find_docs"].score is not None
    assert steps_by_name["find_docs"].score > 0.5

    assert steps_by_name["generate_ans"].score is not None
    assert steps_by_name["generate_ans"].score > 0.75

    assert steps_by_name["save_metric"].score is not None
    assert steps_by_name["save_metric"].score >= 0.9
