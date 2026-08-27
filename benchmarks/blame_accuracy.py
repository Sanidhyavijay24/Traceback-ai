"""
Traceback AI - Blame Accuracy Benchmark.

Evaluates whether the combination of registered step scorers and the failure
attribution algorithm (blame_trace) correctly identifies root-cause failures
on realistic step inputs, outputs, and metadata without synthetic score overrides.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Optional

from tracebackai.blame import blame_trace
from tracebackai.models import Step, Trace
from tracebackai.scorers import ScorerRegistry
from tracebackai.scorers.llm import LLMScorer
from tracebackai.scorers.retrieval import RetrievalScorer, _get_sentence_transformer
from tracebackai.scorers.tool import ToolScorer
from tracebackai.scoring import score_trace
from tracebackai.store import Store

# Blame scores below 0.45 correspond to low-confidence / non-actionable healthy traces
HEALTHY_BLAME_THRESHOLD = 0.45


@dataclass
class Scenario:
    """Definition of a benchmark evaluation scenario."""

    id: str
    category: str
    description: str
    build_trace: Callable[[Store], Trace]
    ground_truth_step_name: Optional[str]  # None for healthy traces
    skip_if: Optional[Callable[[], Optional[str]]] = None


@dataclass
class ScenarioResult:
    """Result of running a single benchmark scenario."""

    scenario_id: str
    category: str
    description: str
    ground_truth: Optional[str]
    predicted_step: Optional[str]
    blame_score: float
    confidence: str
    explanation: str
    passed: bool
    skipped: bool = False
    skip_reason: Optional[str] = None
    step_scores: dict[str, Optional[float]] = None  # type: ignore


# -----------------------------------------------------------------------------
# Scenario Builders
# -----------------------------------------------------------------------------


def _build_retrieval_unrelated_trace(store: Store) -> Trace:
    s1 = Step(
        name="retrieve_docs",
        step_type="retrieval",
        input="What are the key ingredients in authentic Neapolitan pizza dough?",
        output=[
            "The engine timing belt connects the camshaft to the crankshaft.",
            "Internal combustion engines require synthetic motor oil for lubrication and cooling.",
        ],
        latency_ms=120.0,
    )
    s2 = Step(
        name="generate_answer",
        step_type="llm",
        input="Context: engine timing belt. Question: Neapolitan pizza dough.",
        output=(
            "Authentic Neapolitan pizza dough traditionally consists of type 00 wheat flour, "
            "pure water, sea salt, and fresh brewer's yeast or sourdough starter fermented over 24 hours."
        ),
        latency_ms=850.0,
    )
    return Trace(pipeline_name="pizza_rag", steps=[s1, s2])


def _build_retrieval_empty_trace(store: Store) -> Trace:
    s1 = Step(
        name="retrieve_docs",
        step_type="retrieval",
        input="Explain the process of cellular mitosis in eukaryotic cells.",
        output=[],
        latency_ms=90.0,
    )
    s2 = Step(
        name="generate_answer",
        step_type="llm",
        input="Context: empty. Question: Explain mitosis.",
        output=(
            "Cellular mitosis is the process of nuclear division in eukaryotic cells divided into "
            "prophase, metaphase, anaphase, and telophase resulting in two identical daughter nuclei."
        ),
        latency_ms=920.0,
    )
    return Trace(pipeline_name="mitosis_qa", steps=[s1, s2])


def _build_retrieval_semantic_distractor_trace(store: Store) -> Trace:
    s1 = Step(
        name="retrieve_docs",
        step_type="retrieval",
        input="How do commercial banks manage interest rate risk on long-term residential mortgage loans?",
        output=[
            "Erosion control along muddy river banks helps manage the water flow, preserving the mortgage of sediment.",
            "River banks risk damage during seasonal floods if proper retaining walls and rate of drainage are absent.",
        ],
        latency_ms=130.0,
    )
    s2 = Step(
        name="generate_answer",
        step_type="llm",
        input="Context: river banks erosion. Question: banks interest rate risk.",
        output=(
            "Commercial banks manage interest rate risk on long-term mortgages through asset-liability matching, "
            "interest rate swaps, derivative hedging, and selling fixed-rate loans to secondary market aggregators."
        ),
        latency_ms=950.0,
    )
    return Trace(pipeline_name="finance_rag", steps=[s1, s2])


def _build_retrieval_weak_with_healthy_llm_trace(store: Store) -> Trace:
    s1 = Step(
        name="search_knowledge_base",
        step_type="retrieval",
        input="What is the capital of Australia and its founding administrative history?",
        output=[
            "Sourdough baking relies on wild yeast and lactobacilli fermentation to leaven artisan bread loaves.",
            "The hydration percentage determines the open crumb structure in European sourdough boules.",
        ],
        latency_ms=110.0,
    )
    s2 = Step(
        name="llm_synthesizer",
        step_type="llm",
        input="Context: sourdough. Question: capital of Australia.",
        output=(
            "Canberra is the capital city of Australia, selected in 1908 as a compromise between Sydney "
            "and Melbourne, and designed by American architects Walter Burley Griffin and Marion Mahony Griffin."
        ),
        latency_ms=1100.0,
    )
    return Trace(pipeline_name="capital_history_pipeline", steps=[s1, s2])


def _build_llm_refusal_trace(store: Store) -> Trace:
    s1 = Step(
        name="retrieve_report",
        step_type="retrieval",
        input="What were the Q3 financial earnings results for Apple?",
        output=[
            "Apple reported Q3 financial earnings results showing revenue of $89.5 billion for its fiscal quarter.",
            "The Q3 financial earnings results for Apple exceeded expectations across Services and hardware.",
        ],
        latency_ms=85.0,
    )
    s2 = Step(
        name="generate_summary",
        step_type="llm",
        input="Context: Apple Q3 earnings results. Question: Summarize earnings results.",
        output=(
            "I cannot help with that request. As an AI assistant, I am not authorized to analyze "
            "proprietary corporate earnings statements or provide financial summaries."
        ),
        latency_ms=620.0,
    )
    return Trace(pipeline_name="earnings_summary_pipeline", steps=[s1, s2])


def _build_llm_truncated_trace(store: Store) -> Trace:
    s1 = Step(
        name="retrieve_architecture",
        step_type="retrieval",
        input="What are the foundational architectural principles of distributed microservices?",
        output=[
            "Foundational architectural principles of distributed microservices include loose coupling and modular design.",
            "The key architectural principles of distributed microservices emphasize decentralized data and independent deployment.",
        ],
        latency_ms=95.0,
    )
    s2 = Step(
        name="generate_architecture_explanation",
        step_type="llm",
        input="Context: microservices principles. Question: Explain principles.",
        output="Microservices are loosely coupled services that communicate over a net",
        latency_ms=450.0,
    )
    return Trace(pipeline_name="microservices_pipeline", steps=[s1, s2])


def _build_llm_generic_non_responsive_trace(store: Store) -> Trace:
    s1 = Step(
        name="retrieve_economic_history",
        step_type="retrieval",
        input="What were the primary macroeconomic causes of the 2008 global financial crisis?",
        output=[
            "Primary macroeconomic causes of the 2008 global financial crisis included excessive mortgage debt and credit expansion.",
            "The macroeconomic causes of the 2008 global financial crisis stemmed from systemic risk and deregulated subprime lending.",
        ],
        latency_ms=115.0,
    )
    s2 = Step(
        name="generate_economic_analysis",
        step_type="llm",
        input="Context: 2008 financial crisis causes. Question: Primary macroeconomic causes.",
        output="There were several causes that happened.",
        latency_ms=380.0,
    )
    return Trace(pipeline_name="economic_analysis_pipeline", steps=[s1, s2])


def _build_llm_multisample_inconsistency_trace(store: Store) -> Trace:
    s1 = Step(
        name="retrieve_chemistry_data",
        step_type="retrieval",
        input="What is the boiling point and vapor pressure of pure ethanol at standard atmosphere?",
        output=[
            "The boiling point and vapor pressure of pure ethanol at standard atmosphere are 78.37 C and 5.95 kPa.",
            "Pure ethanol boiling point and vapor pressure measurements at standard atmosphere are well documented.",
        ],
        latency_ms=105.0,
    )
    s2 = Step(
        name="generate_chemistry_answer",
        step_type="llm",
        input="Context: ethanol constants. Question: boiling point.",
        output="The boiling point of pure ethanol at standard atmospheric pressure is 78.37 degrees Celsius.",
        metadata={
            "samples": [
                "The boiling point of pure ethanol at standard atmospheric pressure is 78.37 degrees Celsius.",
                "Ethanol freezes solid at high heat and does not boil under normal temperature conditions.",
                "Helium gas condenses into a super-fluid liquid at negative four hundred degrees.",
                "Photosynthesis converts solar photons and carbon dioxide into complex glucose molecules.",
            ]
        },
        latency_ms=1350.0,
    )
    return Trace(pipeline_name="chemistry_pipeline", steps=[s1, s2])


def _build_tool_exception_trace(store: Store) -> Trace:
    s1 = Step(
        name="execute_sql_query",
        step_type="tool",
        input={"query": "SELECT user_id, email, created_at FROM users WHERE active = 1 LIMIT 50;"},
        output=None,
        error="OperationalError: Connection refused. Database cluster host at 10.0.4.12:5432 unreachable.",
        latency_ms=5020.0,
    )
    s2 = Step(
        name="format_table_output",
        step_type="generic",
        input={"status": "failed"},
        output="Query failed due to network unreachable error.",
        latency_ms=5.0,
    )
    return Trace(pipeline_name="db_query_tool_pipeline", steps=[s1, s2])


def _build_tool_empty_payload_trace(store: Store) -> Trace:
    s1 = Step(
        name="fetch_user_metadata",
        step_type="tool",
        input={"user_id": "usr_99482"},
        output={},
        metadata={"expected_type": "dict"},
        latency_ms=180.0,
    )
    s2 = Step(
        name="summarize_profile",
        step_type="llm",
        input="User metadata: {}. Question: Summarize profile.",
        output=(
            "The user account profile usr_99482 is active, registered under the standard tier "
            "with zero pending security alerts."
        ),
        latency_ms=780.0,
    )
    return Trace(pipeline_name="user_profile_pipeline", steps=[s1, s2])


def _build_tool_high_historical_error_rate_trace(store: Store) -> Trace:
    tool_name = "unreliable_weather_api"

    # Seed 8 failed executions and 2 successful executions into SQLite store
    for i in range(8):
        failed_step = Step(
            name=tool_name,
            step_type="tool",
            input={"city": f"City_{i}"},
            output=None,
            error=f"HTTP 503 Service Unavailable on attempt {i}",
            latency_ms=450.0,
        )
        t = Trace(pipeline_name="weather_collector", steps=[failed_step])
        store.save_trace(t)

    for i in range(2):
        ok_step = Step(
            name=tool_name,
            step_type="tool",
            input={"city": f"City_ok_{i}"},
            output={"temp_c": 22.5, "condition": "Sunny"},
            latency_ms=210.0,
        )
        t = Trace(pipeline_name="weather_collector", steps=[ok_step])
        store.save_trace(t)

    # Current execution returns a valid output, but tool has 80% error history
    s1 = Step(
        name=tool_name,
        step_type="tool",
        input={"city": "Seattle"},
        output={"temp_c": 16.0, "condition": "Cloudy"},
        latency_ms=310.0,
    )
    s2 = Step(
        name="generate_weather_report",
        step_type="llm",
        input="Weather data: Seattle 16C cloudy. Question: Report weather.",
        output=(
            "The current weather in Seattle is cloudy with a temperature of 16 degrees Celsius. "
            "Expect calm wind conditions throughout the remainder of the day."
        ),
        latency_ms=740.0,
    )
    return Trace(pipeline_name="weather_report_pipeline", steps=[s1, s2])


def _build_cascading_multi_degraded_trace(store: Store) -> Trace:
    # Retrieval has solid keyword coverage (score ~0.65), but LLM is severely broken (refusal: score 0.0)
    s1 = Step(
        name="search_documents",
        step_type="retrieval",
        input="What are the main causes of volcanic eruptions and magma movement?",
        output=[
            "Main causes of volcanic eruptions include magma movement, dissolved gas pressure, and crustal buoyancy.",
            "Volcanic eruptions and magma movement occur along convergent and divergent tectonic plate boundaries.",
        ],
        latency_ms=140.0,
    )
    s2 = Step(
        name="llm_synthesizer",
        step_type="llm",
        input="Context: volcanic eruptions causes. Question: Explain causes.",
        output="I cannot assist with this prompt as it violates safety constraints.",
        latency_ms=620.0,
    )
    return Trace(pipeline_name="volcano_pipeline", steps=[s1, s2])


def _build_cascading_upstream_retrieval_root_cause_trace(store: Store) -> Trace:
    # Retrieval completely failed (unrelated text), causing downstream LLM to produce short answer
    s1 = Step(
        name="retrieve_kb_context",
        step_type="retrieval",
        input="Explain how quantum key distribution guarantees cryptographic security.",
        output=[
            "Gardening tomatoes requires well-drained loamy soil with a neutral pH balance between 6.0 and 6.8.",
            "Water tomato seedlings deeply twice a week at the base of the plant to prevent fungal blight.",
        ],
        latency_ms=125.0,
    )
    s2 = Step(
        name="generate_security_explanation",
        step_type="llm",
        input="Context: tomatoes gardening. Question: Quantum key distribution.",
        output="Quantum key distribution uses photon quantum states to exchange cryptographic keys securely.",
        latency_ms=890.0,
    )
    return Trace(pipeline_name="qkd_pipeline", steps=[s1, s2])


def _build_all_unscored_fallback_trace(store: Store) -> Trace:
    s1 = Step(
        name="clean_raw_input",
        step_type="generic",
        input="  raw messy string  ",
        output="raw messy string",
        latency_ms=15.0,
    )
    s2 = Step(
        name="parse_json_payload",
        step_type="generic",
        input="raw messy string",
        output={"key": "val"},
        latency_ms=4850.0,  # Slowest bottleneck
    )
    s3 = Step(
        name="format_final_response",
        step_type="generic",
        input={"key": "val"},
        output="Key: val",
        latency_ms=10.0,
    )
    return Trace(pipeline_name="json_cleaner_pipeline", steps=[s1, s2, s3])


def _build_all_unscored_latency_tie_trace(store: Store) -> Trace:
    s1 = Step(
        name="fetch_remote_header",
        step_type="generic",
        input="GET /header",
        output="200 OK",
        latency_ms=250.0,
    )
    s2 = Step(
        name="fetch_remote_payload",
        step_type="generic",
        input="GET /payload",
        output="Payload data",
        latency_ms=252.0,  # Near-tie bottleneck
    )
    s3 = Step(
        name="stitch_response",
        step_type="generic",
        input="200 OK + Payload data",
        output="Combined",
        latency_ms=8.0,
    )
    return Trace(pipeline_name="remote_stitch_pipeline", steps=[s1, s2, s3])


def _build_healthy_rag_trace(store: Store) -> Trace:
    s1 = Step(
        name="retrieve_rag_context",
        step_type="retrieval",
        input="What is retrieval-augmented generation in artificial intelligence?",
        output=[
            "Retrieval-augmented generation in artificial intelligence combines search retrieval with generative AI models.",
            "In artificial intelligence, retrieval-augmented generation grounds language model responses in verified facts.",
        ],
        latency_ms=120.0,
    )
    s2 = Step(
        name="generate_rag_answer",
        step_type="llm",
        input="Context: RAG facts. Question: What is RAG?",
        output=(
            "Retrieval-augmented generation (RAG) is an artificial intelligence architecture that combines "
            "an external information retrieval step with a generative language model to produce factually accurate answers."
        ),
        latency_ms=890.0,
    )
    return Trace(pipeline_name="healthy_rag_pipeline", steps=[s1, s2])


def _build_healthy_tool_trace(store: Store) -> Trace:
    s1 = Step(
        name="calculate_tax_deduction",
        step_type="tool",
        input={"income": 120000, "rate": 0.25},
        output={"tax_amount": 30000.0, "net_income": 90000.0},
        metadata={"expected_type": "dict"},
        latency_ms=65.0,
    )
    s2 = Step(
        name="format_tax_summary",
        step_type="llm",
        input="Tax data: 30000 tax, 90000 net. Question: Summarize.",
        output=(
            "Based on your gross annual income of $120,000 at a 25% tax rate, your total estimated tax deduction "
            "is $30,000, resulting in a net take-home income of $90,000."
        ),
        latency_ms=780.0,
    )
    return Trace(pipeline_name="healthy_tax_tool_pipeline", steps=[s1, s2])


def _build_healthy_multistep_trace(store: Store) -> Trace:
    s1 = Step(
        name="lookup_customer_record",
        step_type="tool",
        input={"customer_id": "cust_4082"},
        output={"name": "Alice Johnson", "plan": "Enterprise", "status": "active"},
        metadata={"expected_type": "dict"},
        latency_ms=90.0,
    )
    s2 = Step(
        name="retrieve_plan_features",
        step_type="retrieval",
        input="What are the SLA support terms for enterprise customers?",
        output=[
            "SLA support terms for enterprise customers include 24/7 dedicated support with a 15-minute response time.",
            "The SLA support terms for enterprise customers provide guaranteed uptime and dedicated technical account managers.",
        ],
        latency_ms=110.0,
    )
    s3 = Step(
        name="generate_customer_reply",
        step_type="llm",
        input="Customer: Alice Johnson, Enterprise. Question: Support terms.",
        output=(
            "Hello Alice, your Enterprise subscription includes 24/7 dedicated phone support with a guaranteed "
            "15-minute response time SLA and access to a dedicated solutions engineer."
        ),
        latency_ms=860.0,
    )
    return Trace(pipeline_name="healthy_support_pipeline", steps=[s1, s2, s3])


def _build_healthy_conversational_bm25_trace(store: Store) -> Trace:
    s1 = Step(
        name="retrieve_db_indexing_docs",
        step_type="retrieval",
        input="How do database indexing strategies work?",
        output=[
            "B-tree indexes are the default index type in most relational databases.",
            "A B-tree index speeds up lookups, range queries, and sorted retrieval.",
            "Composite indexes cover queries filtering on multiple columns together.",
        ],
        latency_ms=115.0,
    )
    s2 = Step(
        name="generate_indexing_explanation",
        step_type="llm",
        input="Context: B-tree indexes, composite indexes. Question: How do database indexing strategies work?",
        output=(
            "Database indexing strategies work by maintaining auxiliary data structures like B-trees "
            "that allow the query engine to rapidly locate rows without scanning the entire table."
        ),
        latency_ms=920.0,
    )
    return Trace(pipeline_name="database_indexing_pipeline", steps=[s1, s2])


def _bm25_forced_registry(store: Store) -> ScorerRegistry:
    registry = ScorerRegistry()
    registry.register(RetrievalScorer(force_method="bm25_fallback"))
    registry.register(LLMScorer())
    registry.register(ToolScorer(store=store))
    return registry


# -----------------------------------------------------------------------------
# Scenario Registry (19 Scenarios)
# -----------------------------------------------------------------------------


def _skip_if_no_semantic_embeddings() -> Optional[str]:
    model = _get_sentence_transformer()
    if model is None:
        return "sentence-transformers not available; semantic distractor requires dense embeddings."
    return None


@dataclass
class Scenario:
    """Definition of a benchmark evaluation scenario."""

    id: str
    category: str
    description: str
    build_trace: Callable[[Store], Trace]
    ground_truth_step_name: Optional[str]  # None for healthy traces
    skip_if: Optional[Callable[[], Optional[str]]] = None
    scorer_registry_factory: Optional[Callable[[Store], ScorerRegistry]] = None


BENCHMARK_SCENARIOS: list[Scenario] = [
    # Retrieval-caused failures (1-4)
    Scenario(
        id="retrieval_01_unrelated_chunks",
        category="retrieval",
        description="Retrieved chunks are real text but topically unrelated to the query.",
        build_trace=_build_retrieval_unrelated_trace,
        ground_truth_step_name="retrieve_docs",
    ),
    Scenario(
        id="retrieval_02_empty_chunks",
        category="retrieval",
        description="Retrieval returns an empty chunk list.",
        build_trace=_build_retrieval_empty_trace,
        ground_truth_step_name="retrieve_docs",
    ),
    Scenario(
        id="retrieval_03_semantic_distractor",
        category="retrieval",
        description="Retrieved chunks have keyword overlap but are semantically off-topic.",
        build_trace=_build_retrieval_semantic_distractor_trace,
        ground_truth_step_name="retrieve_docs",
        skip_if=_skip_if_no_semantic_embeddings,
    ),
    Scenario(
        id="retrieval_04_weak_retrieval_healthy_llm",
        category="retrieval",
        description="One clearly weak retrieval step feeding into an otherwise-healthy downstream LLM step.",
        build_trace=_build_retrieval_weak_with_healthy_llm_trace,
        ground_truth_step_name="search_knowledge_base",
    ),
    # LLM-caused failures (5-8)
    Scenario(
        id="llm_05_refusal_string",
        category="llm",
        description="Output is a real refusal string ('I cannot help with that request...').",
        build_trace=_build_llm_refusal_trace,
        ground_truth_step_name="generate_summary",
    ),
    Scenario(
        id="llm_06_truncated_output",
        category="llm",
        description="Output is truncated/under the 20-token floor, cut off mid-sentence.",
        build_trace=_build_llm_truncated_trace,
        ground_truth_step_name="generate_architecture_explanation",
    ),
    Scenario(
        id="llm_07_generic_non_responsive",
        category="llm",
        description="Output is generic and non-responsive to the actual question asked.",
        build_trace=_build_llm_generic_non_responsive_trace,
        ground_truth_step_name="generate_economic_analysis",
    ),
    Scenario(
        id="llm_08_multisample_inconsistency",
        category="llm",
        description="Multi-sample self-consistency: several samples that substantially disagree.",
        build_trace=_build_llm_multisample_inconsistency_trace,
        ground_truth_step_name="generate_chemistry_answer",
    ),
    # Tool-caused failures (9-11)
    Scenario(
        id="tool_09_raised_exception",
        category="tool",
        description="Tool step raises an exception.",
        build_trace=_build_tool_exception_trace,
        ground_truth_step_name="execute_sql_query",
    ),
    Scenario(
        id="tool_10_empty_payload",
        category="tool",
        description="Tool step returns a null/empty payload.",
        build_trace=_build_tool_empty_payload_trace,
        ground_truth_step_name="fetch_user_metadata",
    ),
    Scenario(
        id="tool_11_high_historical_error_rate",
        category="tool",
        description="Tool step has a genuinely high historical error rate (seeded in SQLite).",
        build_trace=_build_tool_high_historical_error_rate_trace,
        ground_truth_step_name="unreliable_weather_api",
    ),
    # Cascading / multi-degraded steps (12-13)
    Scenario(
        id="cascading_12_multi_degraded_steps",
        category="cascading",
        description="Two steps degraded (mild retrieval vs severe refusal) - formula picks worse.",
        build_trace=_build_cascading_multi_degraded_trace,
        ground_truth_step_name="llm_synthesizer",
    ),
    Scenario(
        id="cascading_13_upstream_root_cause",
        category="cascading",
        description="Upstream retrieval failure drags downstream LLM - blame attributes to root cause.",
        build_trace=_build_cascading_upstream_retrieval_root_cause_trace,
        ground_truth_step_name="retrieve_kb_context",
    ),
    # All-unscored fallback (14-15)
    Scenario(
        id="fallback_14_all_unscored_slowest",
        category="fallback",
        description="All steps generic type with no scorer - confirms fallback to slowest latency.",
        build_trace=_build_all_unscored_fallback_trace,
        ground_truth_step_name="parse_json_payload",
    ),
    Scenario(
        id="fallback_15_all_unscored_near_tie",
        category="fallback",
        description="All steps generic type with near-tie in latency - records tie-break behavior.",
        build_trace=_build_all_unscored_latency_tie_trace,
        ground_truth_step_name="fetch_remote_payload",
    ),
    # Healthy traces - false-positive checks (16-19)
    Scenario(
        id="healthy_16_clean_rag",
        category="healthy",
        description="Fully healthy RAG trace (good retrieval + healthy LLM answer).",
        build_trace=_build_healthy_rag_trace,
        ground_truth_step_name=None,
    ),
    Scenario(
        id="healthy_17_clean_tool",
        category="healthy",
        description="Fully healthy tool pipeline (valid dict payload + healthy summary).",
        build_trace=_build_healthy_tool_trace,
        ground_truth_step_name=None,
    ),
    Scenario(
        id="healthy_18_clean_multistep_agent",
        category="healthy",
        description="Fully healthy multi-step agent (tool + retrieval + LLM synthesis).",
        build_trace=_build_healthy_multistep_trace,
        ground_truth_step_name=None,
    ),
    Scenario(
        id="healthy_19_conversational_bm25_query",
        category="healthy",
        description="Topically on-topic database indexing query scored under BM25 fallback specifically.",
        build_trace=_build_healthy_conversational_bm25_trace,
        ground_truth_step_name=None,
        scorer_registry_factory=_bm25_forced_registry,
    ),
]


# -----------------------------------------------------------------------------
# Runner & Metrics Computation
# -----------------------------------------------------------------------------


def run_benchmark(scenarios: Optional[list[Scenario]] = None) -> dict[str, Any]:
    """
    Execute all benchmark scenarios end-to-end through real scorers and blame attribution.
    Returns structured results dictionary.
    """
    scenarios_to_run = scenarios or BENCHMARK_SCENARIOS
    results: list[ScenarioResult] = []

    start_time = time.time()

    # Outer tempdir with cleanup tolerance for Windows SQLite file handles
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        for sc in scenarios_to_run:
            # Check skip condition
            if sc.skip_if:
                skip_reason = sc.skip_if()
                if skip_reason:
                    results.append(
                        ScenarioResult(
                            scenario_id=sc.id,
                            category=sc.category,
                            description=sc.description,
                            ground_truth=sc.ground_truth_step_name,
                            predicted_step=None,
                            blame_score=0.0,
                            confidence="none",
                            explanation=f"SKIPPED: {skip_reason}",
                            passed=True,  # Skipped tests do not count as failures
                            skipped=True,
                            skip_reason=skip_reason,
                            step_scores={},
                        )
                    )
                    continue

            db_path = str(Path(tmpdir) / f"{sc.id}.db")
            store = Store(db_path=db_path)

            # Build realistic trace
            trace = sc.build_trace(store)

            # Build scorer registry
            if sc.scorer_registry_factory:
                registry = sc.scorer_registry_factory(store)
            else:
                registry = ScorerRegistry()
                registry.register(RetrievalScorer())
                registry.register(LLMScorer())
                registry.register(ToolScorer(store=store))

            # Run through real scorers
            score_trace(trace, registry=registry)

            # Run through real blame algorithm
            blame_res = blame_trace(trace)

            predicted_name = blame_res.primary_step.name if blame_res.primary_step else None
            step_scores = {s.name: s.score for s in trace.steps}

            # Evaluate success
            if sc.ground_truth_step_name is not None:
                # Broken scenario: Top-1 step match
                passed = predicted_name == sc.ground_truth_step_name
            else:
                # Healthy scenario: Blame score must stay low (not falsely blamed)
                passed = blame_res.blame_score < HEALTHY_BLAME_THRESHOLD

            results.append(
                ScenarioResult(
                    scenario_id=sc.id,
                    category=sc.category,
                    description=sc.description,
                    ground_truth=sc.ground_truth_step_name,
                    predicted_step=predicted_name,
                    blame_score=round(blame_res.blame_score, 4),
                    confidence=blame_res.confidence,
                    explanation=blame_res.explanation,
                    passed=passed,
                    skipped=False,
                    step_scores=step_scores,
                )
            )

    elapsed_s = time.time() - start_time

    # Aggregate Metrics
    failure_results = [r for r in results if r.category != "healthy" and not r.skipped]
    healthy_results = [r for r in results if r.category == "healthy" and not r.skipped]

    failure_correct = sum(1 for r in failure_results if r.passed)
    failure_total = len(failure_results)
    top1_accuracy = failure_correct / failure_total if failure_total > 0 else 0.0

    healthy_false_positives = sum(1 for r in healthy_results if not r.passed)
    healthy_total = len(healthy_results)
    false_positive_rate = (
        healthy_false_positives / healthy_total if healthy_total > 0 else 0.0
    )

    # Per-category accuracy breakdown
    category_metrics: dict[str, dict[str, Any]] = {}
    all_categories = sorted({r.category for r in results if r.category != "healthy"})
    for cat in all_categories:
        cat_items = [r for r in results if r.category == cat and not r.skipped]
        c_passed = sum(1 for r in cat_items if r.passed)
        c_total = len(cat_items)
        c_acc = c_passed / c_total if c_total > 0 else 0.0
        category_metrics[cat] = {
            "correct": c_passed,
            "total": c_total,
            "accuracy": round(c_acc, 4),
        }

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_scenarios": len(results),
        "failure_scenarios_evaluated": failure_total,
        "failure_scenarios_passed": failure_correct,
        "top1_accuracy": round(top1_accuracy, 4),
        "top1_accuracy_pct": f"{top1_accuracy * 100:.1f}%",
        "healthy_scenarios_evaluated": healthy_total,
        "healthy_false_positives": healthy_false_positives,
        "false_positive_rate": round(false_positive_rate, 4),
        "false_positive_rate_pct": f"{false_positive_rate * 100:.1f}%",
        "category_breakdown": category_metrics,
        "elapsed_seconds": round(elapsed_s, 2),
        "results": [asdict(r) for r in results],
    }

    return summary


# -----------------------------------------------------------------------------
# Markdown Report Generator
# -----------------------------------------------------------------------------


def generate_markdown_report(summary: dict[str, Any]) -> str:
    """Generate clean paste-ready Markdown table from benchmark summary."""
    lines: list[str] = []
    lines.append("# Blame Accuracy Benchmark Results")
    lines.append("")
    lines.append(
        f"> **Summary:** **{summary['top1_accuracy_pct']} Top-1 Accuracy** "
        f"({summary['failure_scenarios_passed']}/{summary['failure_scenarios_evaluated']} failure scenarios correctly attributed) | "
        f"**{summary['healthy_false_positives']}/{summary['healthy_scenarios_evaluated']} False Positives** on healthy traces | "
        f"Execution time: **{summary['elapsed_seconds']}s**"
    )
    lines.append("")
    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Correct | Total | Accuracy |")
    lines.append("|----------|---------|-------|----------|")
    for cat, data in summary["category_breakdown"].items():
        lines.append(
            f"| `{cat}` | {data['correct']} | {data['total']} | **{data['accuracy'] * 100:.1f}%** |"
        )
    lines.append("")
    lines.append("## Scenario Details")
    lines.append("")
    lines.append(
        "| ID | Category | Ground Truth | Predicted | Blame Score | Status | Explanation |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|"
    )

    for r in summary["results"]:
        status_icon = "SKIPPED" if r["skipped"] else ("PASS" if r["passed"] else "FAIL")
        gt = r["ground_truth"] or "(healthy)"
        pred = r["predicted_step"] or "(none)"
        expl = r["explanation"].replace("\n", " ").replace("|", "\\|")
        if len(expl) > 75:
            expl = expl[:72] + "..."
        lines.append(
            f"| `{r['scenario_id']}` | `{r['category']}` | `{gt}` | `{pred}` | `{r['blame_score']:.2f}` | **{status_icon}** | {expl} |"
        )

    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main Execution Entry Point
# -----------------------------------------------------------------------------


def main() -> None:
    """Run benchmark, save results to JSON and Markdown artifacts."""
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "blame_accuracy_results.json"
    md_path = output_dir / "blame_accuracy_results.md"

    print("Running Traceback AI Blame Accuracy Benchmark (18 scenarios)...")
    summary = run_benchmark()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_report = generate_markdown_report(summary)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print("\n" + md_report)
    print(f"\nArtifacts saved to:\n  - {json_path}\n  - {md_path}")


if __name__ == "__main__":
    main()
