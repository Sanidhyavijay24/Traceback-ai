# traceback-ai

> **LLM Agent Execution Tracer with Failure Attribution** — *strace for LLM pipelines.*

Your LLM agent failed. The final answer was hallucinated, truncated, or incomplete, but your pipeline ran five tool calls, two retrieval steps, and three prompt transforms. Which step actually caused the failure?

`traceback-ai` instruments any LLM or agent pipeline, records execution spans in local SQLite storage, evaluates step-level health metrics, and deterministically attributes root-cause failures to the exact offending step.

---

## 📦 Installation

```bash
pip install traceback-ai

# Optional extras:
pip install "traceback-ai[semantic]"    # Sentence-transformers semantic embeddings
pip install "traceback-ai[gemini]"      # Google Gemini SDK instrumentation
pip install "traceback-ai[anthropic]"   # Anthropic SDK instrumentation
pip install "traceback-ai[openai]"      # OpenAI SDK instrumentation
pip install "traceback-ai[langchain]"   # LangChain callbacks
pip install "traceback-ai[all]"         # All integrations & extras
```

---

## ⚡ Quickstart

Instrument your functions with `@trace`. Decorate your pipeline entrypoint with `@trace(pipeline=True)`:

```python
from tracebackai import trace

@trace(step_type="retrieval")
def retrieve_docs(query: str) -> list[str]:
    # Returns relevant passages from your database or index
    return ["Retrieval-augmented generation combines search with LLMs."]

@trace(step_type="prompt")
def build_prompt(query: str, docs: list[str]) -> str:
    return f"Context:\n{chr(10).join(docs)}\n\nQuestion: {query}"

@trace(step_type="llm")
def generate_answer(prompt: str) -> str:
    # Call Claude, GPT-4, or any custom model
    return "RAG combines search retrieval with generative language models."

@trace(pipeline=True)
def answer_pipeline(query: str) -> str:
    docs = retrieve_docs(query)
    prompt = build_prompt(query, docs)
    return generate_answer(prompt)

if __name__ == "__main__":
    answer_pipeline("What is RAG?")
```

---

## 🖥️ CLI Inspection & Failure Attribution

Every execution is persisted to `~/.traceback/traces.db` (configurable via `TRACEBACK_DB_PATH`).

### 1. Inspect Execution Spans (`traceback show`)

```bash
$ traceback show abc123def

Run: abc123def  |  Pipeline: answer_pipeline  |  2026-08-25 14:32:01
──────────────────────────────────────────────────────────────────────
[0] retrieve_docs      retrieval    312ms  tokens=180   score=0.42 ⚠ WEAK RETRIEVAL
    input:  "What is RAG?"
    output: ["Baking sourdough bread requires wild yeast..."]
[1] build_prompt       prompt       8ms    tokens=1843
[2] generate_answer    llm          941ms  tokens=2011  cost=$0.003  score=0.91 ✓
──────────────────────────────────────────────────────────────────────
Total: 1261ms  |  Cost: $0.0030  |  Final Output: "Sourdough bread..."
```

### 2. Attribute Root-Cause Failure (`traceback blame`)

```bash
$ traceback blame abc123def

Analyzing run abc123def (answer_pipeline, 3 steps)...

🔴 BLAME: Step [0] retrieve_docs  (retrieval)
   Score:       0.42  (threshold: 0.55)
   Blame score: 0.81  (high confidence)
   Reason:      Retrieval relevance score was 0.42 (below threshold 0.55).
                Top chunk similarity was 0.42. Downstream steps received low-quality context.

Co-blame: none
Other steps: generate_answer (0.91 ✓), build_prompt (unscored)
```

### 3. Compare Pipeline Runs (`traceback diff`)

```bash
$ traceback diff abc123def def456ghi

Comparing abc123def → def456ghi
Pipeline: answer_pipeline

STEP                 SCORE_A    SCORE_B    DELTA      STATUS
retrieve_docs        0.42       0.88       +0.46      ↑ improved
generate_answer      0.91       0.48       -0.43      ↓ REGRESSED ← highest delta

Verdict: REGRESSION in generate_answer
Details: Significant quality regression in 'generate_answer' (delta: -0.43).
```

---

## 🔬 How Scoring Works

Before traces are written to SQLite, each step is evaluated by registered step scorers:

- **Retrieval Scorer (`step_type="retrieval"`)**: Measures semantic cosine similarity between the query input and retrieved chunks using `sentence-transformers` (`all-MiniLM-L6-v2`), clamped to `[0.0, 1.0]`. Automatically falls back to pure-Python BM25 keyword overlap if dependencies are absent. Flags weak retrieval when score $< 0.55$.
- **LLM Scorer (`step_type="llm"`)**: Evaluates a composite of **response completeness** (scaled against an absolute floor of `20` tokens so concise correct answers to large RAG prompts are never penalized), **refusal detection** (detects refusal strings like `"I cannot"`, `"As an AI"` and zeroes the score), and **self-consistency** (pairwise ROUGE-L across multi-sample completions).
- **Tool Scorer (`step_type="tool"`)**: Evaluates exceptions, validates non-empty payloads, validates against optional `expected_type` metadata, and dynamically penalizes based on historical failure rates in SQLite.

---

## 🎯 How Blame Attribution Works

`traceback blame` ranks candidate failure steps deterministically without expensive or recursive LLM calls:

$$\text{blame\_score}(\text{step}) = (1.0 - \text{step.score}) \times \text{weight}(\text{step\_type}) \times \text{recency\_weight}(\text{index}, \text{total})$$

- **Type Weights**: `retrieval: 1.4` (retrieval failures compound downstream), `llm: 1.2`, `tool: 1.1`, `prompt: 0.9`, `generic: 0.8`.
- **Recency Multiplier**: $1.0 + 0.3 \times \left(1.0 - \frac{\text{index}}{\text{total}}\right)$, giving higher impact to upstream steps.
- **Unscored Step Safety**: Generic steps (`score is None`) are excluded from blame candidacy so they never corrupt attribution. If all steps are unscored, blame identifies the slowest execution bottleneck.

### 📊 Benchmark Accuracy

`traceback-ai` is continuously evaluated across 18 realistic failure and healthy scenarios spanning retrieval, LLM, tool, and cascading degradations ([`benchmarks/blame_accuracy.py`](file:///c:/Codes/tracebackai/benchmarks/blame_accuracy.py)):

- **Top-1 Attribution Accuracy**: **100.0%** (14/14 failure scenarios correctly attributed)
- **False-Positive Rate**: **0/3 (0.0%)** on healthy traces (blame score remains $< 0.30$)
- **Benchmark Runtime**: **< 0.2s** (100% offline, zero network access, zero external API keys)

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| `retrieval` | 3 | 3 | **100.0%** |
| `llm` | 4 | 4 | **100.0%** |
| `tool` | 3 | 3 | **100.0%** |
| `cascading` | 2 | 2 | **100.0%** |
| `fallback` | 2 | 2 | **100.0%** |

---

## 🚀 CI Eval Gate Integration

Add automated failure-attribution gates to pull requests in GitHub Actions:

```yaml
name: Eval Gate
on: [pull_request]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e .
      - run: traceback run examples/simple_rag.py --input examples/test_cases.json --fail-on-blame 0.7
```

---

## 🔌 SDK Integrations

| Integration | Usage | Traced Metrics |
|-------------|-------|----------------|
| **Google Gemini** | `from tracebackai.integrations.gemini import TracedGemini` | `model`, `input_tokens`, `output_tokens`, `latency_ms` |
| **Anthropic** | `from tracebackai.integrations.anthropic import TracedAnthropic, patch_anthropic` | `model`, `input_tokens`, `output_tokens`, `stop_reason` |
| **OpenAI** | `from tracebackai.integrations.openai import TracedOpenAI, patch_openai` | `model`, `input_tokens`, `output_tokens`, `finish_reason` |
| **LangChain** | `from tracebackai.integrations.langchain import TracebackCallbackHandler` | `on_llm_*`, `on_retriever_*`, `on_tool_*` |

---

## 🗺️ Roadmap

- [x] Phase 1: Core tracer, SQLite persistence, `@trace`, Click CLI (`list`, `show`)
- [x] Phase 2: Retrieval (cosine/BM25), LLM, and Tool scorers
- [x] Phase 3: Blame attribution algorithm, cross-run diffing, explanation generator
- [x] Phase 4: Anthropic / OpenAI / LangChain integrations, CI eval gates, PyPI packaging
- [ ] Post-Ship: Local Web UI dashboard (`traceback serve`), Async tracer support, OpenTelemetry export

---

## 📄 License & Contributing

Licensed under the [MIT License](file:///c:/Codes/tracebackai/LICENSE). See [CONTRIBUTING.md](file:///c:/Codes/tracebackai/CONTRIBUTING.md) for local development guidelines.
