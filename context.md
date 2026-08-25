# Traceback AI — Context & Single Source of Truth

## 1. Project Overview
- **Name:** `traceback-ai` (`tracebackai`)
- **Description:** LLM Agent Execution Tracer with Failure Attribution ("strace for LLM agents").
- **Mission:** Instrument any LLM/agent pipeline, record structured execution traces in SQLite, score steps, and attribute root cause failure to the most likely causative step.

---

## 2. Tech Stack
- **Language:** Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- **Trace Persistence:** SQLite (default `~/.traceback/traces.db`, configurable via `TRACEBACK_DB_PATH`)
- **CLI:** Click (`traceback` and `tb` entry points)
- **Token Counting:** `tiktoken` (`cl100k_base`) with character-length fallback
- **Semantic Scoring:** `sentence-transformers` (optional `all-MiniLM-L6-v2`) with pure-Python BM25 term-overlap fallback
- **Integrations:** Google Gemini (`TracedGemini`), Anthropic (`TracedAnthropic`, `patch_anthropic`), OpenAI (`TracedOpenAI`, `patch_openai`), LangChain (`TracebackCallbackHandler`)
- **Environment:** Configured via `.env` (supports `GEMINI_API_KEY`, `GOOGLE_API_KEY`)
- **Packaging:** `hatchling` / `pyproject.toml`
- **Testing:** `pytest`, `pytest-cov`

---

## 3. Architecture & Directory Structure
```
traceback-ai/
├── src/
│   └── tracebackai/
│       ├── __init__.py        # Public API exports
│       ├── models.py          # Data models: Step, Trace
│       ├── store.py           # SQLite persistence layer with safe recursive JSON serialization & error rate metrics
│       ├── token_utils.py     # Token counter with fallback
│       ├── tracer.py          # @trace decorator, TraceContext context manager, nested span handling, pre-save scoring
│       ├── scoring.py         # score_trace() mutator applying scorers before persistence
│       ├── scorers/           # BaseScorer & step-specific scorers
│       │   ├── __init__.py    # ScorerRegistry mapping step_type to scorers
│       │   ├── base.py        # BaseScorer ABC (can_score, score)
│       │   ├── retrieval.py   # RetrievalScorer (cosine similarity & BM25 keyword overlap fallback)
│       │   ├── llm.py         # LLMScorer (absolute-floor completeness, refusal detection, multi-sample consistency)
│       │   └── tool.py        # ToolScorer (type verification, emptiness checks, historical error rate penalty)
│       ├── blame.py           # Blame attribution & cross-run diffing (heuristic weighting, co-blame, explanations)
│       ├── cli.py             # Click CLI commands: list, show, blame, diff, run
│       └── integrations/      # Thin SDK wrappers: Gemini, Anthropic, OpenAI, LangChain
│           ├── __init__.py    # Lazy exports for optional integrations
│           ├── gemini.py      # TracedGemini wrapper supporting google-genai and google.generativeai
│           ├── anthropic.py   # TracedAnthropic wrapper & patch_anthropic
│           ├── openai.py      # TracedOpenAI wrapper & patch_openai
│           └── langchain.py   # TracebackCallbackHandler
├── tests/
│   ├── test_models.py         # Model dataclass tests
│   ├── test_store.py          # SQLite round-trip and serialization tests (resilient numpy test)
│   ├── test_tracer.py         # @trace, TraceContext, nesting, and error handling tests
│   ├── test_token_utils.py    # Token counter and fallback tests
│   ├── test_scorers.py        # Unit & integration tests for all step scorers and pre-save scoring
│   ├── test_blame.py          # Single-run blame attribution, cross-run diffs, and CLI tests
│   ├── test_integrations.py   # SDK integration mocks (Gemini, Anthropic, OpenAI, LangChain) and CLI run eval gate tests
│   ├── test_examples.py       # Zero-secret demo execution tests
│   └── test_cli.py            # Click CLI runner tests for list, show, blame, diff
├── examples/
│   ├── simple_rag.py          # Standalone RAG pipeline using Google Gemini with automatic .env loading
│   └── test_cases.json        # Evaluation test cases
├── .github/workflows/
│   ├── ci.yml                 # GitHub Actions CI workflow across Python versions (3.10, 3.11, 3.12)
│   ├── eval_gate.yml          # Pull request CI evaluation gate workflow
│   └── publish.yml            # PyPI distribution publishing workflow
├── pyproject.toml             # Hatchling build configuration and entry points (with dev numpy and gemini extras)
├── requirements.txt           # Project dependencies
├── .env.example               # Environment variables example template
├── .env                       # Local environment file (gitignored)
├── .gitignore                 # Git ignore rules (ignores .env and traceback_build_plan.md)
├── LICENSE                    # MIT License
├── DECISIONS.md               # Technical decisions and tradeoffs log
├── CONTRIBUTING.md            # Open-source contribution guidelines
├── README.md                  # Comprehensive developer documentation (with Gemini docs and install extras)
└── context.md                 # Single source of truth ledger
```

---

## 4. Feature Status Checklist
- [x] **Phase 1: Core Tracer**
- [x] **Phase 2: Step Scorers**
- [x] **Phase 3: Blame Algorithm**
- [x] **Phase 4: Integrations, Packaging & CI**
- [x] **CI & Dependencies Verified**:
  - `numpy>=1.24.0` added to `dev` in `pyproject.toml` and `requirements.txt`.
  - `tests/test_store.py` safeguarded with `pytest.importorskip("numpy")`.
  - `gemini = ["google-genai>=0.1.0"]` extra added to `pyproject.toml` and included in `all`.
  - `README.md` updated with `traceback-ai[gemini]` install and SDK integrations table entry.
