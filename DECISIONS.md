# Technical Decisions & Architecture Tradeoffs Log

| Decision | Chosen | Rejected | Rationale |
|----------|--------|----------|-----------|
| **Trace Persistence** | SQLite (`~/.traceback/traces.db` or `TRACEBACK_DB_PATH`) | Postgres, Redis, MongoDB | Zero configuration for local development and CI; easily inspected or swapped via `TRACEBACK_DB_PATH`. Enforces relational integrity via `PRAGMA foreign_keys = ON;`. |
| **Retrieval Scorer** | `sentence-transformers` (`all-MiniLM-L6-v2`) with pure-Python BM25 fallback | OpenAI embeddings API | Zero external API costs, offline capable, and works seamlessly in closed air-gapped CI test runners. BM25 fallback guarantees zero required external dependencies. |
| **Token Counting** | `tiktoken` (`cl100k_base`) with character-length fallback | Heuristic character counting only | Matches OpenAI / modern LLM tokenizer byte-pair encodings accurately while falling back gracefully. |
| **Blame Attribution Algorithm** | Deterministic Weighted Heuristics ($(\text{1 - score}) \times \text{type\_weight} \times \text{recency\_weight}$) | LLM-as-a-Judge | Instant execution (< 50ms), deterministic reproducibility, zero circular dependency on another LLM, zero evaluation token costs. |
| **Unscored Step Isolation** | Exclude unscored steps from candidacy; fallback to latency bottleneck | Defaulting unscored steps to 0.0 or 1.0 | Unscored generic steps (e.g. string formatting) should not receive false penalties or obscure true root causes. |
| **Completeness Scorer Floor** | Absolute floor (`MIN_HEALTHY_OUTPUT_TOKENS = 20`) | Prompt-length relative scaling | Avoids penalizing concise, correct factual answers (e.g., 25 tokens) to large RAG context prompts (2000+ tokens). |
| **Packaging Backend** | `hatchling` via standard `pyproject.toml` | `setuptools` with `setup.py` | Modern PEP 517/518 build system with clean editable installs and zero legacy boilerplate. |
| **Zero-Secret Demo Mode** | Auto-branching demo mode in example RAG pipeline | Requiring mock environment variables | Allows CI eval gates and open-source contributors to run end-to-end evaluation pipelines with zero secrets configured. |
