# Blame Accuracy Benchmark Results

> **Summary:** **100.0% Top-1 Accuracy** (14/14 failure scenarios correctly attributed) | **0/3 False Positives** on healthy traces | Execution time: **0.14s**

## Category Breakdown

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| `cascading` | 2 | 2 | **100.0%** |
| `fallback` | 2 | 2 | **100.0%** |
| `llm` | 4 | 4 | **100.0%** |
| `retrieval` | 3 | 3 | **100.0%** |
| `tool` | 3 | 3 | **100.0%** |

## Scenario Details

| ID | Category | Ground Truth | Predicted | Blame Score | Status | Explanation |
|---|---|---|---|---|---|---|
| `retrieval_01_unrelated_chunks` | `retrieval` | `retrieve_docs` | `retrieve_docs` | `1.82` | **PASS** | Retrieved passages had low query similarity (0.00 < 0.55). Downstream st... |
| `retrieval_02_empty_chunks` | `retrieval` | `retrieve_docs` | `retrieve_docs` | `1.82` | **PASS** | Retrieval returned no document chunks. Downstream steps received empty c... |
| `retrieval_03_semantic_distractor` | `retrieval` | `retrieve_docs` | `(none)` | `0.00` | **SKIPPED** | SKIPPED: sentence-transformers not available; semantic distractor requir... |
| `retrieval_04_weak_retrieval_healthy_llm` | `retrieval` | `search_knowledge_base` | `search_knowledge_base` | `1.82` | **PASS** | Retrieved passages had low query similarity (0.00 < 0.55). Downstream st... |
| `llm_05_refusal_string` | `llm` | `generate_summary` | `generate_summary` | `1.56` | **PASS** | Model triggered a safety or policy refusal response instead of answering... |
| `llm_06_truncated_output` | `llm` | `generate_architecture_explanation` | `generate_architecture_explanation` | `0.34` | **PASS** | Model response was unusually short (11 tokens). May indicate truncation,... |
| `llm_07_generic_non_responsive` | `llm` | `generate_economic_analysis` | `generate_economic_analysis` | `0.44` | **PASS** | Model response was unusually short (7 tokens). May indicate truncation, ... |
| `llm_08_multisample_inconsistency` | `llm` | `generate_chemistry_answer` | `generate_chemistry_answer` | `0.16` | **PASS** | Model response was unusually short (18 tokens). May indicate truncation,... |
| `tool_09_raised_exception` | `tool` | `execute_sql_query` | `execute_sql_query` | `1.00` | **PASS** | Step raised an unhandled exception: OperationalError: Connection refused... |
| `tool_10_empty_payload` | `tool` | `fetch_user_metadata` | `fetch_user_metadata` | `1.14` | **PASS** | Tool returned null or empty output payload. |
| `tool_11_high_historical_error_rate` | `tool` | `unreliable_weather_api` | `unreliable_weather_api` | `1.14` | **PASS** | Tool output health was low (0.20). This tool has a high historical failu... |
| `cascading_12_multi_degraded_steps` | `cascading` | `llm_synthesizer` | `llm_synthesizer` | `1.56` | **PASS** | Model triggered a safety or policy refusal response instead of answering... |
| `cascading_13_upstream_root_cause` | `cascading` | `retrieve_kb_context` | `retrieve_kb_context` | `1.82` | **PASS** | Retrieved passages had low query similarity (0.00 < 0.55). Downstream st... |
| `fallback_14_all_unscored_slowest` | `fallback` | `parse_json_payload` | `parse_json_payload` | `0.50` | **PASS** | All steps in this run were unscored. Blame fell back to step 'parse_json... |
| `fallback_15_all_unscored_near_tie` | `fallback` | `fetch_remote_payload` | `fetch_remote_payload` | `0.50` | **PASS** | All steps in this run were unscored. Blame fell back to step 'fetch_remo... |
| `healthy_16_clean_rag` | `healthy` | `(healthy)` | `generate_rag_answer` | `0.10` | **PASS** | LLM response quality sub-scores degraded (overall health score: 0.93). |
| `healthy_17_clean_tool` | `healthy` | `(healthy)` | `format_tax_summary` | `0.10` | **PASS** | LLM response quality sub-scores degraded (overall health score: 0.93). |
| `healthy_18_clean_multistep_agent` | `healthy` | `(healthy)` | `generate_customer_reply` | `0.10` | **PASS** | LLM response quality sub-scores degraded (overall health score: 0.93). |
