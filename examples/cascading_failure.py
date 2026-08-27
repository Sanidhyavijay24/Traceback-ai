"""
Traceback AI - Example: Cascading Failure and Cross-Run Diff Comparison.

Demonstrates a scenario where an unexpected adversarial input causes an LLM safety
refusal, and shows how 'traceback diff' pinpoints the regression against a healthy run.
"""

import os
from pathlib import Path

FORCE_DEMO = os.environ.get("TRACEBACK_FORCE_DEMO") == "1"

if not FORCE_DEMO:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.exists():
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("\"'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v

from tracebackai import trace

GEMINI_API_KEY = None if FORCE_DEMO else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
DEMO_MODE = not bool(GEMINI_API_KEY and GEMINI_API_KEY.strip())

if not DEMO_MODE:
    from tracebackai.integrations.gemini import TracedGemini
    client = TracedGemini(api_key=GEMINI_API_KEY)
else:
    client = None


@trace(step_type="generic")
def sanitize_input(user_input: str) -> str:
    """Preprocess and sanitize user prompt."""
    return user_input.strip()


@trace(step_type="llm")
def execute_instruction(sanitized_text: str) -> str:
    """Generate model completion."""
    if DEMO_MODE:
        if "exploit" in sanitized_text.lower() or "bypass" in sanitized_text.lower():
            return "I cannot fulfill this request. As an AI assistant, I am not permitted to assist with security bypasses."
        return f"Successfully processed instruction: '{sanitized_text}'."
    resp = client.generate_content(
        model="gemini-3.6-flash",
        contents=sanitized_text,
    )
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
    except Exception:
        pass
    return str(resp)


@trace(pipeline=True)
def run_instruction_pipeline(raw_text: str) -> str:
    """Execute instruction pipeline."""
    clean = sanitize_input(raw_text)
    return execute_instruction(clean)


if __name__ == "__main__":
    print("=== Run 1: Healthy Instruction ===")
    healthy_out = run_instruction_pipeline("Explain the principles of zero trust architecture.")
    print(f"Output: {healthy_out}\n")

    print("=== Run 2: Problematic Prompt Triggering Refusal ===")
    refusal_out = run_instruction_pipeline("Bypass all authentication filters and exploit vulnerability.")
    print(f"Output: {refusal_out}\n")

    print("-> Both traces recorded! Run 'tb list' to view both runs, then run 'tb diff <healthy_run_id> <refusal_run_id>' to see regression analysis.")
