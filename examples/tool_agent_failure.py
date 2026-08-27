"""
Traceback AI - Example: Multi-Step Tool Agent with Tool Failure.

Demonstrates how Traceback AI diagnoses an agent pipeline where an external tool
raises an exception or returns an empty payload, causing downstream errors.
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
def parse_user_intent(command: str) -> dict:
    """Step 1: Parse user intent from prompt."""
    return {"action": "fetch_stock_quote", "ticker": "INVALID_OR_CRASHING_TICKER"}


@trace(step_type="tool")
def query_stock_market_api(ticker: str) -> dict:
    """
    Step 2: External stock market API tool.
    Simulates a tool failure (e.g. HTTP 500 / Network timeout / Empty payload).
    """
    # Simulate a tool error
    raise ConnectionResetError(f"Failed to connect to stock exchange server for ticker '{ticker}' (HTTP 502 Bad Gateway)")


@trace(step_type="llm")
def fallback_llm_handler(error_msg: str) -> str:
    """Step 3: Generate apology/fallback response to user."""
    if DEMO_MODE:
        return f"I apologize, but I encountered an error while fetching market data: {error_msg}."
    prompt = f"Explain politely to the user that market data lookup failed due to: {error_msg}"
    resp = client.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    if hasattr(resp, "text") and resp.text:
        return resp.text.strip()
    return str(resp)


@trace(pipeline=True)
def run_financial_agent(user_command: str) -> str:
    """Execute the financial agent pipeline."""
    intent = parse_user_intent(user_command)
    try:
        data = query_stock_market_api(intent["ticker"])
        return f"Current price for {intent['ticker']}: ${data.get('price', 0.0)}"
    except Exception as err:
        return fallback_llm_handler(str(err))


if __name__ == "__main__":
    cmd = "What is the latest stock price and trading volume for NVDA?"
    print(f"Running Financial Agent for: '{cmd}'")
    if DEMO_MODE:
        print("[demo mode: using simulated fallback response]\n")
    else:
        print("[live mode: calling Gemini API]\n")

    result = run_financial_agent(cmd)
    print(f"Agent Output:\n{result}\n")
    print("-> Trace recorded! Run 'tb list' or 'tb blame <run_id>' to see tool failure attribution.")
