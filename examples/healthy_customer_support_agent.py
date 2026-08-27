"""
Traceback AI - Example: Fully Healthy Customer Support Agent.

Demonstrates a multi-step agent workflow where every step (tool, retrieval, LLM synthesis)
executes flawlessly with high quality scores and zero false-positive blame.
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


@trace(step_type="tool")
def lookup_customer_account(customer_id: str) -> dict:
    """Step 1 (Tool): Query customer account status."""
    return {
        "customer_id": customer_id,
        "name": "Sarah Jenkins",
        "plan": "Enterprise Tier",
        "billing_cycle": "Annual",
        "status": "Active",
    }


@trace(step_type="retrieval")
def search_support_policies(query: str) -> list[str]:
    """Step 2 (Retrieval): Retrieve company policy documents."""
    return [
        "Enterprise Tier customers are eligible for 100% prorated refunds within 30 days of billing.",
        "To process an Enterprise refund, verify active account status and submit request to accounts.",
        "Enterprise accounts receive priority refund processing with direct credit within 3 business days.",
    ]


@trace(step_type="prompt")
def prepare_email_prompt(customer: dict, policies: list[str], question: str) -> str:
    """Step 3 (Prompt): Assemble context into structured prompt."""
    policy_str = "\n".join(policies)
    return (
        f"Customer Account Details:\n{customer}\n\n"
        f"Company Refund Policies:\n{policy_str}\n\n"
        f"Customer Inquiry: {question}\n\n"
        f"Draft a polite, professional support response confirming their refund eligibility."
    )


@trace(step_type="llm")
def draft_support_email(prompt: str) -> str:
    """Step 4 (LLM): Generate final response to customer."""
    if DEMO_MODE:
        return (
            "Dear Sarah,\n\n"
            "Thank you for contacting Enterprise Support. As an active Enterprise Tier customer, "
            "you are eligible for a 100% prorated refund under our 30-day billing policy. We have "
            "initiated the refund process, and the credit will appear on your original payment method "
            "within 3 business days.\n\n"
            "Best regards,\nEnterprise Support Team"
        )
    resp = client.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    if hasattr(resp, "text") and resp.text:
        return resp.text.strip()
    return str(resp)


@trace(pipeline=True)
def run_support_agent(cust_id: str, inquiry: str) -> str:
    """Run the full support agent pipeline."""
    account = lookup_customer_account(cust_id)
    policies = search_support_policies(inquiry)
    prompt = prepare_email_prompt(account, policies, inquiry)
    return draft_support_email(prompt)


if __name__ == "__main__":
    cid = "cust_ent_8841"
    inquiry_text = "Can I get a prorated refund on my Enterprise annual billing?"
    print(f"Running Support Agent for customer {cid}: '{inquiry_text}'")
    if DEMO_MODE:
        print("[demo mode: using simulated response]\n")
    else:
        print("[live mode: calling Gemini API]\n")

    email = run_support_agent(cid, inquiry_text)
    print(f"Generated Email:\n{email}\n")
    print("-> Trace recorded! Run 'tb list' or 'tb show <run_id>' to see healthy trace execution.")
