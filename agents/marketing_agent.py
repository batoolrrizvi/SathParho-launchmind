# agents/marketing_agent.py

import os
import sys
import json
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from message_bus import send_message, get_messages
from groq_client import call_llm

load_dotenv()

MARKETING_MODEL = "llama-3.3-70b-versatile"

# STATE
class MarketingState(TypedDict):
    startup_idea: str
    product_spec: dict
    pr_url: str
    tagline: str
    description: str
    cold_email_subject: str
    cold_email_body: str
    social_posts: dict
    email_sent: bool
    slack_posted: bool
    is_revision: bool
    revision_feedback: str


def parse_json_response(raw: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()
    return json.loads(clean)


# NODE 1 — Read inputs from message bus
def read_inputs(state: MarketingState) -> MarketingState:
    print("\n" + "="*50)
    print("📥 MARKETING AGENT — Reading inputs")
    print("="*50)

    inbox = get_messages("marketing")

    for msg in inbox:
        payload = msg["payload"]

        if msg["from_agent"] == "ceo":
            if msg["message_type"] == "revision_request":
                state["is_revision"] = True
                state["revision_feedback"] = payload.get("feedback", "")
                state["startup_idea"] = payload.get("idea", state.get("startup_idea", ""))
                state["pr_url"] = payload.get("pr_url", state.get("pr_url", ""))
                print(f"  ⚠️  Revision request from CEO: {state['revision_feedback']}")
            else:
                state["startup_idea"] = payload.get("idea", "")
                state["pr_url"] = payload.get("pr_url", "")
                print(f"  ✅ Task from CEO: {payload.get('focus', '')}")

        elif msg["from_agent"] == "product":
            if "product_spec" in payload:
                state["product_spec"] = payload["product_spec"]
                state["startup_idea"] = payload.get("startup_idea", state.get("startup_idea", ""))
            else:
                state["product_spec"] = payload
            print(f"  ✅ Product spec received from Product agent")

    return state


# NODE 2 — Generate all marketing copy with LLM
def generate_copy(state: MarketingState) -> MarketingState:
    print("\n" + "="*50)
    print("✍️  MARKETING AGENT — Generating copy with LLM")
    print(f"   Using model: {MARKETING_MODEL}")
    print("="*50)

    spec      = state.get("product_spec", {})
    idea      = state.get("startup_idea", "")
    val_prop  = spec.get("value_proposition", idea)
    features  = spec.get("features", [])
    personas  = spec.get("personas", [])

    features_text = "\n".join(
        [f"- {f['name']}: {f['description']}" for f in features]
    ) if features else "No features provided"

    personas_text = "\n".join(
        [f"- {p['name']} ({p['role']}): {p['pain_point']}" for p in personas]
    ) if personas else "No personas provided"

    revision_note = ""
    if state.get("is_revision"):
        revision_note = f"""
IMPORTANT REVISION: The CEO rejected the previous marketing copy with this feedback:
"{state.get('revision_feedback', '')}"
Address ALL feedback points specifically in this new version."""

    system = """You are a world-class growth marketer for a student-focused tech startup.
Generate marketing copy that is punchy, clear, and speaks directly to students.

Respond ONLY with a valid JSON object. No explanation, no markdown fences, just pure JSON.

Required format:
{
    "tagline": "Under 10 words. Bold, benefit-driven, no fluff.",
    "description": "2-3 sentences for the landing page. Clear, friendly, benefit-first.",
    "email_subject": "Compelling cold email subject line under 60 characters",
    "email_body": "Cold outreach email (HTML) addressed to a potential early user. Include a greeting, the pain point, how the product solves it, and a clear CTA. 3-4 short paragraphs.",
    "social_posts": {
        "twitter": "Under 280 chars. Hook + benefit + light CTA. No hashtag spam.",
        "linkedin": "2-3 sentences. Professional tone. Focus on the problem being solved.",
        "instagram": "Casual, visual language. 2 sentences + 3-5 relevant hashtags."
    }
}"""

    user = f"""Startup name: SathParho
Startup idea: {idea}
Value proposition: {val_prop}

Features:
{features_text}

Target users:
{personas_text}
{revision_note}

Generate compelling marketing copy now."""

    raw = call_llm(system, user, model=MARKETING_MODEL, max_tokens=2048)

    try:
        copy = parse_json_response(raw)
        state["tagline"]           = copy.get("tagline", "Study smarter. Together.")
        state["description"]       = copy.get("description", val_prop)
        state["cold_email_subject"] = copy.get("email_subject", f"Introducing {idea[:40]}")
        state["cold_email_body"]   = copy.get("email_body", f"<p>{val_prop}</p>")
        state["social_posts"]      = copy.get("social_posts", {})
        print(f"  ✅ Copy generated")
        print(f"     Tagline: {state['tagline']}")
        print(f"     Email subject: {state['cold_email_subject']}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ⚠️  JSON parse failed ({e}), using fallback copy")
        state["tagline"]            = "Find your study squad."
        state["description"]        = "SathParho connects university students with peers studying the same subjects. Post a group, find a group, study together."
        state["cold_email_subject"] = "Stop studying alone — SathParho is here"
        state["cold_email_body"]    = "<p>Hi there,</p><p>Studying alone is hard. SathParho helps you find study partners at your university in seconds.</p><p>Check it out and join a group today.</p><p>— The SathParho Team</p>"
        state["social_posts"]       = {
            "twitter":   "Tired of studying alone? SathParho connects you with students taking the same courses. Find your group today.",
            "linkedin":  "SathParho is solving academic isolation for university students by making it easy to form and discover study groups by subject and schedule.",
            "instagram": "Your study group is waiting 📚 Find peers, pick a time, and ace your courses together. #StudentLife #StudyGroup #SathParho"
        }

    return state


# NODE 3 — Send cold outreach email via SendGrid
def send_email(state: MarketingState) -> MarketingState:
    print("\n" + "="*50)
    print("📧 MARKETING AGENT — Sending cold outreach email")
    print("="*50)

    sendgrid_key  = os.environ.get("SENDGRID_API_KEY", "")
    from_email = os.environ.get("SENDGRID_FROM_EMAIL", "")
    to_email = from_email  

    if not sendgrid_key or not from_email:
        print("  ⚠️  SendGrid credentials not found — skipping email")
        state["email_sent"] = False
        return state

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        message = Mail(
            from_email=from_email,
            to_emails=to_email,
            subject=state["cold_email_subject"],
            html_content=state["cold_email_body"]
        )

        sg = SendGridAPIClient(sendgrid_key)
        response = sg.send(message)

        if response.status_code in (200, 202):
            state["email_sent"] = True
            print(f"  ✅ Email sent to {to_email} (status {response.status_code})")
        else:
            print(f"  ❌ Email failed with status {response.status_code}")
            state["email_sent"] = False

    except Exception as e:
        print(f"  ❌ SendGrid error: {e}")
        state["email_sent"] = False

    return state

# NODE 4 — Post launch message to Slack (Block Kit)
def post_to_slack(state: MarketingState) -> MarketingState:
    print("\n" + "="*50)
    print("💬 MARKETING AGENT — Posting to Slack #launches")
    print("="*50)

    slack_token = os.environ.get("SLACK_BOT_TOKEN", "")

    if not slack_token:
        print("  ⚠️  No Slack token — skipping Slack post")
        state["slack_posted"] = False
        return state

    pr_url      = state.get("pr_url", "#")
    tagline     = state.get("tagline", "")
    description = state.get("description", "")
    social      = state.get("social_posts", {})

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🚀 SathParho — Product Launch"}
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Tagline:* _{tagline}_"}
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": description}
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*GitHub PR:*\n<{pr_url}|View Pull Request>"},
                {"type": "mrkdwn", "text": f"*Email Status:*\n{'Sent ✅' if state.get('email_sent') else 'Skipped ⚠️'}"}
            ]
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Social Media Drafts:*\n"
                        + f"• *Twitter/X:* {social.get('twitter', 'N/A')}\n"
                        + f"• *LinkedIn:* {social.get('linkedin', 'N/A')}\n"
                        + f"• *Instagram:* {social.get('instagram', 'N/A')}"
            }
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "✅ *Marketing agent complete. Launch materials ready.*"}
        }
    ]

    try:
        import requests
        r = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {slack_token}"},
            json={"channel": "#launches", "blocks": blocks}
        )
        result = r.json()
        if result.get("ok"):
            state["slack_posted"] = True
            print("  ✅ Slack message posted to #launches")
        else:
            print(f"  ❌ Slack error: {result.get('error')}")
            state["slack_posted"] = False
    except Exception as e:
        print(f"  ❌ Slack post failed: {e}")
        state["slack_posted"] = False

    return state

# NODE 5 — Send all outputs back to CEO
def send_results(state: MarketingState) -> MarketingState:
    print("\n" + "="*50)
    print("📤 MARKETING AGENT — Sending results to CEO")
    print("="*50)

    send_message(
        from_agent="marketing",
        to_agent="ceo",
        message_type="result",
        payload={
            "tagline":            state.get("tagline", ""),
            "description":        state.get("description", ""),
            "email_subject":      state.get("cold_email_subject", ""),
            "email_sent":         state.get("email_sent", False),
            "slack_posted":       state.get("slack_posted", False),
            "social_posts":       state.get("social_posts", {}),
        }
    )

    print(f"  ✅ Results sent to CEO")
    print(f"     Tagline:      {state.get('tagline')}")
    print(f"     Email sent:   {state.get('email_sent')}")
    print(f"     Slack posted: {state.get('slack_posted')}")

    return state

# BUILD THE GRAPH
def build_marketing_graph():
    graph = StateGraph(MarketingState)

    graph.add_node("read_inputs",   read_inputs)
    graph.add_node("generate_copy", generate_copy)
    graph.add_node("send_email",    send_email)
    graph.add_node("post_to_slack", post_to_slack)
    graph.add_node("send_results",  send_results)

    graph.set_entry_point("read_inputs")
    graph.add_edge("read_inputs",   "generate_copy")
    graph.add_edge("generate_copy", "send_email")
    graph.add_edge("send_email",    "post_to_slack")
    graph.add_edge("post_to_slack", "send_results")
    graph.add_edge("send_results",  END)

    return graph.compile()

def run_marketing_agent():
    print("\n" + "="*60)
    print("📣  MARKETING AGENT — LaunchMind")
    print(f"   LLM Model: {MARKETING_MODEL}")
    print("="*60)

    app = build_marketing_graph()

    initial_state: MarketingState = {
        "startup_idea":        "",
        "product_spec":        {},
        "pr_url":              "",
        "tagline":             "",
        "description":         "",
        "cold_email_subject":  "",
        "cold_email_body":     "",
        "social_posts":        {},
        "email_sent":          False,
        "slack_posted":        False,
        "is_revision":         False,
        "revision_feedback":   ""
    }

    return app.invoke(initial_state)


if __name__ == "__main__":
    run_marketing_agent()
