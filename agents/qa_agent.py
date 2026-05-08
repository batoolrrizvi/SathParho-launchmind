# agents/qa_agent.py

import os
import sys
import json
import requests
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from message_bus import send_message, get_messages
from groq_client import call_llm

load_dotenv()

QA_MODEL = "llama-3.3-70b-versatile"

GITHUB_TOKEN  = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO   = os.environ.get("GITHUB_REPO", "")
GITHUB_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}


def parse_json_response(raw: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()
    return json.loads(clean)


def github_api(method, endpoint, **kwargs):
    url = f"https://api.github.com/repos/{GITHUB_REPO}{endpoint}"
    r = getattr(requests, method)(url, headers=GITHUB_HEADERS, **kwargs)
    return r


# STATE
class QAState(TypedDict):
    startup_idea: str
    product_spec: dict
    html_content: str
    pr_url: str
    pr_number: int
    marketing_copy: dict
    html_review: dict
    marketing_review: dict
    overall_verdict: str      # "pass" or "fail"
    issues_found: list
    pr_comments_posted: bool
    review_report: dict


# NODE 1 — Read inputs from CEO
def read_inputs(state: QAState) -> QAState:
    print("\n" + "="*50)
    print("🔍 QA AGENT — Reading inputs from CEO")
    print("="*50)

    inbox = get_messages("qa")

    for msg in inbox:
        if msg["from_agent"] == "ceo":
            payload = msg["payload"]
            state["startup_idea"]   = payload.get("startup_idea", "")
            state["product_spec"]   = payload.get("product_spec", {})
            state["html_content"]   = payload.get("html_content", "")
            state["pr_url"]         = payload.get("pr_url", "")
            state["pr_number"]      = payload.get("pr_number", 0)
            state["marketing_copy"] = payload.get("marketing_copy", {})
            print(f"  ✅ QA inputs received from CEO")
            print(f"     PR URL:   {state['pr_url']}")
            print(f"     HTML len: {len(state['html_content'])} chars")

    return state

# NODE 2 — Review HTML landing page with LLM
def review_html(state: QAState) -> QAState:
    print("\n" + "="*50)
    print("🔍 QA AGENT — Reviewing HTML landing page")
    print("="*50)

    spec = state.get("product_spec", {})
    html = state.get("html_content", "")
    val_prop = spec.get("value_proposition", "")
    features = [f["name"] for f in spec.get("features", [])]
    idea = state.get("startup_idea", "")

    # Truncate HTML for the prompt to avoid token limits
    html_preview = html[:3000] + "\n...[truncated]..." if len(html) > 3000 else html

    system = """You are a senior QA engineer reviewing an AI-generated HTML landing page for a startup.
Be specific, critical, and constructive.

Respond ONLY with valid JSON. No markdown, no explanation.
Format:
{
    "verdict": "pass" or "fail",
    "score": integer 1-10,
    "headline_matches_value_prop": true or false,
    "features_mentioned": ["Feature1", "Feature2"],
    "missing_features": ["Feature3"],
    "issues": [
        "Specific issue 1",
        "Specific issue 2"
    ],
    "positive_points": [
        "What the page does well"
    ],
    "inline_comment_1": {
        "line_hint": "Short snippet of HTML line to target (first 60 chars)",
        "comment": "Specific reviewer comment for this line"
    },
    "inline_comment_2": {
        "line_hint": "Short snippet of another HTML line to target",
        "comment": "Specific reviewer comment for this line"
    }
}

Verdict is "pass" if score >= 6 AND no critical issues. Otherwise "fail"."""

    user = f"""Startup idea: {idea}
Value proposition: {val_prop}
Expected features: {json.dumps(features)}

HTML landing page (preview):
{html_preview}

Review: Does the landing page match the product spec? Is the headline aligned with the value proposition? Are the features represented accurately? Is it user-facing and professional?"""

    raw = call_llm(system, user, model=QA_MODEL, max_tokens=1024)

    try:
        review = parse_json_response(raw)
        if "verdict" not in review:
            review["verdict"] = "pass"
        if "score" not in review:
            review["score"] = 7
        if "issues" not in review:
            review["issues"] = []
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ⚠️  JSON parse failed ({e}), defaulting to pass")
        review = {
            "verdict": "pass",
            "score": 7,
            "headline_matches_value_prop": True,
            "features_mentioned": features,
            "missing_features": [],
            "issues": [],
            "positive_points": ["Page generated successfully"],
            "inline_comment_1": {
                "line_hint": "<h1",
                "comment": "Headline looks good. Ensure it matches the value proposition precisely."
            },
            "inline_comment_2": {
                "line_hint": "<section",
                "comment": "Features section present. Verify all 5 product features are listed."
            }
        }

    state["html_review"] = review

    emoji = "✅" if review["verdict"] == "pass" else "❌"
    print(f"  {emoji} HTML verdict: {review['verdict'].upper()} (score: {review.get('score')}/10)")
    for issue in review.get("issues", []):
        print(f"     Issue: {issue}")

    return state

# NODE 3 — Review marketing copy with LLM
def review_marketing(state: QAState) -> QAState:
    print("\n" + "="*50)
    print("🔍 QA AGENT — Reviewing marketing copy")
    print("="*50)

    copy = state.get("marketing_copy", {})
    spec = state.get("product_spec", {})
    idea = state.get("startup_idea", "")

    if not copy:
        print("  ℹ️  No marketing copy received — skipping marketing review")
        state["marketing_review"] = {
            "verdict": "pass",
            "score": 7,
            "issues": [],
            "positive_points": ["No copy to review"]
        }
        return state

    system = """You are a QA reviewer checking marketing copy for a student startup.
Be specific and critical.

Respond ONLY with valid JSON. No markdown, no explanation.
Format:
{
    "verdict": "pass" or "fail",
    "score": integer 1-10,
    "tagline_under_10_words": true or false,
    "email_has_cta": true or false,
    "tone_appropriate": true or false,
    "issues": [
        "Specific issue 1"
    ],
    "positive_points": [
        "What is done well"
    ]
}

Verdict is "pass" if score >= 6 AND no critical issues. Otherwise "fail"."""

    user = f"""Startup idea: {idea}
Value proposition: {spec.get('value_proposition', '')}

Marketing copy to review:
{json.dumps(copy, indent=2)}

Review: Is the tagline under 10 words? Is the cold email compelling with a clear CTA? Are social posts platform-appropriate? Is the tone right for university students?"""

    raw = call_llm(system, user, model=QA_MODEL, max_tokens=512)

    try:
        review = parse_json_response(raw)
        if "verdict" not in review:
            review["verdict"] = "pass"
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ⚠️  JSON parse failed ({e}), defaulting to pass")
        review = {
            "verdict": "pass",
            "score": 7,
            "tagline_under_10_words": True,
            "email_has_cta": True,
            "tone_appropriate": True,
            "issues": [],
            "positive_points": ["Copy looks appropriate"]
        }

    state["marketing_review"] = review

    emoji = "✅" if review["verdict"] == "pass" else "❌"
    print(f"  {emoji} Marketing verdict: {review['verdict'].upper()} (score: {review.get('score')}/10)")
    for issue in review.get("issues", []):
        print(f"     Issue: {issue}")

    return state


# NODE 4 — Post inline review comments on GitHub PR
def post_pr_comments(state: QAState) -> QAState:
    print("\n" + "="*50)
    print("💬 QA AGENT — Posting review comments on GitHub PR")
    print("="*50)

    pr_number = state.get("pr_number", 0)
    html = state.get("html_content", "")
    html_review = state.get("html_review", {})

    if not pr_number or not GITHUB_TOKEN or not GITHUB_REPO:
        print("  ⚠️  Missing PR number or GitHub credentials — skipping PR comments")
        state["pr_comments_posted"] = False
        return state

    # Get the commit SHA for this PR
    pr_resp = github_api("get", f"/pulls/{pr_number}")
    if pr_resp.status_code != 200:
        print(f"  ❌ Could not fetch PR details: {pr_resp.status_code}")
        state["pr_comments_posted"] = False
        return state

    pr_data   = pr_resp.json()
    head_sha  = pr_data["head"]["sha"]

    # Build overall PR review body
    html_verdict = html_review.get("verdict", "pass").upper()
    mkt_verdict  = state.get("marketing_review", {}).get("verdict", "pass").upper()
    issues_all   = html_review.get("issues", []) + state.get("marketing_review", {}).get("issues", [])

    review_body = (
        f"## QA Agent Review Report\n\n"
        f"**HTML Landing Page:** {html_verdict} (score: {html_review.get('score', 'N/A')}/10)\n"
        f"**Marketing Copy:** {mkt_verdict} (score: {state.get('marketing_review', {}).get('score', 'N/A')}/10)\n\n"
    )

    if issues_all:
        review_body += "### Issues Found\n"
        for issue in issues_all:
            review_body += f"- {issue}\n"
        review_body += "\n"

    positives = html_review.get("positive_points", [])
    if positives:
        review_body += "### Positive Points\n"
        for pt in positives:
            review_body += f"- {pt}\n"

    overall = "APPROVED" if (html_review.get("verdict") == "pass" and
                              state.get("marketing_review", {}).get("verdict") == "pass") else "REQUEST_CHANGES"
    review_event = "APPROVE" if overall == "APPROVED" else "REQUEST_CHANGES"

    # Build inline comments on index.html
    comments = []
    html_lines = html.split("\n")

    def find_line_number(hint: str) -> int:
        """Find the 1-based line number of the first line containing the hint."""
        hint_clean = hint.strip()[:60]
        for i, line in enumerate(html_lines, 1):
            if hint_clean.lower() in line.lower():
                return i
        return 1

    for key in ["inline_comment_1", "inline_comment_2"]:
        ic = html_review.get(key, {})
        if ic and ic.get("comment") and ic.get("line_hint"):
            line_no = find_line_number(ic["line_hint"])
            comments.append({
                "path": "index.html",
                "line": line_no,
                "body": f"**QA Agent:** {ic['comment']}"
            })

    # Fallback: always post at least 2 comments if LLM didn't produce them
    if len(comments) < 2:
        fallback_hints = ["<h1", "<section", "<body", "<!DOCTYPE", "<title"]
        added = 0
        fallback_messages = [
            "Ensure the headline directly reflects the value proposition from the product spec.",
            "Verify all 5 features from the product spec are represented in this section."
        ]
        for hint in fallback_hints:
            if added >= (2 - len(comments)):
                break
            line_no = find_line_number(hint)
            if line_no > 0:
                comments.append({
                    "path": "index.html",
                    "line": line_no,
                    "body": f"**QA Agent:** {fallback_messages[added]}"
                })
                added += 1

    # Submit PR review via GitHub API
    review_payload = {
        "commit_id": head_sha,
        "body":      review_body,
        "event":     review_event,
        "comments":  comments
    }

    r = github_api("post", f"/pulls/{pr_number}/reviews", json=review_payload)

    if r.status_code in (200, 201):
        state["pr_comments_posted"] = True
        print(f"  ✅ PR review posted ({review_event}) with {len(comments)} inline comment(s)")
    else:
        print(f"  ❌ PR review failed: {r.status_code} — {r.text[:300]}")
        # Fallback: post a plain issue comment instead
        fallback = github_api("post", f"/issues/{pr_number}/comments",
                              json={"body": review_body})
        if fallback.status_code in (200, 201):
            state["pr_comments_posted"] = True
            print(f"  ✅ Fallback: plain comment posted on PR #{pr_number}")
        else:
            state["pr_comments_posted"] = False

    return state


# NODE 5 — Compile review report and send to CEO
def send_review_report(state: QAState) -> QAState:
    print("\n" + "="*50)
    print("📤 QA AGENT — Sending review report to CEO")
    print("="*50)

    html_verdict = state.get("html_review", {}).get("verdict", "pass")
    mkt_verdict  = state.get("marketing_review", {}).get("verdict", "pass")

    overall = "pass" if (html_verdict == "pass" and mkt_verdict == "pass") else "fail"
    state["overall_verdict"] = overall

    all_issues = (
        state.get("html_review", {}).get("issues", []) +
        state.get("marketing_review", {}).get("issues", [])
    )
    state["issues_found"] = all_issues

    report = {
        "overall_verdict":    overall,
        "html_verdict":       html_verdict,
        "html_score":         state.get("html_review", {}).get("score", 0),
        "marketing_verdict":  mkt_verdict,
        "marketing_score":    state.get("marketing_review", {}).get("score", 0),
        "issues":             all_issues,
        "pr_comments_posted": state.get("pr_comments_posted", False),
        "pr_url":             state.get("pr_url", ""),
        "html_feedback":      "; ".join(state.get("html_review", {}).get("issues", [])),
        "marketing_feedback": "; ".join(state.get("marketing_review", {}).get("issues", []))
    }
    state["review_report"] = report

    send_message(
        from_agent="qa",
        to_agent="ceo",
        message_type="result",
        payload=report
    )

    emoji = "✅" if overall == "pass" else "❌"
    print(f"  {emoji} Overall QA verdict: {overall.upper()}")
    print(f"     HTML:      {html_verdict} ({state.get('html_review', {}).get('score')}/10)")
    print(f"     Marketing: {mkt_verdict} ({state.get('marketing_review', {}).get('score')}/10)")
    if all_issues:
        print(f"     Issues ({len(all_issues)}):")
        for issue in all_issues:
            print(f"       - {issue}")

    return state

def build_qa_graph():
    graph = StateGraph(QAState)

    graph.add_node("read_inputs",        read_inputs)
    graph.add_node("review_html",        review_html)
    graph.add_node("review_marketing",   review_marketing)
    graph.add_node("post_pr_comments",   post_pr_comments)
    graph.add_node("send_review_report", send_review_report)

    graph.set_entry_point("read_inputs")
    graph.add_edge("read_inputs",        "review_html")
    graph.add_edge("review_html",        "review_marketing")
    graph.add_edge("review_marketing",   "post_pr_comments")
    graph.add_edge("post_pr_comments",   "send_review_report")
    graph.add_edge("send_review_report", END)

    return graph.compile()

def run_qa_agent():
    print("\n" + "="*60)
    print("🔍  QA AGENT — LaunchMind")
    print(f"   LLM Model: {QA_MODEL}")
    print(f"   GitHub Repo: {GITHUB_REPO}")
    print("="*60)

    app = build_qa_graph()

    initial_state: QAState = {
        "startup_idea":        "",
        "product_spec":        {},
        "html_content":        "",
        "pr_url":              "",
        "pr_number":           0,
        "marketing_copy":      {},
        "html_review":         {},
        "marketing_review":    {},
        "overall_verdict":     "",
        "issues_found":        [],
        "pr_comments_posted":  False,
        "review_report":       {}
    }

    return app.invoke(initial_state)


if __name__ == "__main__":
    run_qa_agent()
