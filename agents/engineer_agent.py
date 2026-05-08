# agents/engineer_agent.py

import os
import sys
import json
import base64
import requests
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from message_bus import send_message, get_messages
from groq_client import call_llm

load_dotenv()

ENGINEER_MODEL = "qwen/qwen3-32b"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")
GITHUB_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

BRANCH_NAME = "agent-landing-page"

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

class EngineerState(TypedDict):
    startup_idea: str
    product_spec: dict
    html_content: str
    issue_url: str
    pr_url: str
    pr_number: int 
    branch_name: str
    status: str
    is_revision: bool
    revision_feedback: str

# NODE 1 — Read inputs from message bus
def read_inputs(state: EngineerState) -> EngineerState:
    print("\n" + "="*50)
    print(" ENGINEER AGENT — Reading inputs")
    print("="*50)

    inbox = get_messages("engineer")

    for msg in inbox:
        payload = msg["payload"]

        if msg["from_agent"] == "ceo":
            if msg["message_type"] == "revision_request":
                state["is_revision"] = True
                state["revision_feedback"] = payload.get("feedback", "")
                state["startup_idea"] = payload.get("idea", state.get("startup_idea", ""))
                state["product_spec"] = payload.get("product_spec", state.get("product_spec", {}))
                print(f"   Revision request from CEO: {state['revision_feedback']}")
            else:
                state["startup_idea"] = payload.get("idea", "")
                print(f"  ✅ Task from CEO: {payload.get('focus', '')}")

        elif msg["from_agent"] == "product":
            if "product_spec" in payload:
                state["product_spec"] = payload["product_spec"]
                state["startup_idea"] = payload.get("startup_idea", state.get("startup_idea", ""))
            else:
                state["product_spec"] = payload
            print(f"  ✅ Product spec received from Product agent")

    state["branch_name"] = BRANCH_NAME
    return state


# NODE 2 — Generate HTML landing page using LLM
def generate_html(state: EngineerState) -> EngineerState:
    print("\n" + "="*50)
    print(" ENGINEER AGENT — Generating HTML landing page")
    print(f"   Using model: {ENGINEER_MODEL}")
    print("="*50)

    spec = state.get("product_spec", {})
    idea = state.get("startup_idea", "")

    features_text = ""
    if spec.get("features"):
        features_text = "\n".join(
            [f"- {f['name']}: {f['description']}" for f in spec["features"]]
        )

    personas_text = ""
    if spec.get("personas"):
        personas_text = "\n".join(
            [f"- {p['name']} ({p['role']}): {p['pain_point']}" for p in spec["personas"]]
        )

    stories_text = ""
    if spec.get("user_stories"):
        stories_text = "\n".join([f"- {s}" for s in spec["user_stories"]])

    revision_note = ""
    if state.get("is_revision"):
        revision_note = f"""
IMPORTANT REVISION: The CEO rejected the previous landing page with this feedback:
"{state.get('revision_feedback', '')}"
Address ALL feedback points in this new version."""

    system = """You are a senior frontend engineer. Generate a complete, production-quality HTML landing page for a STARTUP aimed at END USERS — not developers.

The page MUST include ONLY these sections:
1. Hero section: short catchy headline (max 8 words), one-sentence subheadline, and a CTA button
2. Features section: list the provided features with a short title and 1-sentence description each. Use simple SVG icons or Unicode symbols (like ★ ● ◆ →) — do NOT use emojis
3. "Who is this for?" section: brief cards for each persona (name, role, one-line pain point)
4. A final call-to-action section with a button
5. A minimal footer with copyright year and startup name

STRICTLY FORBIDDEN — do NOT include any of these:
- NO code snippets, database schemas, API examples, algorithm pseudocode, or technical implementation details
- NO emojis anywhere in the page (no 📚 🎓 💡 etc.)
- NO "How it works" steps that describe backend/technical processes
- NO developer documentation of any kind
- This is a MARKETING landing page, not a technical README

Design requirements:
- Complete HTML5 document with <!DOCTYPE html>
- All CSS in a <style> tag in <head>
- Color palette: use a modern blue-purple gradient primary (#4F46E5 to #7C3AED), white backgrounds, dark gray text (#1F2937), light gray accents (#F3F4F6)
- Google Fonts: Inter for body, Poppins for headings (via CDN link)
- Clean whitespace, subtle box shadows, rounded corners (8-12px)
- Responsive with media queries for mobile
- NO external CSS frameworks, NO JavaScript
- Keep it clean and minimal — white space is good

Output ONLY the raw HTML code. No markdown fences, no explanation, no thinking. Start directly with <!DOCTYPE html>."""

    user = f"""Startup name: SathParho
Startup description: {idea}

Value Proposition: {spec.get('value_proposition', idea)}

Features (show these as user-facing benefits, NOT technical specs):
{features_text}

Target users:
{personas_text}

REMINDER: This is a marketing landing page for students. Do NOT include any code, schemas, algorithms, or technical details. No emojis. Start with <!DOCTYPE html>.
{revision_note}"""

    raw = call_llm(system, user, model=ENGINEER_MODEL, max_tokens=4096)

    # Clean markdown fences if present
    html = raw.strip()
    if html.startswith("```html"):
        html = html[7:]
    elif html.startswith("```"):
        html = html[3:]
    if html.endswith("```"):
        html = html[:-3]
    html = html.strip()

    # Basic validation
    if not html.lower().startswith("<!doctype") and not html.lower().startswith("<html"):
        if "<" in html:
            html = "<!DOCTYPE html>\n<html lang=\"en\">\n<head><meta charset=\"UTF-8\"><title>LaunchMind</title></head>\n<body>\n" + html + "\n</body>\n</html>"

    state["html_content"] = html
    print(f"  ✅ HTML landing page generated ({len(html)} characters)")

    return state


# NODE 3 — Create GitHub issue
def create_github_issue(state: EngineerState) -> EngineerState:
    print("\n" + "="*50)
    print(" ENGINEER AGENT — Creating GitHub issue")
    print("="*50)

    spec = state.get("product_spec", {})
    idea = state.get("startup_idea", "")

    # Generate issue description with LLM
    system = """You are a software engineer writing a GitHub issue. Write a clear, concise issue description.
Output ONLY the issue body text with markdown formatting. No fences wrapping the output."""

    features_list = ", ".join([f['name'] for f in spec.get('features', [])])
    user = f"""Write a GitHub issue body for:
Title: Initial landing page
Startup: {idea}
Value proposition: {spec.get('value_proposition', '')}
Features to include: {features_list}

Describe what the landing page should contain and why it is needed."""

    issue_body = call_llm(system, user, model=ENGINEER_MODEL)

    r = github_api("post", "/issues", json={
        "title": "Initial landing page",
        "body": issue_body,
        "labels": ["enhancement"]
    })

    if r.status_code in (200, 201):
        issue = r.json()
        state["issue_url"] = issue["html_url"]
        print(f"  ✅ Issue created: {state['issue_url']}")
    else:
        print(f"  ❌ Failed to create issue: {r.status_code} — {r.text[:200]}")
        state["issue_url"] = f"https://github.com/{GITHUB_REPO}/issues"

    return state

# NODE 4 — Create branch, commit HTML, open PR
def commit_and_open_pr(state: EngineerState) -> EngineerState:
    print("\n" + "="*50)
    print(" ENGINEER AGENT — Committing code & opening PR")
    print("="*50)

    branch = state["branch_name"]

    # ── Step 1: Get SHA of main branch ────────────────
    r = github_api("get", "/git/ref/heads/main")
    if r.status_code != 200:
        print(f"  Could not get main branch: {r.status_code} — {r.text[:200]}")
        state["pr_url"] = f"https://github.com/{GITHUB_REPO}/pulls"
        state["status"] = "failed"
        return state

    base_sha = r.json()["object"]["sha"]
    print(f"  ✅ Main branch SHA: {base_sha[:12]}...")

    # Step 2: Create new branch
    r = github_api("post", "/git/refs", json={
        "ref": f"refs/heads/{branch}",
        "sha": base_sha
    })

    if r.status_code in (200, 201):
        print(f"   Branch '{branch}' created")
    elif r.status_code == 422:
        print(f"   Branch '{branch}' already exists — will update file")
    else:
        print(f"   Failed to create branch: {r.status_code} — {r.text[:200]}")
        state["pr_url"] = f"https://github.com/{GITHUB_REPO}/pulls"
        state["status"] = "failed"
        return state

    # Step 3: Commit HTML file
    content_b64 = base64.b64encode(state["html_content"].encode()).decode()

    file_data = {
        "message": "Add initial landing page — generated by EngineerAgent",
        "content": content_b64,
        "branch": branch,
        "committer": {
            "name": "EngineerAgent",
            "email": "agent@launchmind.ai"
        }
    }

    # Check if file already exists on branch (for updates)
    check = github_api("get", "/contents/index.html", params={"ref": branch})
    if check.status_code == 200:
        file_data["sha"] = check.json()["sha"]
        file_data["message"] = "Update landing page — revised by EngineerAgent"

    r = github_api("put", "/contents/index.html", json=file_data)

    if r.status_code in (200, 201):
        print(f"  ✅ index.html committed to '{branch}'")
    else:
        print(f"   Failed to commit file: {r.status_code} — {r.text[:200]}")
        state["status"] = "failed"
        return state

    # Step 4: Generate PR body with LLM
    system = """You are a software engineer writing a pull request description. Be professional and concise.
Output ONLY the PR body text in markdown. No wrapping fences."""

    spec = state.get("product_spec", {})
    feature_names = [f['name'] for f in spec.get('features', [])]

    user = f"""Write a PR description for:
Title: Initial landing page for LaunchMind
Startup: {state.get('startup_idea', '')}
Value proposition: {spec.get('value_proposition', '')}
Features included on page: {json.dumps(feature_names)}
Related issue: {state.get('issue_url', 'N/A')}

Describe what was built, the tech used, and what reviewers should check."""

    pr_body = call_llm(system, user, model=ENGINEER_MODEL)

    # Step 5: Open pull request
    r = github_api("post", "/pulls", json={
        "title": "Initial landing page — generated by EngineerAgent",
        "body": pr_body + f"\n\nResolves {state.get('issue_url', '')}",
        "head": branch,
        "base": "main"
    })

    if r.status_code in (200, 201):
        pr = r.json()
        state["pr_url"] = pr["html_url"]
        state["pr_number"] = pr["number"]
        state["status"] = "success"
        print(f"  ✅ Pull request opened: {state['pr_url']}")
    elif r.status_code == 422 and "pull request already exists" in r.text.lower():
        # PR already exists - fetch it
        existing = github_api("get", "/pulls", params={
            "head": f"{GITHUB_REPO.split('/')[0]}:{branch}",
            "state": "open"
        })
        if existing.status_code == 200 and existing.json():
            state["pr_url"] = existing.json()[0]["html_url"]
            state["status"] = "success"
            print(f"  PR already exists: {state['pr_url']}")
        else:
            state["pr_url"] = f"https://github.com/{GITHUB_REPO}/pulls"
            state["status"] = "partial"
            print(f"    PR exists but could not retrieve URL")
    else:
        print(f"   Failed to open PR: {r.status_code} — {r.text[:200]}")
        state["pr_url"] = f"https://github.com/{GITHUB_REPO}/pulls"
        state["status"] = "failed"

    return state


# NODE 5 — Send results back to CEO
def send_results(state: EngineerState) -> EngineerState:
    print("\n" + "="*50)
    print(" ENGINEER AGENT — Sending results to CEO")
    print("="*50)

    send_message(
        from_agent="engineer",
        to_agent="ceo",
        message_type="result",
        payload={
            "pr_url": state.get("pr_url", "N/A"),
            "issue_url": state.get("issue_url", "N/A"),
            "branch": state.get("branch_name", BRANCH_NAME),
            "status": state.get("status", "unknown"),
            "html_length": len(state.get("html_content", "")),
            "html_content": state.get("html_content", ""),  
            "pr_number":    state.get("pr_number", 0),      
        }
    )

    print(f"  ✅ Results sent to CEO")
    print(f"     PR:    {state.get('pr_url', 'N/A')}")
    print(f"     Issue: {state.get('issue_url', 'N/A')}")

    return state


def build_engineer_graph():
    graph = StateGraph(EngineerState)

    graph.add_node("read_inputs",    read_inputs)
    graph.add_node("generate_html",  generate_html)
    graph.add_node("create_issue",   create_github_issue)
    graph.add_node("commit_and_pr",  commit_and_open_pr)
    graph.add_node("send_results",   send_results)

    graph.set_entry_point("read_inputs")
    graph.add_edge("read_inputs",   "generate_html")
    graph.add_edge("generate_html", "create_issue")
    graph.add_edge("create_issue",  "commit_and_pr")
    graph.add_edge("commit_and_pr", "send_results")
    graph.add_edge("send_results",  END)

    return graph.compile()

def run_engineer_agent():
    print("\n" + "="*60)
    print("  ENGINEER AGENT — LaunchMind")
    print(f"   LLM Model: {ENGINEER_MODEL}")
    print(f"   GitHub Repo: {GITHUB_REPO}")
    print("="*60)

    app = build_engineer_graph()

    initial_state: EngineerState = {
        "startup_idea": "",
        "product_spec": {},
        "html_content": "",
        "issue_url": "",
        "pr_url": "",
        "pr_number": 0,
        "branch_name": BRANCH_NAME,
        "status": "",
        "is_revision": False,
        "revision_feedback": ""
    }

    return app.invoke(initial_state)


if __name__ == "__main__":
    run_engineer_agent()
