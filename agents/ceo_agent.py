# agents/ceo_agent.py

import os
import sys
import json
import requests
from dotenv import load_dotenv
from typing import TypedDict

from langgraph.graph import StateGraph, END

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from message_bus import send_message, get_messages
from groq_client import call_llm

load_dotenv()

# STATE - the memory flowing through every node
class CEOState(TypedDict):
    startup_idea: str
    tasks: dict
    agent_outputs: dict
    revision_counts: dict
    review_results: dict
    decision_log: list
    final_summary: str
    qa_report: dict           
    qa_done: bool             


def parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON safely."""
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()
    return json.loads(clean)

# NODE 1 - Decompose idea into tasks 
def decompose_idea(state: CEOState) -> CEOState:
    print("\n" + "=" * 50)
    print("🧠 NODE 1: Decomposing startup idea into tasks")
    print("=" * 50)

    system = """You are the CEO of a startup. Given a startup idea, break it into 
    specific tasks for three teams.
    
    Respond ONLY with a valid JSON object. No explanation, no markdown, just JSON.
    Format:
    {
        "product_task": "specific instruction for product team",
        "engineer_task": "specific instruction for engineering team",
        "marketing_task": "specific instruction for marketing team"
    }"""

    user = f"Startup idea: {state['startup_idea']}"

    raw = call_llm(system, user)

    try:
        tasks = parse_json_response(raw)
    except json.JSONDecodeError:
        print("⚠️  JSON parse failed, using fallback tasks")
        tasks = {
            "product_task": f"Define user personas and top 5 features for: {state['startup_idea']}",
            "engineer_task": f"Build an HTML landing page for: {state['startup_idea']}",
            "marketing_task": f"Create tagline, cold email, and social posts for: {state['startup_idea']}"
        }

    state["tasks"] = tasks
    state["decision_log"].append({
        "step": "decompose_idea",
        "decision": "Broke startup idea into 3 agent tasks",
        "reasoning": "CEO used LLM to generate role-specific tasks from the startup idea",
        "output": tasks
    })

    print(f"\n✅ Tasks generated:")
    for role, task in tasks.items():
        print(f"   {role}: {task}")

    return state

# NODE 2 — Send tasks to all agents via message bus
def send_tasks_to_agents(state: CEOState) -> CEOState:
    print("\n" + "=" * 50)
    print("📤 NODE 2: Sending tasks to sub-agents")
    print("=" * 50)

    idea = state["startup_idea"]
    tasks = state["tasks"]

    send_message(
        from_agent="ceo",
        to_agent="product",
        message_type="task",
        payload={
            "idea": idea,
            "focus": tasks.get("product_task")
        }
    )

    send_message(
        from_agent="ceo",
        to_agent="engineer",
        message_type="task",
        payload={
            "idea": idea,
            "focus": tasks.get("engineer_task")
        }
    )

    send_message(
        from_agent="ceo",
        to_agent="marketing",
        message_type="task",
        payload={
            "idea": idea,
            "focus": tasks.get("marketing_task")
        }
    )

    state["decision_log"].append({
        "step": "send_tasks",
        "decision": "Dispatched structured JSON tasks to Product, Engineer, Marketing agents",
        "reasoning": "Each agent needs its own focused task based on its role"
    })

    return state

# NODE 3 — Collect responses from agents
# CEO orchestrates: invokes sub-agents, then reads their responses.
def collect_agent_responses(state: CEOState) -> CEOState:
    print("\n" + "=" * 50)
    print("📥 NODE 3: Collecting agent responses")
    print("=" * 50)

    from message_bus import message_bus as bus

    # Run Product agent if it has pending messages
    if bus.get("product", []):
        from agents.product_agent import run_product_agent
        print("  🚀 Invoking Product agent...")
        run_product_agent()

    # Run Engineer agent if it has pending messages 
    if bus.get("engineer", []):
        from agents.engineer_agent import run_engineer_agent
        print("  🚀 Invoking Engineer agent...")
        run_engineer_agent()

    #  Run Marketing agent if it has pending messages
    if bus.get("marketing", []):
        from agents.marketing_agent import run_marketing_agent
        print("  🚀 Invoking Marketing agent...")
        run_marketing_agent()

    #  Now read all responses sent back to CEO 
    inbox = get_messages("ceo")
    outputs = dict(state.get("agent_outputs") or {})

    for msg in inbox:
        sender = msg["from_agent"]
        if sender == "qa":
            continue
        outputs[sender] = msg["payload"]
        print(f"  ✅ Response received from {sender.upper()}")

    state["agent_outputs"] = outputs
    return state

# NODE 4 — Review outputs with LLM 
def review_agent_outputs(state: CEOState) -> CEOState:
    print("\n" + "=" * 50)
    print("🔍 NODE 4: Reviewing agent outputs with LLM")
    print("=" * 50)

    review_results = {}

    for agent_name, output in state["agent_outputs"].items():

        system = f"""You are a demanding startup CEO reviewing your {agent_name} team's work.
        
        Be specific and critical. Respond ONLY with valid JSON. No markdown, no explanation.
        Format:
        {{
            "verdict": "accept" or "revise",
            "reason": "one sentence explaining your verdict",
            "feedback": "specific actionable improvement (empty string if verdict is accept)"
        }}"""

        user = f"""Startup idea: {state['startup_idea']}

{agent_name.upper()} agent output:
{json.dumps(output, indent=2)}

Review: Is this output specific, relevant, complete, and high quality for this startup idea?"""

        raw = call_llm(system, user)

        try:
            review = parse_json_response(raw)
            if "verdict" not in review:
                review = {"verdict": "accept", "reason": "Output accepted", "feedback": ""}
        except json.JSONDecodeError:
            print(f"  ⚠️  Could not parse review for {agent_name}, defaulting to accept")
            review = {"verdict": "accept", "reason": "Parse error - defaulting to accept", "feedback": ""}

        review_results[agent_name] = review

        emoji = "✅" if review["verdict"] == "accept" else "⚠️"
        print(f"\n  {emoji} {agent_name.upper()}: {review['verdict'].upper()}")
        print(f"     Reason: {review['reason']}")
        if review.get("feedback"):
            print(f"     Feedback: {review['feedback']}")

    state["review_results"] = review_results
    state["decision_log"].append({
        "step": "review_outputs",
        "decision": "CEO reviewed all agent outputs using LLM reasoning",
        "reasoning": "Each output was evaluated for specificity, completeness, and relevance",
        "verdicts": {k: v["verdict"] for k, v in review_results.items()}
    })

    return state

# NODE 5 — Send revision requests if needed
def handle_revisions(state: CEOState) -> CEOState:
    print("\n" + "=" * 50)
    print("🔄 NODE 5: Handling revision requests")
    print("=" * 50)

    any_revision = False

    for agent_name, review in state["review_results"].items():
        if review["verdict"] == "revise":
            count = state["revision_counts"].get(agent_name, 0)

            if count < 2:
                any_revision = True
                state["revision_counts"][agent_name] = count + 1

                send_message(
                    from_agent="ceo",
                    to_agent=agent_name,
                    message_type="revision_request",
                    payload={
                        "idea":            state["startup_idea"],
                        "feedback":        review["feedback"],
                        "reason":          review["reason"],
                        "original_output": state["agent_outputs"].get(agent_name, {}),
                        "product_spec":    state["agent_outputs"].get("product", {}),
                        "attempt":         count + 1
                    }
                )

                state["decision_log"].append({
                    "step": "revision_request",
                    "agent": agent_name,
                    "attempt": count + 1,
                    "reason": review["reason"],
                    "feedback_sent": review["feedback"],
                    "reasoning": f"CEO determined {agent_name}'s output was insufficient and requested revision"
                })

                print(f"  📝 Sent revision request to {agent_name.upper()} (attempt {count + 1}/2)")

            else:
                print(f"  ⚠️  {agent_name.upper()} hit max revisions (2). Accepting as-is.")
                state["review_results"][agent_name]["verdict"] = "accept"
                state["decision_log"].append({
                    "step": "max_revisions_reached",
                    "agent": agent_name,
                    "reasoning": "Max revision attempts reached — accepting output to avoid infinite loop"
                })

    if not any_revision:
        print("  ✅ No revisions needed")

    return state

# CONDITIONAL EDGE — loop back or go to QA?
def check_if_revisions_needed(state: CEOState) -> str:
    for agent_name, review in state["review_results"].items():
        if review["verdict"] == "revise":
            count = state["revision_counts"].get(agent_name, 0)
            if count < 2:
                print(f"\n🔁 CEO: Still needs revision from {agent_name}. Looping back.")
                return "needs_revision"

    print("\n✅ CEO: All outputs accepted. Sending to QA agent.")
    return "all_accepted"

# NODE 6 — Send outputs to QA agent for review
def run_qa_review(state: CEOState) -> CEOState:
    print("\n" + "=" * 50)
    print("🔬 NODE 6: Triggering QA Agent review")
    print("=" * 50)

    # Skip if QA already ran
    if state.get("qa_done"):
        print("  ℹ️  QA already completed — skipping")
        return state

    outputs = state["agent_outputs"]

    engineer_output = outputs.get("engineer", {})
    marketing_output = outputs.get("marketing", {})
    product_output = outputs.get("product", {})

    # Pull html_content from engineer if available
    html_content = engineer_output.get("html_content", "")
    pr_url = engineer_output.get("pr_url", "")
    pr_number = engineer_output.get("pr_number", 0)

    # If engineer returned a pr_url but not html_content directly,
    # the engineer agent sends html_length — html itself lives in engineer state.
    # We pass what we have; QA agent handles missing html gracefully.
    send_message(
        from_agent="ceo",
        to_agent="qa",
        message_type="task",
        payload={
            "startup_idea":   state["startup_idea"],
            "product_spec":   product_output,
            "html_content":   html_content,
            "pr_url":         pr_url,
            "pr_number":      pr_number,
            "marketing_copy": {
                "tagline":      marketing_output.get("tagline", ""),
                "description":  marketing_output.get("description", ""),
                "email_subject": marketing_output.get("email_subject", ""),
                "social_posts": marketing_output.get("social_posts", {}),
            }
        }
    )

    # Invoke QA agent 
    from agents.qa_agent import run_qa_agent
    print("  🚀 Invoking QA agent...")
    run_qa_agent()

    # Read QA report back from message bus 
    inbox = get_messages("ceo")
    qa_report = {}

    for msg in inbox:
        if msg["from_agent"] == "qa":
            qa_report = msg["payload"]
            print(f"  ✅ QA report received — overall verdict: {qa_report.get('overall_verdict', 'N/A').upper()}")

    state["qa_report"] = qa_report
    state["qa_done"] = True

    state["decision_log"].append({
        "step": "qa_review",
        "decision": "CEO dispatched outputs to QA agent and received structured review",
        "reasoning": "QA checks HTML and marketing copy against product spec",
        "verdict": qa_report.get("overall_verdict", "unknown")
    })

    return state

# CONDITIONAL EDGE — act on QA verdict
def check_qa_verdict(state: CEOState) -> str:
    qa_report = state.get("qa_report", {})
    verdict   = qa_report.get("overall_verdict", "pass")

    if verdict == "fail":
        # Check we haven't already retried too many times
        html_revisions = state["revision_counts"].get("engineer_qa", 0)
        mkt_revisions  = state["revision_counts"].get("marketing_qa", 0)

        if html_revisions < 1 or mkt_revisions < 1:
            print("\n🔁 CEO: QA failed — requesting targeted revisions.")
            return "qa_failed"

    print("\n✅ CEO: QA passed (or max retries reached). Posting final summary.")
    return "qa_passed"

# NODE 7 — Handle QA-driven revisions
def handle_qa_revisions(state: CEOState) -> CEOState:
    print("\n" + "=" * 50)
    print("🔄 NODE 7: Handling QA revision requests")
    print("=" * 50)

    qa_report = state.get("qa_report", {})

    # Engineer revision if HTML failed 
    if qa_report.get("html_verdict") == "fail":
        count = state["revision_counts"].get("engineer_qa", 0)
        if count < 1:
            state["revision_counts"]["engineer_qa"] = count + 1
            send_message(
                from_agent="ceo",
                to_agent="engineer",
                message_type="revision_request",
                payload={
                    "idea":         state["startup_idea"],
                    "feedback":     qa_report.get("html_feedback", "Improve the landing page."),
                    "reason":       f"QA score {qa_report.get('html_score', 'N/A')}/10 — did not meet threshold",
                    "product_spec": state["agent_outputs"].get("product", {}),
                    "attempt":      count + 1
                }
            )
            print(f"  📝 Revision request sent to ENGINEER (QA attempt {count + 1})")

            from agents.engineer_agent import run_engineer_agent
            run_engineer_agent()

            # Collect updated engineer output
            for msg in get_messages("ceo"):
                if msg["from_agent"] == "engineer":
                    state["agent_outputs"]["engineer"] = msg["payload"]
                    print("  ✅ Updated Engineer output collected")

    # Marketing revision if copy failed
    if qa_report.get("marketing_verdict") == "fail":
        count = state["revision_counts"].get("marketing_qa", 0)
        if count < 1:
            state["revision_counts"]["marketing_qa"] = count + 1
            send_message(
                from_agent="ceo",
                to_agent="marketing",
                message_type="revision_request",
                payload={
                    "idea":         state["startup_idea"],
                    "feedback":     qa_report.get("marketing_feedback", "Improve the marketing copy."),
                    "reason":       f"QA score {qa_report.get('marketing_score', 'N/A')}/10 — did not meet threshold",
                    "product_spec": state["agent_outputs"].get("product", {}),
                    "attempt":      count + 1
                }
            )
            print(f"  📝 Revision request sent to MARKETING (QA attempt {count + 1})")

            from agents.marketing_agent import run_marketing_agent
            run_marketing_agent()

            for msg in get_messages("ceo"):
                if msg["from_agent"] == "marketing":
                    state["agent_outputs"]["marketing"] = msg["payload"]
                    print("  ✅ Updated Marketing output collected")

    # Reset qa_done so QA re-runs on the revised outputs
    state["qa_done"] = False

    state["decision_log"].append({
        "step": "qa_revision_requested",
        "reasoning": "CEO acted on QA fail verdict by requesting specific revisions from relevant agents"
    })

    return state

# NODE 8 — Post final summary to Slack
def post_final_summary(state: CEOState) -> CEOState:
    print("\n" + "=" * 50)
    print("📣 NODE 8: Posting final summary to Slack")
    print("=" * 50)

    outputs  = state["agent_outputs"]
    idea = state["startup_idea"]
    tagline = outputs.get("marketing", {}).get("tagline", "N/A")
    pr_url = outputs.get("engineer",  {}).get("pr_url",  "N/A")
    val_prop = outputs.get("product",   {}).get("value_proposition", "N/A")
    qa_report = state.get("qa_report", {})

    summary = (
        f"🚀 LaunchMind Launch Report\n"
        f"Idea: {idea}\n"
        f"Value Prop: {val_prop}\n"
        f"Tagline: {tagline}\n"
        f"PR: {pr_url}"
    )
    state["final_summary"] = summary

    slack_token = os.environ.get("SLACK_BOT_TOKEN", "")

    if slack_token:
        qa_section = (
            f"*🔬 QA Verdict:* {qa_report.get('overall_verdict', 'N/A').upper()}\n"
            f"• HTML: {qa_report.get('html_verdict', 'N/A').upper()} "
            f"({qa_report.get('html_score', 'N/A')}/10)\n"
            f"• Marketing: {qa_report.get('marketing_verdict', 'N/A').upper()} "
            f"({qa_report.get('marketing_score', 'N/A')}/10)"
        ) if qa_report else "*🔬 QA:* Not run"

        payload = {
            "channel": "#launches",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "🚀 LaunchMind: Final Launch Report"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*💡 Startup Idea:*\n{idea}"}
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*🎯 Value Proposition:*\n{val_prop}"},
                        {"type": "mrkdwn", "text": f"*✨ Tagline:*\n_{tagline}_"}
                    ]
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*🔗 GitHub PR:* <{pr_url}|View Pull Request>"}
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": qa_section}
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*📊 CEO Review Summary:*\n"
                            + "\n".join([
                                f"• {k.upper()}: {v['verdict'].upper()} — {v['reason']}"
                                for k, v in state["review_results"].items()
                            ])
                        )
                    }
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "✅ *All agents completed. LaunchMind is live.*"}
                }
            ]
        }

        r = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {slack_token}"},
            json=payload
        )
        result = r.json()
        if result.get("ok"):
            print("✅ Final summary posted to Slack #launches successfully!")
        else:
            print(f"❌ Slack error: {result.get('error')}")
    else:
        print("⚠️  No Slack token — printing summary instead:")
        print(summary)

    # Print full decision log 
    print("\n" + "=" * 50)
    print("📋 FULL CEO DECISION LOG")
    print("=" * 50)
    for i, entry in enumerate(state["decision_log"], 1):
        print(f"\n[{i}] Step: {entry.get('step')}")
        print(f"    Decision: {entry.get('decision', 'N/A')}")
        print(f"    Reasoning: {entry.get('reasoning', 'N/A')}")

    return state

# BUILD THE GRAPH
def build_ceo_graph():
    graph = StateGraph(CEOState)

    graph.add_node("decompose_idea",     decompose_idea)
    graph.add_node("send_tasks",         send_tasks_to_agents)
    graph.add_node("collect_responses",  collect_agent_responses)
    graph.add_node("review_outputs",     review_agent_outputs)
    graph.add_node("handle_revisions",   handle_revisions)
    graph.add_node("run_qa_review",      run_qa_review)
    graph.add_node("handle_qa_revisions", handle_qa_revisions)
    graph.add_node("post_summary",       post_final_summary)

    graph.set_entry_point("decompose_idea")
    graph.add_edge("decompose_idea",   "send_tasks")
    graph.add_edge("send_tasks",       "collect_responses")
    graph.add_edge("collect_responses", "review_outputs")
    graph.add_edge("review_outputs",   "handle_revisions")

    # CEO feedback loop: revise → re-collect, or move to QA
    graph.add_conditional_edges(
        "handle_revisions",
        check_if_revisions_needed,
        {
            "needs_revision": "collect_responses",
            "all_accepted":   "run_qa_review"
        }
    )

    # QA feedback loop: fail → revise → re-run QA, or post summary
    graph.add_conditional_edges(
        "run_qa_review",
        check_qa_verdict,
        {
            "qa_failed": "handle_qa_revisions",
            "qa_passed": "post_summary"
        }
    )

    graph.add_edge("handle_qa_revisions", "run_qa_review")
    graph.add_edge("post_summary", END)

    return graph.compile()

# RUN CEO AGENT
def run_ceo_agent(startup_idea: str):
    print("\n" + "=" * 60)
    print("🏢  CEO AGENT — LaunchMind")
    print(f"💡  Idea: {startup_idea}")
    print("=" * 60)

    app = build_ceo_graph()

    initial_state: CEOState = {
        "startup_idea":    startup_idea,
        "tasks":           {},
        "agent_outputs":   {},
        "revision_counts": {},
        "review_results":  {},
        "decision_log":    [],
        "final_summary":   "",
        "qa_report":       {},
        "qa_done":         False,
    }

    return app.invoke(initial_state)

if __name__ == "__main__":
    run_ceo_agent(
        "SathParho — a web platform where university students can post and "
        "discover study groups for specific subjects and courses"
    )