# agents/product_agent.py

import os
import sys
import json
from typing import TypedDict

from langgraph.graph import StateGraph, END

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from message_bus import send_message, get_messages
from groq_client import call_llm


PRODUCT_MODEL = "llama-3.3-70b-versatile"

def parse_json_response(raw: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()
    return json.loads(clean)


# STATE
class ProductState(TypedDict):
    startup_idea: str
    focus: str
    product_spec: dict
    is_revision: bool


# NODE 1 — Read task or revision request from CEO
def read_task(state: ProductState) -> ProductState:
    print("\n" + "="*50)
    print(" PRODUCT AGENT — Reading task from CEO")
    print("="*50)

    inbox = get_messages("product")

    for msg in inbox:
        if msg["from_agent"] == "ceo":
            payload = msg["payload"]
            state["startup_idea"] = payload.get("idea", state.get("startup_idea", ""))

            if msg["message_type"] == "revision_request":
                state["is_revision"] = True
                state["focus"] = payload.get("feedback", "")
                print(f"   Revision request received: {state['focus']}")
            else:
                state["is_revision"] = False
                state["focus"] = payload.get("focus", "")
                print(f"  ✅ Task received: {state['focus']}")

    return state


# NODE 2 — Generate product spec using LLM
def generate_spec(state: ProductState) -> ProductState:
    print("\n" + "="*50)
    print(" PRODUCT AGENT — Generating product spec with LLM")
    print(f"   Using model: {PRODUCT_MODEL}")
    print("="*50)

    revision_context = ""
    if state.get("is_revision"):
        revision_context = f"""
IMPORTANT: This is a revision. The CEO rejected the previous spec with this feedback:
"{state['focus']}"

Address this feedback specifically in your new spec. Make it more specific and detailed."""

    system = """You are a senior product manager at a startup. Given a startup idea, create a detailed product specification.

Respond ONLY with a valid JSON object. No explanation, no markdown fences, just pure JSON.

Required format:
{
    "value_proposition": "One sentence describing what the product does and for whom",
    "personas": [
        {"name": "RealName", "role": "Their role or description", "pain_point": "Their specific pain point"},
        {"name": "RealName", "role": "Their role or description", "pain_point": "Their specific pain point"},
        {"name": "RealName", "role": "Their role or description", "pain_point": "Their specific pain point"}
    ],
    "features": [
        {"name": "Feature Name", "description": "What it does in detail", "priority": 1},
        {"name": "Feature Name", "description": "What it does in detail", "priority": 2},
        {"name": "Feature Name", "description": "What it does in detail", "priority": 3},
        {"name": "Feature Name", "description": "What it does in detail", "priority": 4},
        {"name": "Feature Name", "description": "What it does in detail", "priority": 5}
    ],
    "user_stories": [
        "As a [specific user], I want to [specific action] so that [specific benefit]",
        "As a [specific user], I want to [specific action] so that [specific benefit]",
        "As a [specific user], I want to [specific action] so that [specific benefit]"
    ]
}

Make everything highly specific to the startup idea. Use Pakistani student names (e.g. Hamza, Fatima, Zain, Ayesha). Do NOT mention specific university names. No generic placeholders."""

    user = f"""Startup idea: {state['startup_idea']}

Focus area: {state.get('focus', 'Define the core user personas and top 5 features')}
{revision_context}
Generate the product specification now."""

    raw = call_llm(system, user, model=PRODUCT_MODEL)

    try:
        spec = parse_json_response(raw)
        required = ["value_proposition", "personas", "features", "user_stories"]
        for field in required:
            if field not in spec:
                raise ValueError(f"Missing required field: {field}")
        state["product_spec"] = spec
        print(f"   Product spec generated successfully")
        print(f"     Value prop: {spec['value_proposition']}")
        print(f"     Personas: {len(spec['personas'])}")
        print(f"     Features: {len(spec['features'])}")
        print(f"     User stories: {len(spec['user_stories'])}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"   JSON parse failed ({e}), using fallback spec")
        state["product_spec"] = {
            "value_proposition": "SathParho helps university students find and join study groups for their courses, reducing isolation and making collaborative learning easy.",
            "personas": [
                {"name": "Hamza", "role": "Third-year CS student", "pain_point": "Studies alone every semester and struggles to find peers taking the same electives"},
                {"name": "Fatima", "role": "Second-year business student", "pain_point": "Wants to form study groups but has no way to reach students outside her section"},
                {"name": "Zain", "role": "First-year engineering student", "pain_point": "New to campus and doesn't know anyone to study with for tough courses"}
            ],
            "features": [
                {"name": "Group Listings", "description": "Create study groups with subject, time slots, and preferred study style (online or in-person)", "priority": 1},
                {"name": "Smart Search", "description": "Search and filter groups by course, subject, schedule, and study style", "priority": 2},
                {"name": "Join Requests", "description": "Request to join groups and get notified when accepted", "priority": 3},
                {"name": "Schedule Matching", "description": "Automatically find groups that fit your available time slots", "priority": 4},
                {"name": "Verified Students", "description": "University email verification to ensure only real students join", "priority": 5}
            ],
            "user_stories": [
                "As a student, I want to create a study group listing so that I can find peers for my course.",
                "As a first-year student, I want to search groups by course code so that I find people studying the same material.",
                "As a busy student, I want to filter by time slot so that I join a group that fits my schedule."
            ]
        }

    return state


# NODE 3 — Send product spec to other agents
def send_outputs(state: ProductState) -> ProductState:
    print("\n" + "="*50)
    print(" PRODUCT AGENT — Sending outputs to other agents")
    print("="*50)

    spec = state["product_spec"]

    # Send to Engineer (include startup idea so Engineer has full context)
    send_message(
        from_agent="product",
        to_agent="engineer",
        message_type="result",
        payload={
            "startup_idea": state["startup_idea"],
            "product_spec": spec
        }
    )

    # Send to Marketing (include startup idea)
    send_message(
        from_agent="product",
        to_agent="marketing",
        message_type="result",
        payload={
            "startup_idea": state["startup_idea"],
            "product_spec": spec
        }
    )

    # Send confirmation + spec to CEO
    send_message(
        from_agent="product",
        to_agent="ceo",
        message_type="result",
        payload=spec
    )

    print("  ✅ Product spec sent to Engineer, Marketing, and CEO")

    return state

#build graph
def build_product_graph():
    graph = StateGraph(ProductState)

    graph.add_node("read_task", read_task)
    graph.add_node("generate_spec", generate_spec)
    graph.add_node("send_outputs", send_outputs)

    graph.set_entry_point("read_task")
    graph.add_edge("read_task", "generate_spec")
    graph.add_edge("generate_spec", "send_outputs")
    graph.add_edge("send_outputs", END)

    return graph.compile()


def run_product_agent():
    print("\n" + "="*60)
    print("  PRODUCT AGENT — LaunchMind")
    print(f"   LLM Model: {PRODUCT_MODEL}")
    print("="*60)

    app = build_product_graph()

    initial_state: ProductState = {
        "startup_idea": "",
        "focus": "",
        "product_spec": {},
        "is_revision": False
    }

    return app.invoke(initial_state)


if __name__ == "__main__":
    run_product_agent()
