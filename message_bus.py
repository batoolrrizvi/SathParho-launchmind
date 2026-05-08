# message_bus.py

import json
from datetime import datetime
import uuid

message_bus = {
    "ceo": [],
    "product": [],
    "engineer": [],
    "marketing": [],
    "qa": []
}

def send_message(from_agent, to_agent, message_type, payload, parent_id=None):
    message = {
        "message_id": str(uuid.uuid4()),
        "from_agent": from_agent,
        "to_agent": to_agent,
        "message_type": message_type,
        "payload": payload,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "parent_message_id": parent_id
    }
    message_bus[to_agent].append(message)
    print(f"\n📨 [{from_agent.upper()} → {to_agent.upper()}] Type: {message_type}")
    print(json.dumps(payload, indent=2))
    return message["message_id"]

def get_messages(agent_name):
    messages = message_bus[agent_name].copy()
    message_bus[agent_name] = []
    return messages

def get_full_log():
    return message_bus