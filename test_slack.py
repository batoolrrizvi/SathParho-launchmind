import requests
import os
from dotenv import load_dotenv

load_dotenv()

token = os.environ["SLACK_BOT_TOKEN"]

r = requests.post("https://slack.com/api/chat.postMessage", 
    headers={"Authorization": f"Bearer {token}"},
    json={
        "channel": "#launches",
        "text": "LaunchMind bot is working!"
    }
)
print(r.json())