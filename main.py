# main.py
from agents.ceo_agent import run_ceo_agent

if __name__ == "__main__":
    idea = input("💡 Enter your startup idea: ").strip()
    if not idea:
        idea = "SathParho — a web platform where university students can post and discover study groups for specific subjects and courses. Students create a group listing with their subject, university, available time slots, and preferred study style (online or in-person). Other students can browse and request to join. The goal is to reduce the isolation of self-studying and help students find peers who are at the same point in the course."
        print(f"  ℹ️  No idea entered — using default: SathParho")
    run_ceo_agent(idea)