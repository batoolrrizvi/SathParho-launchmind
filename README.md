# LaunchMind — Multi-Agent Startup System

> A team of autonomous AI agents that takes a startup idea and runs with it — defining the product, building a landing page on GitHub, writing marketing copy, sending emails, posting to Slack, and reviewing its own work.

**Group Members:**
| Name | Agent |
|------|-------|
| Arisha Khan| CEO Agent |
| Rameen Babar | Product Agent + Engineer Agent |
| Batool Rizvi | Marketing Agent + QA Agent |

---

## What Is This?

LaunchMind is a Multi-Agent System (MAS). You give it a startup idea in plain text. Five autonomous AI agents then collaborate - passing structured JSON messages to each other to produce real outputs on real platforms with no human involvement after the idea is entered.

**Startup idea used:** SathParho - a web platform where university students can post and discover study groups for specific subjects and courses. Students create a group listing with their subject, available time slots, and preferred study style (online or in-person). Other students can browse and request to join, reducing the isolation of solo studying.

---

## Agent Architecture

```
                        ┌─────────────┐
                        │  CEO Agent  │  ← You enter the idea here
                        └──────┬──────┘
                               │ decomposes idea into tasks
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
       │Product Agent│  │Engineer Agent│  │ Marketing Agent │
       └──────┬──────┘  └──────┬───────┘  └────────┬────────┘
              │                │                    │
         product spec     PR + HTML            copy + email
              └────────────────┴────────────────────┘
                               │ all outputs sent back to CEO
                        ┌──────▼──────┐
                        │   CEO Agent │  ← reviews outputs, requests revisions
                        └──────┬──────┘
                               │ sends outputs for quality check
                        ┌──────▼──────┐
                        │   QA Agent  │  ← reviews HTML + marketing copy
                        └──────┬──────┘
                               │ pass/fail verdict
                        ┌──────▼──────┐
                        │  CEO Agent  │  ← posts final summary to Slack
                        └─────────────┘
```

**Message flow summary:**
- CEO → Product: task
- CEO → Engineer: task
- CEO → Marketing: task
- Product → Engineer: product spec JSON
- Product → Marketing: product spec JSON
- Product → CEO: confirmation + spec
- Engineer → CEO: PR URL + HTML content + issue URL
- Marketing → CEO: tagline + email status + social posts
- CEO → QA: all outputs bundled
- QA → CEO: structured pass/fail review report
- CEO → any agent: `revision_request` if output is rejected

---

## What Each Agent Does

### Agent 1 — CEO Agent (`agents/ceo_agent.py`)
The orchestrator. Runs a LangGraph state machine with 8 nodes.

**Responsibilities:**
- Takes the startup idea as input
- Uses an LLM to decompose it into specific tasks for each sub-agent
- Sends structured JSON task messages via the message bus
- Invokes each sub-agent in sequence and collects their responses
- Uses an LLM to review every agent's output - does not just forward it
- Sends `revision_request` messages if an output is insufficient (up to 2 retries per agent)
- Triggers the QA agent once all outputs are accepted
- If QA fails, requests targeted revisions from the specific agent that failed
- Posts the final launch summary to Slack using Block Kit


---

### Agent 2 — Product Agent (`agents/product_agent.py`)
Thinks like a product manager.

**Responsibilities:**
- Reads its task from the CEO via the message bus
- Uses an LLM to generate a full product specification as structured JSON
- Sends the spec to the Engineer agent, Marketing agent, and CEO

**Output — product spec JSON:**
```json
{
  "value_proposition": "One sentence describing what the product does and for whom",
  "personas": [
    { "name": "Hamza Khan", "role": "Third-year engineering student", "pain_point": "..." }
  ],
  "features": [
    { "name": "Study Group Listing", "description": "...", "priority": 1 }
  ],
  "user_stories": [
    "As a student, I want to..."
  ]
}
```

**Model used:** `llama-3.3-70b-versatile`

---

### Agent 3 — Engineer Agent (`agents/engineer_agent.py`)
The builder. Takes real action on GitHub.

**Responsibilities:**
- Reads the product spec from the Product agent
- Uses an LLM to generate a complete HTML landing page (no frameworks, inline CSS)
- Creates a GitHub issue titled "Initial landing page" with an LLM-generated description
- Creates a new branch (`agent-landing-page`) on the repository
- Commits `index.html` to that branch (authored as `EngineerAgent <agent@launchmind.ai>`)
- Opens a pull request with an LLM-generated title and body
- Sends the PR URL, issue URL, PR number, and full HTML back to the CEO

**Real GitHub actions:** issue created, file committed, PR opened - all visible at:
   https://github.com/rameenbabarr/launchmind-SathParho/pulls

**Model used:** `qwen/qwen3-32b`

---

### Agent 4 — Marketing Agent (`agents/marketing_agent.py`)
Thinks like a growth marketer.

**Responsibilities:**
- Reads the product spec from the Product agent
- Uses an LLM to generate: a tagline (under 10 words), a 2–3 sentence landing page description, a cold outreach email with subject and HTML body, and social posts for Twitter/X, LinkedIn, and Instagram
- Sends the cold email to a verified SendGrid address
- Posts a Block Kit formatted message to the `#launches` Slack channel
- Sends all generated copy back to the CEO

**Real platform actions:** email sent via SendGrid, Slack message posted with Block Kit

---

### Agent 5 — QA / Reviewer Agent (`agents/qa_agent.py`)
The quality gate.

**Responsibilities:**
- Receives the HTML landing page, marketing copy, and product spec from the CEO
- Uses an LLM to review the HTML: does the headline match the value proposition? Are all features present? Is there a clear CTA?
- Uses an LLM to review the marketing copy: is the tagline under 10 words? Does the email have a CTA? Is the tone right for students?
- Posts a structured review (APPROVE or REQUEST_CHANGES) on the GitHub pull request with at least 2 inline comments on `index.html`
- Sends a structured `pass`/`fail` report back to the CEO with specific feedback per agent

**Dynamic decision-making:** if QA returns `fail`, the CEO reads the specific `html_feedback` and `marketing_feedback` fields and sends targeted `revision_request` messages to only the agents that need to revise. This loop runs until QA passes or the retry limit is reached.

---

## Platforms Used

| Platform | What the agents do |
|----------|-------------------|
| **GitHub** | Engineer opens a real issue, commits `index.html` to a branch, and opens a pull request. QA posts an inline review with comments. |
| **Slack** | Marketing posts a Block Kit launch message to `#launches`. CEO posts the final summary with QA verdict and all agent results.**Join the workspace:** [Click here to join](https://join.slack.com/t/launchmind-sathparho/shared_invite/zt-3v395zdn8-x139RafVqSpamAhSQPRLHw) |



| **SendGrid** | Marketing sends a cold outreach email to a verified address with an LLM-generated subject and HTML body. |
| **Groq API** | All 5 agents use Groq-hosted LLMs for reasoning. API keys rotate automatically on rate limit. |

---

## Message Bus

All agents communicate through a shared in-memory Python dictionary (`message_bus.py`). Every message follows this schema:

```json
{
  "message_id": "uuid-string",
  "from_agent": "ceo",
  "to_agent": "product",
  "message_type": "task",
  "payload": { "idea": "...", "focus": "..." },
  "timestamp": "2025-04-10T12:00:00Z",
  "parent_message_id": null
}
```

`message_type` is one of: `task`, `result`, `revision_request`, `confirmation`.

The full message history is printed to the terminal during every run and stored in the CEO's `decision_log`.

---

## Setup Instructions

## Repository Structure

```
launchmind-SathParho/
├── agents/
│   ├── __init__.py           # empty, marks folder as a Python package
│   ├── ceo_agent.py       
│   ├── product_agent.py  
│   ├── engineer_agent.py 
│   ├── marketing_agent.py
│   └── qa_agent.py       
├── main.py               
├── message_bus.py        
├── groq_client.py        
├── test_email.py         
├── test_slack.py         
├── requirements.txt
├── .env.example          
├── .gitignore            
└── README.md
```

### 1. Clone the repository
```bash
git clone https://github.com/rameenbabarr/launchmind-SathParho.git
cd launchmind-SathParho
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Open `.env` and add your values:
```
GROQ_API_KEY_1=your_groq_key_here
GROQ_API_KEY_2=optional_second_key
GROQ_API_KEY_3=optional_third_key
GROQ_API_KEY_4=optional_fourth_key
GROQ_API_KEY_5=optional_fifth_key

GITHUB_TOKEN=your_github_pat_here
GITHUB_REPO=your_username/launchmind-SathParho

SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SENDGRID_API_KEY=SG.your_sendgrid_key
SENDGRID_FROM_EMAIL=your_verified_sender@email.com
```

### 4. Test each platform individually 

```bash
# Test Slack connection
python test_slack.py

# Test email sending
python test_email.py
```

### 5. Run the full system
```bash
python main.py
```

When prompted, either type your startup idea or press Enter to use the default SathParho idea.

---

