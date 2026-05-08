# groq_client.py

import os
import time
import groq
from dotenv import load_dotenv
from pathlib import Path

# Always load .env from the project root regardless of where the script is called from
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ─────────────────────────────────────────────
# Load all 5 API keys from .env
# ─────────────────────────────────────────────
API_KEYS = []
for i in range(1, 6):
    key = os.environ.get(f"GROQ_API_KEY_{i}")
    if key:
        API_KEYS.append(key)

if not API_KEYS:
    raise ValueError("❌ No Groq API keys found in .env file!")

print(f"✅ Loaded {len(API_KEYS)} Groq API key(s)")

# ─────────────────────────────────────────────
# Key rotation state
# ─────────────────────────────────────────────
current_key_index = 0
key_failure_counts = {i: 0 for i in range(len(API_KEYS))}

# ─────────────────────────────────────────────
# Rate limiting state
# Groq free tier: ~30 requests/minute
# We stay safe at 20 requests/minute = 1 request every 3 seconds
# ─────────────────────────────────────────────
RATE_LIMIT_DELAY = 3.0      # seconds between requests
last_request_time = 0.0


def get_current_client():
    """Return a Groq client using the current API key."""
    return groq.Groq(api_key=API_KEYS[current_key_index])


def rotate_key():
    """Move to the next available API key."""
    global current_key_index
    old_index = current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    print(f"🔄 Rotated from key {old_index + 1} to key {current_key_index + 1}")


def call_llm(system_prompt: str, user_prompt: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 2048, retries: int = 5) -> str:
    """
    Call Groq LLM with:
    - Configurable model (different agents use different LLMs)
    - Rate limiting (waits between requests)
    - Automatic key rotation on rate limit errors
    - Retry logic with exponential backoff
    """
    global last_request_time, current_key_index

    for attempt in range(retries):
        # ── Rate limiting ──────────────────────────────
        elapsed = time.time() - last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            wait = RATE_LIMIT_DELAY - elapsed
            print(f"⏱️  Rate limit: waiting {wait:.1f}s before next request...")
            time.sleep(wait)

        try:
            client = get_current_client()
            print(f"🔑 Using API key {current_key_index + 1} of {len(API_KEYS)} | Model: {model}")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=max_tokens
            )

            last_request_time = time.time()
            result = response.choices[0].message.content
            # Strip <think>...</think> reasoning blocks from models like Qwen3
            import re
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
            return result

        except groq.RateLimitError as e:
            print(f"⚠️  Rate limit hit on key {current_key_index + 1}: {e}")
            key_failure_counts[current_key_index] += 1
            rotate_key()

            wait_time = 2 ** attempt  # exponential backoff: 1s, 2s, 4s, 8s...
            print(f"⏳ Backing off for {wait_time}s before retry {attempt + 1}/{retries}")
            time.sleep(wait_time)

        except groq.AuthenticationError as e:
            print(f"❌ Key {current_key_index + 1} is invalid: {e}")
            key_failure_counts[current_key_index] += 1
            rotate_key()

        except groq.APIError as e:
            print(f"❌ Groq API error on attempt {attempt + 1}: {e}")
            wait_time = 2 ** attempt
            time.sleep(wait_time)

        except Exception as e:
            print(f"❌ Unexpected error on attempt {attempt + 1}: {e}")
            wait_time = 2 ** attempt
            time.sleep(wait_time)

    raise RuntimeError(f"❌ All {retries} attempts failed across all available API keys.")