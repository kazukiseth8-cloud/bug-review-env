---
title: Bug Review Env
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
---

# Bug Review Environment 🔍

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment where an LLM agent reviews Python code snippets and identifies bugs. Built for the Meta × PyTorch × Hugging Face OpenEnv Hackathon.

---

## Overview

Real software teams review hundreds of code changes daily. This environment trains and evaluates agents on the fundamental skill of **bug identification** — a task every production engineer performs.

The agent receives a Python code snippet and must:

1. Identify the **exact line** where the bug occurs
2. Classify the **bug type**
3. Provide an **explanation** of the bug and how to fix it

---

## Tasks

| Task | Difficulty | Bug Type | Description |
|---|---|---|---|
| `find_bug_easy` | Easy | `off_by_one` | List indexing using `len(items)` instead of `len(items)-1` |
| `find_bug_medium` | Medium | `sql_injection` / `hardcoded_secret` | Authentication function with string-concatenated SQL query and hardcoded password |
| `find_bug_hard` | Hard | `race_condition` | Multi-threaded banking code with unsynchronized check-then-act pattern |

---

## Grading

Each task awards partial credit. All scores are strictly between 0 and 1 (exclusive).

| Component | Easy | Medium | Hard |
|---|---|---|---|
| Correct line number | 0.40 | 0.35 | 0.30 |
| Correct bug type | 0.30 | 0.35 | 0.30 |
| Correct explanation | 0.29 | 0.29 | 0.39 |
| **Max score** | **0.99** | **0.99** | **0.99** |

Hard task explanation requires mentioning **2+ key terms** (race condition, thread, lock, atomic, concurrent, etc.) for full credit.

---

## Action Space

```json
{
  "buggy_line": 3,
  "bug_type": "off_by_one",
  "explanation": "Should use len(items)-1 as the index since Python lists are 0-indexed."
}
```

**`bug_type` must be one of:**
`off_by_one` | `null_dereference` | `sql_injection` | `hardcoded_secret` | `race_condition` | `logic_error` | `insecure_deserialization` | `other`

---

## Observation Space

```json
{
  "code_snippet": "def get_last_element(items):\n    return items[len(items)]\n",
  "task_name": "find_bug_easy",
  "instructions": "The function contains a classic indexing bug...",
  "feedback": "",
  "done": false
}
```

---

## Setup & Usage

### Option 1 — Connect to live HF Space

```python
from bug_review_env import BugReviewAction, BugReviewEnv
import asyncio

async def main():
    async with BugReviewEnv(base_url="https://Dan2000000-bug-review-env.hf.space") as env:
        obs = await env.reset(task_name="find_bug_easy")
        print(obs.code_snippet)
        result = await env.step(BugReviewAction(
            buggy_line=3,
            bug_type="off_by_one",
            explanation="Should use len(items)-1 as Python lists are 0-indexed."
        ))
        print(f"Reward: {result.reward}")

asyncio.run(main())
```

### Option 2 — Run locally with Docker

```bash
docker build -t bug-review-env:latest .
docker run -p 7860:7860 bug-review-env:latest
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "find_bug_easy"}'
```

### Option 3 — Run inference baseline

```bash
export HF_TOKEN=your_token_here
export BUG_REVIEW_ENV_URL=https://Dan2000000-bug-review-env.hf.space
python inference.py
```

---

## Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Score |
|---|---|
| `find_bug_easy` | ~0.90 |
| `find_bug_medium` | ~0.70 |
| `find_bug_hard` | ~0.50 |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Environment info |
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode (`{"task_name": "..."}`) |
| `/step` | POST | Submit action (`{"action": {...}}`) |
| `/state` | GET | Current episode state |
| `/tasks` | GET | List all tasks |

---

## Project Structure

```
bug_review_env/
├── inference.py          # Baseline LLM agent (root level)
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml        # Package config
├── README.md
├── __init__.py
├── models.py             # Pydantic Action/Observation/State
├── client.py             # HTTP client
└── server/
    ├── app.py            # FastAPI server
    ├── environment.py    # Task logic + graders
    └── Dockerfile
```

---

## Why This Environment?

Code review is one of the most **high-value, high-frequency** tasks in software engineering. Training agents to reliably identify bugs could:

- Automate first-pass security reviews
- Catch regressions before they reach production
- Serve as a benchmark for code reasoning ability in LLMs

The three difficulty tiers create a meaningful progression that distinguishes weak models (only catch obvious errors) from strong ones (catch subtle concurrency bugs).
