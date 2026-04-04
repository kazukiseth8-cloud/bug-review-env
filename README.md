# \---

# title: Bug Review Env

# emoji: 🔍

# colorFrom: blue

# colorTo: red

# sdk: docker

# pinned: false

# tags:

# &#x20; - openenv

# \---

# Bug Review Environment 🔍

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment where an LLM agent reviews Python code snippets and identifies bugs. Built for the Meta × PyTorch × Hugging Face OpenEnv Hackathon.

\---

## Overview

Real software teams review hundreds of code changes daily. This environment trains and evaluates agents on the fundamental skill of **bug identification** — a task every production engineer performs.

The agent receives a Python code snippet and must:

1. Identify the **exact line** where the bug occurs
2. Classify the **bug type**
3. Provide an **explanation** of the bug and how to fix it

\---

## Tasks

|Task|Difficulty|Bug Type|Description|
|-|-|-|-|
|`find\\\_bug\\\_easy`|Easy|`off\\\_by\\\_one`|List indexing using `len(items)` instead of `len(items)-1`|
|`find\\\_bug\\\_medium`|Medium|`sql\\\_injection` / `hardcoded\\\_secret`|Authentication function with string-concatenated SQL query and hardcoded password|
|`find\\\_bug\\\_hard`|Hard|`race\\\_condition`|Multi-threaded banking code with unsynchronized check-then-act pattern|

\---

## Grading (0.0 – 1.0)

Each task awards partial credit:

|Component|Easy|Medium|Hard|
|-|-|-|-|
|Correct line number|0.40|0.35|0.30|
|Correct bug type|0.30|0.35|0.30|
|Correct explanation|0.30|0.30|0.40|
|**Max score**|**1.00**|**1.00**|**1.00**|

Hard task explanation requires mentioning **2+ key terms** (race condition, thread, lock, atomic, concurrent, etc.) for full credit.

\---

## Action Space

```json
{
  "buggy\\\_line": 3,
  "bug\\\_type": "off\\\_by\\\_one",
  "explanation": "Should use len(items)-1 as the index since Python lists are 0-indexed."
}
```

**`bug\\\_type` must be one of:**
`off\\\_by\\\_one` | `null\\\_dereference` | `sql\\\_injection` | `hardcoded\\\_secret` | `race\\\_condition` | `logic\\\_error` | `insecure\\\_deserialization` | `other`

\---

## Observation Space

```json
{
  "code\\\_snippet": "def get\\\_last\\\_element(items):\\\\n    return items\\\[len(items)]\\\\n",
  "task\\\_name": "find\\\_bug\\\_easy",
  "instructions": "The function contains a classic indexing bug...",
  "feedback": "",
  "done": false
}
```

\---

## Setup \& Usage

### Option 1 — Connect to live HF Space

```python
from bug\\\_review\\\_env import BugReviewAction, BugReviewEnv
import asyncio

async def main():
    async with BugReviewEnv(base\\\_url="https://Dan2000000-bug-review-env.hf.space") as env:
        obs = await env.reset(task\\\_name="find\\\_bug\\\_easy")
        print(obs.code\\\_snippet)
        result = await env.step(BugReviewAction(
            buggy\\\_line=3,
            bug\\\_type="off\\\_by\\\_one",
            explanation="Should use len(items)-1 as Python lists are 0-indexed."
        ))
        print(f"Reward: {result.reward}")  # 1.0 if all correct

asyncio.run(main())
```

### Option 2 — Run locally with Docker

```bash
# Build image
docker build -t bug-review-env:latest -f server/Dockerfile .

# Run server
docker run -p 7860:7860 bug-review-env:latest

# Test it
curl -X POST http://localhost:7860/reset \\\\
  -H "Content-Type: application/json" \\\\
  -d '{"task\\\_name": "find\\\_bug\\\_easy"}'
```

### Option 3 — Run inference baseline

```bash
export HF\\\_TOKEN=your\\\_token\\\_here
export BUG\\\_REVIEW\\\_ENV\\\_URL=https://Dan2000000-bug-review-env.hf.space

python inference.py
```

\---

## Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct`:

|Task|Score|
|-|-|
|`find\\\_bug\\\_easy`|\~0.90|
|`find\\\_bug\\\_medium`|\~0.70|
|`find\\\_bug\\\_hard`|\~0.50|

\---

## API Endpoints

|Endpoint|Method|Description|
|-|-|-|
|`/`|GET|Environment info|
|`/health`|GET|Health check|
|`/reset`|POST|Start new episode (`{"task\\\_name": "..."}`)|
|`/step`|POST|Submit action (`{"action": {...}}`)|
|`/state`|GET|Current episode state|
|`/tasks`|GET|List all tasks|

\---

## Project Structure

```
bug\\\_review\\\_env/
├── inference.py          # Baseline LLM agent (root level)
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml        # Package config
├── README.md
├── \\\_\\\_init\\\_\\\_.py
├── models.py             # Pydantic Action/Observation/State
├── client.py             # HTTP client
└── server/
    ├── app.py            # FastAPI server
    ├── environment.py    # Task logic + graders
    ├── requirements.txt
    └── Dockerfile
```

\---

## Why This Environment?

Code review is one of the most **high-value, high-frequency** tasks in software engineering. Training agents to reliably identify bugs could:

* Automate first-pass security reviews
* Catch regressions before they reach production
* Serve as a benchmark for code reasoning ability in LLMs

The three difficulty tiers create a meaningful progression that distinguishes weak models (only catch obvious errors) from strong ones (catch subtle concurrency bugs).

