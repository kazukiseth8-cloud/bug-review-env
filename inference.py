"""
inference.py — Bug Review Environment Baseline Agent
Strictly follows Meta x Scaler OpenEnv Hackathon guidelines.
All rewards strictly in (0, 1) — never 0.0, never 1.0.
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("BUG_REVIEW_ENV_URL", "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK  = "bug-review-env"
MAX_STEPS  = 3
TEMPERATURE = 0.2
MAX_TOKENS  = 400
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = ["find_bug_easy", "find_bug_medium", "find_bug_hard"]

# ---------------------------------------------------------------------------
# Score clamping — strictly (0, 1) exclusive
# ---------------------------------------------------------------------------

def _clamp(score: float) -> float:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.05
    if s != s:  # NaN
        return 0.05
    if s <= 0.0:
        return 0.05
    if s >= 1.0:
        return 0.95
    return s

# ---------------------------------------------------------------------------
# Logging helpers — MANDATORY FORMAT
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Always clamp reward before logging
    safe_reward = _clamp(reward)
    error_val   = error if error else "null"
    done_val    = str(done).lower()
    action_clean = action.replace(" ", "_").replace("\n", "").replace("\r", "")[:80]
    print(
        f"[STEP] step={step} action={action_clean} reward={safe_reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    # Always clamp all rewards before logging
    safe_rewards = [_clamp(r) for r in rewards] if rewards else [0.05]
    rewards_str  = ",".join(f"{r:.2f}" for r in safe_rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert software security engineer performing code review.
    You will be shown a Python code snippet that contains a bug.
    Identify:
      1. The exact line number (1-indexed) where the bug occurs
      2. The bug type — one of:
           off_by_one | null_dereference | sql_injection | hardcoded_secret |
           race_condition | logic_error | insecure_deserialization | other
      3. A clear explanation of the bug and how to fix it

    If you receive feedback from a previous attempt, use it to improve.

    Respond with ONLY a valid JSON object:
    {
      "buggy_line": <integer>,
      "bug_type": "<type>",
      "explanation": "<explanation>"
    }
""").strip()


def build_user_prompt(code_snippet: str, instructions: str, feedback: str = "") -> str:
    prompt = (
        f"INSTRUCTIONS:\n{instructions}\n\n"
        f"CODE (lines numbered from 1):\n```python\n{code_snippet}\n```\n"
    )
    if feedback:
        prompt += f"\nFEEDBACK FROM PREVIOUS ATTEMPT:\n{feedback}\n"
    prompt += "\nRespond with ONLY JSON: buggy_line, bug_type, explanation."
    return prompt

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_agent_action(client: OpenAI, code_snippet: str, instructions: str, feedback: str = "") -> dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(code_snippet, instructions, feedback)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip().lstrip("json").strip()
                try:
                    parsed = json.loads(part)
                    return {
                        "buggy_line":  int(parsed.get("buggy_line", 0)),
                        "bug_type":    str(parsed.get("bug_type", "other")),
                        "explanation": str(parsed.get("explanation", "")),
                    }
                except Exception:
                    continue
        parsed = json.loads(raw)
        return {
            "buggy_line":  int(parsed.get("buggy_line", 0)),
            "bug_type":    str(parsed.get("bug_type", "other")),
            "explanation": str(parsed.get("explanation", "")),
        }
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return {"buggy_line": 0, "bug_type": "other", "explanation": "llm_error"}

# ---------------------------------------------------------------------------
# Environment calls
# ---------------------------------------------------------------------------

async def env_reset(task_name: str) -> dict:
    import httpx
    async with httpx.AsyncClient() as http:
        resp = await http.post(f"{ENV_URL}/reset", json={"task_name": task_name}, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


async def env_step(action: dict) -> dict:
    import httpx
    async with httpx.AsyncClient() as http:
        resp = await http.post(f"{ENV_URL}/step", json={"action": action}, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task_name: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    success     = False
    error_msg: Optional[str] = None
    feedback    = ""

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs_data     = await env_reset(task_name)
        code_snippet = obs_data.get("code_snippet", "")
        instructions = obs_data.get("instructions", "")
        done         = obs_data.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_dict = get_agent_action(client, code_snippet, instructions, feedback)
            result      = await env_step(action_dict)

            # Clamp reward immediately on receipt
            reward   = _clamp(float(result.get("reward", 0.05)))
            done     = bool(result.get("done", True))
            obs      = result.get("observation", {})
            feedback = obs.get("feedback", "")

            rewards.append(reward)
            steps_taken = step

            action_str = (
                f"line={action_dict['buggy_line']},"
                f"type={action_dict['bug_type']},"
                f"expl={action_dict['explanation'][:30].replace(' ','_')}"
            )

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        final_score = _clamp(max(rewards)) if rewards else 0.05
        success     = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        error_msg = str(exc)[:100]
        print(f"[DEBUG] Task error in {task_name}: {exc}", flush=True)
        if not rewards:
            rewards     = [0.05]
            steps_taken = 1
        success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for task in TASKS:
        try:
            await run_task(client, task)
        except Exception as exc:
            print(f"[DEBUG] Unhandled error in {task}: {exc}", flush=True)
            log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=1, rewards=[0.05])


if __name__ == "__main__":
    asyncio.run(main())
