"""
inference.py — Bug Review Environment Baseline Agent
=====================================================
Strictly follows Meta x Scaler OpenEnv Hackathon guidelines.
Supports multi-step episodes (up to 3 attempts per task).

Environment variables:
    API_BASE_URL   API endpoint for the LLM (default provided)
    MODEL_NAME     Model identifier (default provided)
    HF_TOKEN       Hugging Face API token (mandatory, no default)

STDOUT FORMAT (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
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

BENCHMARK   = "bug-review-env"
MAX_STEPS   = 3        # multi-step: up to 3 attempts per task
TEMPERATURE = 0.2
MAX_TOKENS  = 400
SUCCESS_SCORE_THRESHOLD = 0.5

# Fallback reward — strictly between 0 and 1, never exactly 0.0 or 1.0
FALLBACK_REWARD = 0.01

TASKS = ["find_bug_easy", "find_bug_medium", "find_bug_hard"]

# ---------------------------------------------------------------------------
# Logging helpers — MANDATORY FORMAT
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val    = error if error else "null"
    done_val     = str(done).lower()
    action_clean = action.replace(" ", "_").replace("\n", "").replace("\r", "")[:80]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else f"{FALLBACK_REWARD:.2f}"
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Score clamping — all rewards must be strictly between 0 and 1
# ---------------------------------------------------------------------------

def _clamp(score: float) -> float:
    """Ensure score is strictly inside (0, 1) — never exactly 0.0 or 1.0."""
    return round(max(0.01, min(0.99, score)), 2)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert software security engineer performing code review.
    You will be shown a Python code snippet that contains a bug.
    Your job is to identify:
      1. The exact line number (1-indexed) where the bug occurs
      2. The bug type — one of:
           off_by_one | null_dereference | sql_injection | hardcoded_secret |
           race_condition | logic_error | insecure_deserialization | other
      3. A clear explanation of the bug and how to fix it

    If you receive feedback from a previous attempt, use it to improve your answer.

    You MUST respond with ONLY a valid JSON object, no other text:
    {
      "buggy_line": <integer>,
      "bug_type": "<one of the types above>",
      "explanation": "<your explanation>"
    }
""").strip()


def build_user_prompt(code_snippet: str, instructions: str, feedback: str = "") -> str:
    prompt = (
        f"INSTRUCTIONS:\n{instructions}\n\n"
        f"CODE SNIPPET (lines numbered from 1):\n"
        f"```python\n{code_snippet}\n```\n"
    )
    if feedback:
        prompt += f"\nFEEDBACK FROM PREVIOUS ATTEMPT:\n{feedback}\n"
    prompt += "\nRespond with ONLY a JSON object with keys: buggy_line, bug_type, explanation."
    return prompt

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_agent_action(client: OpenAI, code_snippet: str, instructions: str, feedback: str = "") -> dict:
    """Call the LLM and parse JSON action. Returns safe default on failure."""
    try:
        user_prompt = build_user_prompt(code_snippet, instructions, feedback)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if present
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
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
        resp = await http.post(
            f"{ENV_URL}/reset",
            json={"task_name": task_name},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()


async def env_step(action: dict) -> dict:
    import httpx
    async with httpx.AsyncClient() as http:
        resp = await http.post(
            f"{ENV_URL}/step",
            json={"action": action},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()

# ---------------------------------------------------------------------------
# Run one task episode (multi-step)
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task_name: str) -> None:
    """Run a multi-step task episode and emit START / STEP / END logs."""
    rewards: List[float] = []
    steps_taken = 0
    success     = False
    error_msg: Optional[str] = None
    feedback    = ""

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        obs_data     = await env_reset(task_name)
        code_snippet = obs_data.get("code_snippet", "")
        instructions = obs_data.get("instructions", "")
        done         = obs_data.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Get action from LLM — pass feedback from previous step
            action_dict = get_agent_action(client, code_snippet, instructions, feedback)

            # Step through environment
            result   = await env_step(action_dict)
            reward   = _clamp(float(result.get("reward", FALLBACK_REWARD)))
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

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

        final_score = _clamp(max(rewards)) if rewards else FALLBACK_REWARD
        success     = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        error_msg = str(exc)[:100]
        print(f"[DEBUG] Task error in {task_name}: {exc}", flush=True)
        if not rewards:
            rewards     = [FALLBACK_REWARD]
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
            print(f"[DEBUG] Unhandled error in task {task}: {exc}", flush=True)
            log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=1, rewards=[FALLBACK_REWARD])


if __name__ == "__main__":
    asyncio.run(main())
