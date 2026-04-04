"""
inference.py — Bug Review Environment Baseline Agent
=====================================================
Runs an LLM agent against all 3 bug review tasks and emits structured logs.

Environment variables:
    API_BASE_URL        LLM endpoint (default: HuggingFace router)
    MODEL_NAME          Model to use  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN            Your HuggingFace token (required)
    LOCAL_IMAGE_NAME    Docker image name if using from_docker_image()
    BUG_REVIEW_ENV_URL  URL of a live HF Space (used if set, else Docker)

STDOUT FORMAT (mandatory — do not change):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
IMAGE_NAME    = os.getenv("LOCAL_IMAGE_NAME", "bug-review-env:latest")
ENV_URL       = os.getenv("BUG_REVIEW_ENV_URL", "")   # live HF Space URL if available

BENCHMARK     = "bug-review-env"
MAX_STEPS     = 1          # Each task is single-step (one action per episode)
TEMPERATURE   = 0.2        # Low temperature for deterministic reasoning
MAX_TOKENS    = 400
SUCCESS_SCORE_THRESHOLD = 0.5   # score >= 0.5 → success

TASKS = ["find_bug_easy", "find_bug_medium", "find_bug_hard"]

# ---------------------------------------------------------------------------
# Logging helpers — MANDATORY FORMAT (do not modify field names or order)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# System prompt for the LLM agent
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

    You MUST respond with ONLY a valid JSON object, no other text:
    {
      "buggy_line": <integer>,
      "bug_type": "<one of the types above>",
      "explanation": "<your explanation>"
    }
""").strip()


def build_user_prompt(code_snippet: str, instructions: str) -> str:
    return textwrap.dedent(f"""
        INSTRUCTIONS:
        {instructions}

        CODE SNIPPET (lines are numbered from 1):
        ```python
        {code_snippet}
        ```

        Respond with ONLY a JSON object with keys: buggy_line, bug_type, explanation.
    """).strip()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_agent_action(
    client: OpenAI,
    code_snippet: str,
    instructions: str,
) -> dict:
    """Call the LLM and parse the JSON action. Falls back to a default on failure."""
    user_prompt = build_user_prompt(code_snippet, instructions)
    try:
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
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        return {
            "buggy_line":  int(parsed.get("buggy_line", 0)),
            "bug_type":    str(parsed.get("bug_type", "other")),
            "explanation": str(parsed.get("explanation", "")),
        }
    except Exception as exc:
        print(f"[DEBUG] LLM parse error: {exc}", flush=True)
        return {"buggy_line": 0, "bug_type": "other", "explanation": "parse error"}


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

async def run_task(env, task_name: str, client: OpenAI) -> None:
    """Run a single task episode and emit START / STEP / END logs."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset
        obs = await env.reset(task_name=task_name)

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            # Get action from LLM
            action_dict = get_agent_action(client, obs.code_snippet, obs.instructions)

            # Import here to avoid circular import at module level
            from bug_review_env import BugReviewAction
            action = BugReviewAction(**action_dict)

            # Step through environment
            result = await env.step(action)
            obs    = result.observation
            reward = result.reward
            done   = result.done

            rewards.append(reward)
            steps_taken = step

            # Format action string for log (compact, single-line)
            action_str = (
                f"line={action.buggy_line},"
                f"type={action.bug_type},"
                f"expl={action.explanation[:40].replace(' ', '_')}"
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

        score   = rewards[-1] if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        error_msg = str(exc)
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        if not rewards:
            rewards = [0.0]
            steps_taken = 1
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to environment
    if ENV_URL:
        # Use live HF Space
        from bug_review_env import BugReviewEnv
        env = BugReviewEnv(base_url=ENV_URL)
        async with env:
            for task in TASKS:
                await run_task(env, task, client)
    else:
        # Use local Docker image
        from bug_review_env import BugReviewEnv
        env = await BugReviewEnv.from_docker_image(IMAGE_NAME)
        try:
            for task in TASKS:
                await run_task(env, task, client)
        finally:
            await env.close()


if __name__ == "__main__":
    asyncio.run(main())
