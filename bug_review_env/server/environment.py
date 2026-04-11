"""
Bug Review Environment — core logic (multi-step version).

Three tasks of increasing difficulty:
  easy   → obvious off-by-one error
  medium → SQL injection / hardcoded secret
  hard   → race condition

Multi-step episodes:
  - Agent gets up to MAX_ATTEMPTS tries per task
  - After each attempt, detailed feedback is returned
  - Reward increases as agent gets closer to correct answer
  - Final reward = best score achieved across all attempts
  - Partial credit awarded per component (line, type, explanation)
  - Penalty for repeated identical wrong answers

All scores are strictly between 0 and 1 (never exactly 0.0 or 1.0).
"""

import uuid
from typing import Dict, Tuple, List

from bug_review_env.models import BugReviewAction, BugReviewObservation, BugReviewState

# ---------------------------------------------------------------------------
# Score clamping — scores must be STRICTLY between 0 and 1
# ---------------------------------------------------------------------------

def _clamp(score: float) -> float:
    """Ensure score is strictly inside (0, 1) — never exactly 0.0 or 1.0."""
    return round(max(0.01, min(0.99, score)), 2)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict] = {
    "find_bug_easy": {
        "display_name": "find_bug_easy",
        "code_snippet": """\
def get_last_element(items):
    \"\"\"Return the last element of a list.\"\"\"
    return items[len(items)]   # line 3

def compute_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)  # line 9
""",
        "instructions": (
            "The function `get_last_element` contains a classic indexing bug. "
            "Identify the line number where the bug occurs, classify the bug type "
            "(off_by_one | null_dereference | sql_injection | hardcoded_secret | "
            "race_condition | logic_error | insecure_deserialization | other), "
            "and explain how to fix it. You have up to 3 attempts — use the "
            "feedback from each attempt to improve your answer."
        ),
        "answer": {
            "buggy_line": 3,
            "bug_type": "off_by_one",
            "key_terms": ["len(items) - 1", "len(items)-1", "-1", "index out", "off by one", "off-by-one"],
        },
        "max_attempts": 3,
    },

    "find_bug_medium": {
        "display_name": "find_bug_medium",
        "code_snippet": """\
import sqlite3

def get_user(username):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '" + username + "'"  # line 6
    cursor.execute(query)
    return cursor.fetchone()

def login(username, password):
    DB_PASSWORD = "s3cr3t_admin_pass"   # line 11
    user = get_user(username)
    if user and user[2] == password:
        return True
    return False
""",
        "instructions": (
            "This code contains TWO security vulnerabilities. "
            "Identify the most critical one: its line number, bug type, and explanation. "
            "Bug types: sql_injection | hardcoded_secret | logic_error. "
            "You have up to 3 attempts — use the feedback to refine your answer."
        ),
        "answer": {
            "buggy_line": 6,
            "alt_buggy_line": 11,
            "bug_type": "sql_injection",
            "alt_bug_type": "hardcoded_secret",
            "key_terms": [
                "sql injection", "sql_injection", "parameterized", "parameterise",
                "placeholder", "? ", "hardcoded", "hard-coded", "secret", "password"
            ],
        },
        "max_attempts": 3,
    },

    "find_bug_hard": {
        "display_name": "find_bug_hard",
        "code_snippet": """\
import threading

balance = 1000

def withdraw(amount):
    global balance
    if balance >= amount:          # line 7  — check
        balance -= amount          # line 8  — act
        return True
    return False

def run_concurrent_withdrawals():
    threads = [threading.Thread(target=withdraw, args=(600,)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("Final balance:", balance)   # can go negative!
""",
        "instructions": (
            "This multi-threaded banking code has a subtle concurrency bug. "
            "Identify the line number, classify as race_condition, and explain "
            "why two concurrent withdrawals of 600 can each succeed when balance "
            "is only 1000. You have up to 3 attempts — use feedback to improve."
        ),
        "answer": {
            "buggy_line": 7,
            "alt_buggy_line": 8,
            "bug_type": "race_condition",
            "key_terms": [
                "race condition", "race_condition", "thread", "concurrent", "atomic",
                "lock", "mutex", "synchroni", "check-then-act", "toctou",
                "time of check", "non-atomic"
            ],
        },
        "max_attempts": 3,
    },
}

TASK_ORDER = ["find_bug_easy", "find_bug_medium", "find_bug_hard"]
MAX_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# Grader functions — return (score, feedback, component_scores)
# ---------------------------------------------------------------------------

def _grade(action: BugReviewAction, answer: dict, weights: dict) -> Tuple[float, str, dict]:
    score = 0.0
    parts = []
    components = {"line": False, "type": False, "explanation": False}

    # Line number
    correct_line = action.buggy_line == answer.get("buggy_line")
    if not correct_line and "alt_buggy_line" in answer:
        correct_line = action.buggy_line == answer["alt_buggy_line"]
    if correct_line:
        score += weights["line"]
        components["line"] = True
        parts.append(f"✓ Correct line number (+{weights['line']:.2f})")
    else:
        expected = answer.get("buggy_line")
        hint = "Hint: look between lines 1-10." if expected <= 10 else "Hint: look carefully at the logic flow."
        parts.append(f"✗ Wrong line. {hint} (+0.00)")

    # Bug type
    correct_type = action.bug_type == answer.get("bug_type")
    if not correct_type and "alt_bug_type" in answer:
        correct_type = action.bug_type == answer["alt_bug_type"]
    if correct_type:
        score += weights["type"]
        components["type"] = True
        parts.append(f"✓ Correct bug type (+{weights['type']:.2f})")
    else:
        valid_types = "off_by_one|null_dereference|sql_injection|hardcoded_secret|race_condition|logic_error|insecure_deserialization|other"
        parts.append(f"✗ Wrong bug type '{action.bug_type}'. Consider: {valid_types} (+0.00)")

    # Explanation key terms
    expl_lower = action.explanation.lower()
    matching = [t for t in answer.get("key_terms", []) if t in expl_lower]
    if len(matching) >= 2:
        score += weights["expl_full"]
        components["explanation"] = True
        parts.append(f"✓ Explanation demonstrates understanding (+{weights['expl_full']:.2f})")
    elif len(matching) == 1:
        score += weights["expl_partial"]
        parts.append(f"~ Explanation partially correct — be more specific (+{weights['expl_partial']:.2f})")
    else:
        parts.append("✗ Explanation missing key concepts — think about the root cause (+0.00)")

    # Clamp so intermediate scores are always safe
    score = _clamp(score)
    feedback = " | ".join(parts) + f" → Score: {score:.2f}"
    return score, feedback, components


# Key change: weights now sum to 0.99 max so perfect score = 0.99 (never 1.0)
GRADER_WEIGHTS = {
    "find_bug_easy":   {"line": 0.40, "type": 0.30, "expl_full": 0.29, "expl_partial": 0.14},
    "find_bug_medium": {"line": 0.35, "type": 0.35, "expl_full": 0.29, "expl_partial": 0.14},
    "find_bug_hard":   {"line": 0.30, "type": 0.30, "expl_full": 0.39, "expl_partial": 0.19},
}

# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class BugReviewEnvironment:
    """
    Multi-step OpenEnv-compatible environment for code bug review.

    Episodes allow up to MAX_ATTEMPTS actions per task.
    After each attempt, detailed feedback guides the agent.
    The episode ends when:
      - Agent achieves score >= 0.99 (near-perfect)
      - Agent exhausts all attempts
    Final reward = best score achieved across all attempts.
    All scores are strictly between 0.01 and 0.99.
    """

    def __init__(self):
        self._state = BugReviewState()
        self._current_task: str = TASK_ORDER[0]
        self._best_score: float = 0.01
        self._attempts: int = 0
        self._max_attempts: int = MAX_ATTEMPTS
        self._last_action: dict = {}
        self._all_rewards: List[float] = []

    def reset(self, task_name: str = "find_bug_easy") -> BugReviewObservation:
        """Start a new episode for the given task."""
        if task_name not in TASKS:
            task_name = TASK_ORDER[0]

        self._current_task = task_name
        self._best_score = 0.01
        self._attempts = 0
        self._max_attempts = TASKS[task_name]["max_attempts"]
        self._last_action = {}
        self._all_rewards = []

        self._state = BugReviewState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
            attempts=0,
            last_score=0.01,
        )

        task = TASKS[task_name]
        return BugReviewObservation(
            code_snippet=task["code_snippet"],
            task_name=task["display_name"],
            instructions=task["instructions"],
            feedback="",
            done=False,
        )

    def step(self, action: BugReviewAction) -> Tuple[BugReviewObservation, float, bool]:
        """
        Execute one step: grade the action, return (observation, reward, done).

        Reward shaping:
          - Full credit for each correct component
          - Partial credit for partial explanation
          - Small penalty (-0.05) for repeating exact same wrong answer
          - Episode ends when near-perfect score OR attempts exhausted
          - All rewards strictly between 0.01 and 0.99
        """
        self._state.step_count += 1
        self._attempts += 1
        self._state.attempts = self._attempts

        task = TASKS[self._current_task]
        answer = task["answer"]
        weights = GRADER_WEIGHTS[self._current_task]

        score, feedback, components = _grade(action, answer, weights)

        # Penalty for repeating identical wrong answer
        current_action = {
            "buggy_line": action.buggy_line,
            "bug_type": action.bug_type,
        }
        if self._last_action == current_action and score < 0.99:
            score = _clamp(score - 0.05)
            feedback += " | ⚠ Penalty: repeated same answer (-0.05)"

        self._last_action = current_action
        self._all_rewards.append(score)

        # Update best score
        if score > self._best_score:
            self._best_score = score

        self._state.last_score = score

        # Determine if episode is done (0.99 = near-perfect threshold)
        attempts_left = self._max_attempts - self._attempts
        perfect = score >= 0.99
        exhausted = attempts_left <= 0
        done = perfect or exhausted

        # Build feedback message
        if not done:
            feedback += f" | Attempts remaining: {attempts_left}"
        elif perfect:
            feedback += " | 🎉 Excellent answer! Episode complete."
        else:
            feedback += f" | Episode complete. Best score: {self._best_score:.2f}"

        obs = BugReviewObservation(
            code_snippet=task["code_snippet"],
            task_name=task["display_name"],
            instructions=task["instructions"],
            feedback=feedback,
            done=done,
        )

        # Always clamp the final reward too
        reward = _clamp(self._best_score) if done else score
        return obs, reward, done

    @property
    def state(self) -> BugReviewState:
        return self._state

    @property
    def all_rewards(self) -> List[float]:
        return self._all_rewards
