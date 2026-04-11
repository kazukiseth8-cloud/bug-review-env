"""
Bug Review Environment — core logic.
Scores are STRICTLY between 0 and 1 (exclusive) — never 0.0, never 1.0.

Grader weights are designed so max raw score = 0.94 (never reaches 1.0).
_clamp() ensures minimum score = 0.05 (never reaches 0.0).
"""

import uuid
from typing import Dict, Tuple, List

from bug_review_env.models import BugReviewAction, BugReviewObservation, BugReviewState

# ---------------------------------------------------------------------------
# Score clamping — strictly (0, 1) exclusive
# ---------------------------------------------------------------------------

def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1, never equal to either."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.05
    if s != s:  # NaN check
        return 0.05
    if s <= 0.0:
        return 0.05
    if s >= 1.0:
        return 0.95
    return round(s, 4)

# ---------------------------------------------------------------------------
# Task definitions
# Weights per task sum to 0.94 max — never reaches 1.0
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
            "and explain how to fix it. You have up to 3 attempts."
        ),
        "answer": {
            "buggy_line": 3,
            "bug_type": "off_by_one",
            "key_terms": [
                "len(items) - 1", "len(items)-1", "-1",
                "index out", "off by one", "off-by-one"
            ],
        },
        "max_attempts": 3,
        # line + type + expl_full = 0.38 + 0.28 + 0.28 = 0.94 max
        "weights": {
            "line": 0.38,
            "type": 0.28,
            "expl_full": 0.28,
            "expl_partial": 0.14,
        },
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
            "This code contains security vulnerabilities. "
            "Identify the most critical one: its line number, bug type, and explanation. "
            "Bug types: sql_injection | hardcoded_secret | logic_error. "
            "You have up to 3 attempts."
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
        # line + type + expl_full = 0.33 + 0.33 + 0.28 = 0.94 max
        "weights": {
            "line": 0.33,
            "type": 0.33,
            "expl_full": 0.28,
            "expl_partial": 0.14,
        },
    },

    "find_bug_hard": {
        "display_name": "find_bug_hard",
        "code_snippet": """\
import threading

balance = 1000

def withdraw(amount):
    global balance
    if balance >= amount:          # line 7
        balance -= amount          # line 8
        return True
    return False

def run_concurrent_withdrawals():
    threads = [threading.Thread(target=withdraw, args=(600,)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("Final balance:", balance)
""",
        "instructions": (
            "This multi-threaded banking code has a subtle concurrency bug. "
            "Identify the line number, classify as race_condition, and explain "
            "why two concurrent withdrawals of 600 can each succeed when balance "
            "is only 1000. You have up to 3 attempts."
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
        # line + type + expl_full = 0.28 + 0.28 + 0.38 = 0.94 max
        "weights": {
            "line": 0.28,
            "type": 0.28,
            "expl_full": 0.38,
            "expl_partial": 0.19,
        },
    },
}

TASK_ORDER = ["find_bug_easy", "find_bug_medium", "find_bug_hard"]

# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def _grade(action: BugReviewAction, answer: dict, weights: dict) -> Tuple[float, str]:
    """
    Grade an action. Returns score strictly in (0, 1).
    Max raw = 0.94, min raw = 0.0 → clamped to 0.05.
    """
    raw_score = 0.0
    parts = []

    # Line number
    correct_line = action.buggy_line == answer.get("buggy_line")
    if not correct_line and "alt_buggy_line" in answer:
        correct_line = action.buggy_line == answer["alt_buggy_line"]
    if correct_line:
        raw_score += weights["line"]
        parts.append(f"✓ Correct line (+{weights['line']:.2f})")
    else:
        expected = answer.get("buggy_line")
        parts.append(f"✗ Wrong line (expected {expected}) (+0.00)")

    # Bug type
    correct_type = action.bug_type == answer.get("bug_type")
    if not correct_type and "alt_bug_type" in answer:
        correct_type = action.bug_type == answer["alt_bug_type"]
    if correct_type:
        raw_score += weights["type"]
        parts.append(f"✓ Correct bug type (+{weights['type']:.2f})")
    else:
        parts.append(f"✗ Wrong bug type '{action.bug_type}' (+0.00)")

    # Explanation key terms
    expl_lower = action.explanation.lower()
    matching = [t for t in answer.get("key_terms", []) if t in expl_lower]
    if len(matching) >= 2:
        raw_score += weights["expl_full"]
        parts.append(f"✓ Good explanation (+{weights['expl_full']:.2f})")
    elif len(matching) == 1:
        raw_score += weights["expl_partial"]
        parts.append(f"~ Partial explanation (+{weights['expl_partial']:.2f})")
    else:
        parts.append("✗ Explanation needs more detail (+0.00)")

    # Always clamp — converts 0.0 → 0.05, max 0.94 stays below 0.95
    score = _clamp(raw_score)
    feedback = " | ".join(parts) + f" → Score: {score:.4f}"
    return score, feedback

# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class BugReviewEnvironment:
    """
    Multi-step OpenEnv environment for code bug review.
    All scores strictly in (0, 1) — never 0.0, never 1.0.
    """

    def __init__(self):
        self._state = BugReviewState()
        self._current_task: str = TASK_ORDER[0]
        self._best_score: float = 0.05
        self._attempts: int = 0
        self._max_attempts: int = 3
        self._last_action: dict = {}

    def reset(self, task_name: str = "find_bug_easy") -> BugReviewObservation:
        """Start a new episode. Returns initial observation."""
        if task_name not in TASKS:
            task_name = TASK_ORDER[0]

        self._current_task = task_name
        self._best_score = 0.05
        self._attempts = 0
        self._max_attempts = TASKS[task_name]["max_attempts"]
        self._last_action = {}

        self._state = BugReviewState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
            attempts=0,
            last_score=0.05,
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
        Grade the action and return (observation, reward, done).
        reward is always strictly in (0, 1).
        """
        # Safety: if no task loaded, return safe defaults
        if self._current_task not in TASKS:
            return BugReviewObservation(
                code_snippet="", task_name="", instructions="",
                feedback="No active task", done=True
            ), 0.05, True
        self._state.step_count += 1
        self._attempts += 1
        self._state.attempts = self._attempts

        task = TASKS[self._current_task]
        answer = task["answer"]
        weights = task["weights"]

        score, feedback = _grade(action, answer, weights)

        # Small penalty for repeating exact same wrong answer
        current_action = {
            "buggy_line": action.buggy_line,
            "bug_type": action.bug_type,
        }
        if self._last_action == current_action and score < 0.90:
            score = _clamp(score - 0.04)
            feedback += " | ⚠ Penalty: repeated same answer"
        self._last_action = current_action

        # Track best score
        if score > self._best_score:
            self._best_score = score
        self._best_score = _clamp(self._best_score)
        self._state.last_score = score

        # Episode ends when near-perfect OR attempts exhausted
        attempts_left = self._max_attempts - self._attempts
        done = (score >= 0.90) or (attempts_left <= 0)

        if not done:
            feedback += f" | Attempts remaining: {attempts_left}"
        elif score >= 0.90:
            feedback += " | 🎉 Excellent work!"
        else:
            feedback += f" | Episode complete. Best: {self._best_score:.4f}"

        # Reward: best score on final step, current otherwise — always clamped
        reward = _clamp(self._best_score) if done else _clamp(score)

        obs = BugReviewObservation(
            code_snippet=task["code_snippet"],
            task_name=task["display_name"],
            instructions=task["instructions"],
            feedback=feedback,
            done=done,
        )
        return obs, reward, done

    @property
    def state(self) -> BugReviewState:
        return self._state
