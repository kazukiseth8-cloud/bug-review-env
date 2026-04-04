"""
Bug Review Environment — core logic.

Three tasks of increasing difficulty:
  easy   → obvious off-by-one error
  medium → hardcoded secret / SQL injection
  hard   → subtle race condition

Each task has a deterministic grader that scores 0.0–1.0.
Partial credit is awarded for getting the line right, bug type right,
or explanation that contains key terms — even if not all three.
"""

import uuid
from typing import Dict, Tuple

from bug_review_env.models import BugReviewAction, BugReviewObservation, BugReviewState

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
            "Identify the line number where the bug occurs, classify the bug type, "
            "and explain how to fix it."
        ),
        "answer": {
            "buggy_line": 3,
            "bug_type": "off_by_one",
            "key_terms": ["len(items) - 1", "len(items)-1", "-1", "index out", "off by one", "off-by-one"],
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
            "This code contains TWO security vulnerabilities. "
            "Identify the most critical one: its line number, bug type, and explanation. "
            "Bug types to consider: sql_injection, hardcoded_secret, logic_error."
        ),
        "answer": {
            "buggy_line": 6,           # primary: SQL injection
            "alt_buggy_line": 11,      # secondary: hardcoded secret — also accepted
            "bug_type": "sql_injection",
            "alt_bug_type": "hardcoded_secret",
            "key_terms": [
                "sql injection", "sql_injection", "parameterized", "parameterise",
                "placeholder", "? ", "hardcoded", "hard-coded", "secret", "password"
            ],
        },
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
            "Identify the line number where the bug manifests, classify the bug type "
            "as race_condition, and explain why two concurrent withdrawals of 600 "
            "can each succeed even when the balance is only 1000."
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
    },
}

TASK_ORDER = ["find_bug_easy", "find_bug_medium", "find_bug_hard"]

# ---------------------------------------------------------------------------
# Grader functions
# ---------------------------------------------------------------------------

def _grade_easy(action: BugReviewAction) -> Tuple[float, str]:
    answer = TASKS["find_bug_easy"]["answer"]
    score = 0.0
    parts = []

    # Line number: 0.4 points
    if action.buggy_line == answer["buggy_line"]:
        score += 0.4
        parts.append("✓ Correct line number (+0.40)")
    else:
        parts.append(f"✗ Wrong line. Expected {answer['buggy_line']}, got {action.buggy_line} (+0.00)")

    # Bug type: 0.3 points
    if action.bug_type == answer["bug_type"]:
        score += 0.3
        parts.append("✓ Correct bug type (+0.30)")
    else:
        parts.append(f"✗ Wrong bug type. Expected '{answer['bug_type']}', got '{action.bug_type}' (+0.00)")

    # Explanation key terms: 0.3 points
    expl_lower = action.explanation.lower()
    if any(term in expl_lower for term in answer["key_terms"]):
        score += 0.3
        parts.append("✓ Explanation contains correct fix (+0.30)")
    else:
        parts.append("✗ Explanation missing key fix details (+0.00)")

    feedback = " | ".join(parts) + f" → Total: {score:.2f}"
    return round(score, 2), feedback


def _grade_medium(action: BugReviewAction) -> Tuple[float, str]:
    answer = TASKS["find_bug_medium"]["answer"]
    score = 0.0
    parts = []

    # Line: either primary (6) or secondary (11) accepted
    correct_line = (
        action.buggy_line == answer["buggy_line"] or
        action.buggy_line == answer["alt_buggy_line"]
    )
    if correct_line:
        score += 0.35
        parts.append(f"✓ Correct line number (+0.35)")
    else:
        parts.append(f"✗ Wrong line. Expected {answer['buggy_line']} or {answer['alt_buggy_line']} (+0.00)")

    # Bug type: either sql_injection or hardcoded_secret accepted
    correct_type = (
        action.bug_type == answer["bug_type"] or
        action.bug_type == answer["alt_bug_type"]
    )
    if correct_type:
        score += 0.35
        parts.append("✓ Correct bug type (+0.35)")
    else:
        parts.append(f"✗ Wrong bug type (+0.00)")

    # Explanation key terms
    expl_lower = action.explanation.lower()
    if any(term in expl_lower for term in answer["key_terms"]):
        score += 0.30
        parts.append("✓ Explanation identifies the vulnerability (+0.30)")
    else:
        parts.append("✗ Explanation missing key security terms (+0.00)")

    feedback = " | ".join(parts) + f" → Total: {score:.2f}"
    return round(score, 2), feedback


def _grade_hard(action: BugReviewAction) -> Tuple[float, str]:
    answer = TASKS["find_bug_hard"]["answer"]
    score = 0.0
    parts = []

    # Line: line 7 or 8 both valid
    correct_line = (
        action.buggy_line == answer["buggy_line"] or
        action.buggy_line == answer["alt_buggy_line"]
    )
    if correct_line:
        score += 0.30
        parts.append("✓ Correct line number (+0.30)")
    else:
        parts.append(f"✗ Wrong line. Expected 7 or 8 (+0.00)")

    # Bug type: must be race_condition
    if action.bug_type == answer["bug_type"]:
        score += 0.30
        parts.append("✓ Correct bug type: race_condition (+0.30)")
    else:
        parts.append(f"✗ Wrong bug type. Expected 'race_condition', got '{action.bug_type}' (+0.00)")

    # Explanation: must mention concurrency/threading concepts
    expl_lower = action.explanation.lower()
    matching = [term for term in answer["key_terms"] if term in expl_lower]
    if len(matching) >= 2:
        score += 0.40
        parts.append("✓ Explanation demonstrates deep understanding (+0.40)")
    elif len(matching) == 1:
        score += 0.20
        parts.append("✓ Explanation partially correct (+0.20)")
    else:
        parts.append("✗ Explanation missing concurrency concepts (+0.00)")

    feedback = " | ".join(parts) + f" → Total: {score:.2f}"
    return round(score, 2), feedback


GRADERS = {
    "find_bug_easy": _grade_easy,
    "find_bug_medium": _grade_medium,
    "find_bug_hard": _grade_hard,
}

# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class BugReviewEnvironment:
    """
    OpenEnv-compatible environment for code bug review tasks.

    Episodes run one task at a time. The agent submits a BugReviewAction
    (line number + bug type + explanation). The grader scores it 0.0–1.0.
    The episode ends after the first action (done=True), giving a single
    clean reward signal. Callers may loop over all three tasks manually.
    """

    def __init__(self):
        self._state = BugReviewState()
        self._current_task: str = TASK_ORDER[0]

    def reset(self, task_name: str = "find_bug_easy") -> BugReviewObservation:
        """Start a new episode for the given task."""
        if task_name not in TASKS:
            task_name = TASK_ORDER[0]

        self._current_task = task_name
        self._state = BugReviewState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
            attempts=0,
            last_score=0.0,
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
        The episode always ends after one action (done=True).
        """
        self._state.step_count += 1
        self._state.attempts += 1

        task = TASKS[self._current_task]
        grader = GRADERS[self._current_task]
        score, feedback = grader(action)

        self._state.last_score = score

        obs = BugReviewObservation(
            code_snippet=task["code_snippet"],
            task_name=task["display_name"],
            instructions=task["instructions"],
            feedback=feedback,
            done=True,
        )
        return obs, score, True   # done=True after first action

    @property
    def state(self) -> BugReviewState:
        return self._state
