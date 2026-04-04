"""
Pydantic models for the Bug Review Environment.
Defines Action, Observation, and State types per OpenEnv spec.
"""

from typing import List, Optional
from pydantic import BaseModel


class BugReviewAction(BaseModel):
    """
    Action submitted by the agent.

    Fields:
        buggy_line   - The line number (1-indexed) where the bug is located.
                       Use 0 if the agent cannot identify a specific line.
        bug_type     - Category of bug identified. Must be one of:
                       "off_by_one" | "null_dereference" | "sql_injection" |
                       "hardcoded_secret" | "race_condition" | "logic_error" |
                       "insecure_deserialization" | "other"
        explanation  - Free-text explanation of the bug and how to fix it.
    """
    buggy_line: int
    bug_type: str
    explanation: str


class BugReviewObservation(BaseModel):
    """
    Observation returned to the agent after reset() or step().

    Fields:
        code_snippet   - The code snippet containing a bug (Python).
        task_name      - Name of the current task (easy / medium / hard).
        instructions   - What the agent is asked to do.
        feedback       - Feedback from the last action (empty on reset).
        done           - Whether the episode has ended.
    """
    code_snippet: str
    task_name: str
    instructions: str
    feedback: str
    done: bool


class BugReviewState(BaseModel):
    """
    Internal episode state.

    Fields:
        episode_id     - Unique identifier for this episode.
        step_count     - Number of steps taken so far.
        task_name      - Current task name.
        attempts       - How many actions have been submitted.
        last_score     - Score from the last grader evaluation (0.0–1.0).
    """
    episode_id: str = ""
    step_count: int = 0
    task_name: str = ""
    attempts: int = 0
    last_score: float = 0.0
