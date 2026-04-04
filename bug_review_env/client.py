"""
BugReviewEnv — OpenEnv HTTP client.

Usage (sync):
    from bug_review_env import BugReviewAction, BugReviewEnv

    with BugReviewEnv(base_url="https://Dan2000000-bug-review-env.hf.space").sync() as env:
        obs = env.reset(task_name="find_bug_easy")
        result = env.step(BugReviewAction(
            buggy_line=3,
            bug_type="off_by_one",
            explanation="Should be items[len(items)-1]"
        ))
        print(result.reward)

Usage (async):
    from bug_review_env import BugReviewAction, BugReviewEnv
    import asyncio

    async def main():
        async with BugReviewEnv(base_url="https://Dan2000000-bug-review-env.hf.space") as env:
            obs = await env.reset(task_name="find_bug_easy")
            result = await env.step(BugReviewAction(
                buggy_line=3,
                bug_type="off_by_one",
                explanation="Should be items[len(items)-1]"
            ))
            print(result.reward)

    asyncio.run(main())
"""

import asyncio
from typing import Optional
import httpx

from .models import BugReviewAction, BugReviewObservation, BugReviewState


class StepResult:
    def __init__(self, observation: BugReviewObservation, reward: float, done: bool, info: dict):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


class _SyncWrapper:
    """Synchronous context manager wrapper around BugReviewEnv."""

    def __init__(self, env: "BugReviewEnv"):
        self._env = env

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def reset(self, task_name: str = "find_bug_easy") -> BugReviewObservation:
        return asyncio.get_event_loop().run_until_complete(
            self._env.reset(task_name=task_name)
        )

    def step(self, action: BugReviewAction) -> StepResult:
        return asyncio.get_event_loop().run_until_complete(self._env.step(action))

    def state(self) -> BugReviewState:
        return asyncio.get_event_loop().run_until_complete(self._env.state())

    def close(self):
        asyncio.get_event_loop().run_until_complete(self._env.close())


class BugReviewEnv:
    """Async OpenEnv client for the Bug Review Environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    def sync(self) -> _SyncWrapper:
        """Return a synchronous wrapper for use in non-async code."""
        return _SyncWrapper(self)

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        return self._client

    async def reset(self, task_name: str = "find_bug_easy") -> BugReviewObservation:
        client = await self._get_client()
        response = await client.post("/reset", json={"task_name": task_name})
        response.raise_for_status()
        return BugReviewObservation(**response.json())

    async def step(self, action: BugReviewAction) -> StepResult:
        client = await self._get_client()
        response = await client.post("/step", json={"action": action.model_dump()})
        response.raise_for_status()
        data = response.json()
        return StepResult(
            observation=BugReviewObservation(**data["observation"]),
            reward=data["reward"],
            done=data["done"],
            info=data.get("info", {}),
        )

    async def state(self) -> BugReviewState:
        client = await self._get_client()
        response = await client.get("/state")
        response.raise_for_status()
        return BugReviewState(**response.json())

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    @classmethod
    async def from_docker_image(cls, image_name: str, port: int = 7860) -> "BugReviewEnv":
        """
        Start a local Docker container from image_name and return a connected client.
        Requires Docker to be running locally.
        """
        import subprocess, time

        container_name = "bug-review-env-local"
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
        )
        subprocess.Popen([
            "docker", "run", "--rm", "--name", container_name,
            "-p", f"{port}:7860", image_name
        ])
        # Wait for server to be ready
        import httpx as _httpx
        for _ in range(30):
            try:
                r = _httpx.get(f"http://localhost:{port}/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)

        return cls(base_url=f"http://localhost:{port}")
