import asyncio
import json
import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from request_guard import RequestGuard, GuardStopped, upstream_notice


class GuardChecks(unittest.TestCase):
    def test_exact_budget_and_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.json"
            guard = RequestGuard(8, path)
            for _ in range(8):
                guard.reserve()
            restored = RequestGuard(8, path)
            self.assertEqual(restored.count, 8)
            with self.assertRaises(GuardStopped):
                restored.reserve()
            self.assertEqual(RequestGuard(8, path).stopped, "request_budget_exhausted")

    def test_fail_closed_invalid_or_changed_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.json"
            path.write_text("bad JSON")
            with self.assertRaises(ValueError):
                RequestGuard(8, path)
            guard = RequestGuard(8, Path(folder) / "new.json")
            guard.reserve()
            with self.assertRaises(ValueError):
                RequestGuard(9, guard.ledger)

    def test_observation_only_ledger_counts_and_preserves_quota_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.json"
            guard = RequestGuard(None, path)
            for _ in range(9):
                guard.reserve()
            restored = RequestGuard(None, path)
            self.assertEqual(restored.count, 9)
            restored.stop("upstream_usage_limit")
            with self.assertRaises(GuardStopped):
                RequestGuard(None, path).reserve()

    def test_bounded_run_requires_ledger(self):
        with self.assertRaises(ValueError):
            RequestGuard(8)
        with self.assertRaises(ValueError):
            RequestGuard(0, "unused")

    def test_stop_persists_without_content(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.json"
            guard = RequestGuard(8, path)
            guard.stop("upstream_http_429")
            guard.stop("TimeoutError")
            restored = RequestGuard(8, path)
            with self.assertRaises(GuardStopped):
                restored.reserve()
            self.assertEqual(json.loads(path.read_text()),
                             {"limit": 8, "count": 0, "stopped": "upstream_http_429"})

    def test_quota_notices_and_ordinary_content(self):
        for text in ("You've reached your limit for Pro", "Too many requests",
                     "Quota exceeded", "已達使用上限", "Other UI text\nYou have reached your limit"):
            self.assertEqual(upstream_notice(text), "upstream_usage_limit")
        for text in ("Explain quota handling", "This code tests a rate limit", "Normal reply"):
            self.assertIsNone(upstream_notice(text))


class GuardRouteChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.guard = RequestGuard(8, Path(self.folder.name) / "ledger.json")
        self.patches = [patch.object(server, "request_guard", self.guard),
                        patch.object(server, "model_request_lock", asyncio.Lock())]
        for item in self.patches:
            item.__enter__()

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.__exit__(None, None, None)
        self.folder.cleanup()

    async def test_concurrent_sessions_are_serialized(self):
        active = 0
        peak = 0
        async def send(*args):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            self.guard.reserve()
            await asyncio.sleep(0.001)
            active -= 1
            return "OK"
        with patch.object(server, "guarded_chat_completions", send):
            await asyncio.gather(*(server.chat_completions(None) for _ in range(8)))
            with self.assertRaises(server.HTTPException):
                await server.chat_completions(None)
        self.assertEqual(peak, 1)
        self.assertEqual(self.guard.count, 8)

    async def test_failure_stops_followups_without_retry(self):
        for error in (TimeoutError(), server.HTTPException(429, "mock quota"),
                      server.HTTPException(502, "bad JSON"), RuntimeError("mock failure")):
            self.guard.stopped = None
            calls = 0
            async def send(*args):
                nonlocal calls
                calls += 1
                self.guard.reserve()
                raise error
            with patch.object(server, "guarded_chat_completions", send):
                for _ in range(2):
                    with self.assertRaises(server.HTTPException) as caught:
                        await server.chat_completions(None)
                    self.assertEqual(caught.exception.status_code, 400)
            self.assertEqual(calls, 1)

    async def test_cancelled_run_stays_stopped(self):
        async def send(*args):
            raise asyncio.CancelledError()
        with patch.object(server, "guarded_chat_completions", send):
            with self.assertRaises(asyncio.CancelledError):
                await server.chat_completions(None)
            with self.assertRaises(server.HTTPException):
                await server.chat_completions(None)

    async def test_upstream_429_stops_only_google_host(self):
        server.observe_upstream_response(SimpleNamespace(url="https://other.example/", status=429))
        self.assertIsNone(self.guard.stopped)
        server.observe_upstream_response(SimpleNamespace(url="https://gemini.google.com/test", status=429))
        self.assertEqual(self.guard.stopped, "upstream_http_429")

    async def test_quota_snapshot_stops_before_timeout(self):
        page = SimpleNamespace()
        async def snapshot(*args):
            return {"count": 0, "finished": False, "notice": "You've reached your limit"}
        with patch.object(server, "read_response_snapshot", snapshot):
            with self.assertRaises(GuardStopped):
                await server.wait_for_complete_response(page, 0, 1)
        self.assertEqual(self.guard.stopped, "upstream_usage_limit")

    async def test_final_budget_response_is_allowed(self):
        for _ in range(8):
            self.guard.reserve()
        async def snapshot(*args):
            return {"count": 1, "finished": True, "id": "reply", "html": "OK", "text": "OK"}
        with patch.object(server, "read_response_snapshot", snapshot):
            self.assertEqual((await server.wait_for_complete_response(None, 0, 2))["text"], "OK")


if __name__ == "__main__":
    unittest.main()
