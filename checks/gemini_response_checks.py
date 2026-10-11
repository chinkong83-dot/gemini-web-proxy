import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


def response(text, *, finished=True, count=1, identity="reply1"):
    return {"count": count, "id": identity, "html": f"<p>{text}</p>",
            "text": text, "finished": finished}


class SnapshotPage:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.reads = 0

    async def evaluate(self, script):
        snapshot = self.snapshots[min(self.reads, len(self.snapshots) - 1)]
        self.reads += 1
        if isinstance(snapshot, Exception):
            raise snapshot
        return snapshot


class CompletionChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.elapsed = 0

        async def sleep(seconds):
            self.elapsed += seconds

        self.patches = [patch.object(server, "time", SimpleNamespace(monotonic=lambda: self.elapsed)),
                        patch.object(server.asyncio, "sleep", sleep)]
        for item in self.patches:
            item.__enter__()

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.__exit__(None, None, None)

    async def test_streaming_pause_does_not_truncate_or_force_next_turn_rebuild(self):
        partial = response("ITEM_1 ITEM_2", finished=False)
        full = response("ITEM_1 ITEM_2 ITEM_3 ITEM_4 ITEM_5")
        page = SnapshotPage([partial] * 8 + [full] * 3)
        completed = await server.wait_for_complete_response(page, 0, 10)
        self.assertEqual(completed, full)
        self.assertEqual(page.reads, 11)
        state = server.ConversationSync()
        first = state.plan("rules", ("question",))
        state.commit(first, server.fingerprint(completed["text"]), None)
        continued = state.plan("rules", ("question", server.fingerprint(full["text"]), "next"))
        self.assertFalse(continued.rebuild)

    async def test_equal_length_changes_are_not_stable(self):
        page = SnapshotPage([response("old"), response("new"), response("end")])
        result = await server.wait_for_complete_response(page, 0, 10)
        self.assertEqual(result["text"], "end")
        self.assertEqual(page.reads, 5)

    async def test_incomplete_tool_json_is_not_returned(self):
        partial = response('{"tool_calls":[{"name":"shell"', finished=False)
        full = response('{"tool_calls":[{"name":"shell","arguments":{"command":"printf OK"}}]}')
        page = SnapshotPage([partial] * 8 + [full] * 3)
        completed = await server.wait_for_complete_response(page, 0, 10)
        self.assertEqual(server.parse_tool_calls(completed["text"])[0]["name"], "shell")

    async def test_old_completed_response_cannot_satisfy_new_request(self):
        page = SnapshotPage([response("old", count=1)])
        with self.assertRaises(TimeoutError):
            await server.wait_for_complete_response(page, 1, 1)

    async def test_unfinished_response_times_out_instead_of_returning_partial_text(self):
        page = SnapshotPage([response("partial", finished=False)])
        with self.assertRaises(TimeoutError):
            await server.wait_for_complete_response(page, 0, 1)

    async def test_read_error_resets_stability(self):
        page = SnapshotPage([response("old"), response("old"), RuntimeError("detached"),
                             response("full"), response("full"), response("full")])
        completed = await server.wait_for_complete_response(page, 0, 10)
        self.assertEqual(completed["text"], "full")
        self.assertEqual(page.reads, 6)


if __name__ == "__main__":
    unittest.main()
