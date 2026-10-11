import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import server
import session_sync_checks as sync
from request_guard import RequestGuard


class MonitoredRunChecks(unittest.IsolatedAsyncioTestCase):
    send = sync.BrowserChecks.send
    call = sync.BrowserChecks.call

    async def asyncSetUp(self):
        await sync.BrowserChecks.asyncSetUp(self)
        self.temp = tempfile.TemporaryDirectory()
        self.guard = RequestGuard(None, Path(self.temp.name) / 'ledger.json')
        self.monitors = [patch.object(server, 'request_guard', self.guard),
                         patch.object(server, 'stop_on_failure', True)]
        for value in self.monitors: value.__enter__()

    async def asyncTearDown(self):
        for value in reversed(self.monitors): value.__exit__(None, None, None)
        self.temp.cleanup()
        await sync.BrowserChecks.asyncTearDown(self)

    async def test_three_repeated_replies_stop_subsequent_requests(self):
        self.responses = ['SAME'] * 4
        history = sync.messages()
        for i in range(2):
            result = await self.call(history)
            history += [result['choices'][0]['message'], {'role': 'user', 'content': 'NEXT' + str(i)}]
        with self.assertRaises(server.HTTPException): await self.call(history)
        with self.assertRaises(server.HTTPException): await self.call(history)
        self.assertEqual(len(self.sent), 3)
        self.assertEqual(self.guard.stopped, 'repeated_response_possible_usage_limit')

    async def test_timeout_stops_before_another_model_send(self):
        self.responses = [TimeoutError(), 'UNREACHABLE']
        for _ in range(2):
            with self.assertRaises(server.HTTPException): await self.call(sync.messages())
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.guard.stopped, 'response_timeout_possible_usage_limit')

    async def test_malformed_tool_json_is_an_error_and_stops_followups(self):
        self.responses = ['{"tool_calls":[', 'UNREACHABLE']
        tools = [server.Tool(type='function', function=server.FunctionDef(name='read'))]
        for _ in range(2):
            with self.assertRaises(server.HTTPException): await self.call(sync.messages(), tools=tools)
        self.assertEqual(len(self.sent), 1)

    async def test_nonmonitored_normal_repetition_does_not_stop(self):
        self.responses = ['SAME'] * 3
        history = sync.messages()
        with patch.object(server, 'stop_on_failure', False):
            for i in range(3):
                result = await self.call(history)
                history += [result['choices'][0]['message'], {'role': 'user', 'content': 'NEXT' + str(i)}]
        self.assertIsNone(self.guard.stopped)
