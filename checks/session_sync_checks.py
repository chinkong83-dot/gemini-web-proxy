import asyncio
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from conversation_sync import ConversationSync
from starlette.requests import Request


class FakePage:
    def __init__(self):
        self.url = "https://gemini.google.com/app"
        self.replies = []
        self.closed = False

    async def goto(self, url):
        self.url = url

    async def wait_for_selector(self, *args, **kwargs):
        return self

    async def evaluate(self, script):
        return {"url": self.url, "count": len(self.replies), "last": self.replies[-1] if self.replies else ""}

    def is_closed(self):
        return self.closed

    async def close(self):
        self.closed = True


class FakeBrowser:
    def __init__(self):
        self.pages = []

    async def new_page(self):
        page = FakePage()
        self.pages.append(page)
        return page


def http_request(session="ses_a", **headers):
    values = {"x-opencode-session-id": session, **headers}
    return Request({"type": "http", "headers": [(key.encode(), value.encode()) for key, value in values.items()]})


def messages(question="QUESTION_A", instructions="OpenCode main\nRULE_A"):
    return [{"role": "system", "content": instructions}, {"role": "user", "content": question}]


class PlannerChecks(unittest.TestCase):
    def test_prefix_and_rebuild_reasons(self):
        state = ConversationSync()
        initial = state.plan("rules", ("q1",))
        self.assertEqual(initial.reason, "new-session")
        state.commit(initial, "a1", None)
        delta = state.plan("rules", ("q1", "a1", "q2"))
        self.assertFalse(delta.rebuild)
        self.assertEqual(delta.start, 2)
        self.assertEqual(state.plan("rules", ("summary", "q2")).reason, "history-rewritten")
        self.assertEqual(state.plan("rules2", ("q1", "a1", "q2")).reason, "instructions-changed")
        self.assertEqual(state.plan("rules", ("q1", "a1")).reason, "regeneration")
        self.assertEqual(state.plan("rules", ("q1",), reset=True).reason, "explicit-reset")
        state.invalidate()
        self.assertEqual(state.plan("rules", ("q1", "a1", "q2")).reason, "unsynchronized-browser")

    def test_actual_session_id_wins_over_parent_affinity(self):
        request = server.ChatRequest(messages=messages())
        a = server.session_key(request, http_request("ses_a", **{"x-opencode-session": "parent"}), False)
        b = server.session_key(request, http_request("ses_b", **{"x-opencode-session": "parent"}), False)
        self.assertNotEqual(a, b)
        self.assertIsNone(server.session_key(request, None, False))
        self.assertNotEqual(a, server.session_key(request, http_request("ses_a"), True))
        with self.assertRaises(server.HTTPException):
            server.session_key(server.ChatRequest(messages=messages(), session_id="other"), http_request(), False)

    def test_argument_formatting_does_not_break_prefix_matching(self):
        a = server.Message(role="assistant", tool_calls=[{"id": "id", "function": {"name": "read", "arguments": '{"b":2,"a":1}'}}])
        b = server.Message(role="assistant", content="", tool_calls=[{"id": "id", "function": {"name": "read", "arguments": '{ "a": 1, "b": 2 }'}}])
        self.assertEqual(server.message_fingerprint(server.normalize_message(a, False)),
                         server.message_fingerprint(server.normalize_message(b, False)))


class BrowserChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        server.browser_sessions.clear()
        self.browser = FakeBrowser()
        self.sent = []
        self.responses = []
        self.output = io.StringIO()
        self.patches = [patch.object(server, "context", self.browser), patch.object(server, "is_ready", True),
                        patch.object(server, "send_to_gemini", self.send), contextlib.redirect_stdout(self.output)]
        for item in self.patches:
            item.__enter__()

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.__exit__(None, None, None)
        server.browser_sessions.clear()

    async def send(self, page, text, timeout):
        self.sent.append((page, text))
        response = self.responses.pop(0) if self.responses else "REPLY_" + str(len(self.sent))
        if isinstance(response, BaseException):
            raise response
        page.replies.append(response)
        page.url = "https://gemini.google.com/app/owned-chat"
        return response

    async def call(self, history, session="ses_a", headers=None, **options):
        request = server.ChatRequest(messages=history, **options)
        return await server.chat_completions(request, http_request(session, **(headers or {})))

    async def test_only_new_user_turn_is_forwarded(self):
        initial = messages()
        first = await self.call(initial)
        history = initial + [first["choices"][0]["message"], {"role": "user", "content": "QUESTION_B"}]
        second = await self.call(history)
        self.assertIn("RULE_A", self.sent[0][1])
        self.assertEqual(self.sent[1][1], "QUESTION_B")
        self.assertIs(self.sent[0][0], self.sent[1][0])
        self.assertGreater(second["usage"]["prompt_tokens"], first["usage"]["prompt_tokens"])
        self.assertNotIn("QUESTION_A", self.output.getvalue())
        self.assertNotIn("REPLY_1", self.output.getvalue())

    async def test_identical_first_question_in_different_ui_sessions_is_isolated(self):
        await asyncio.gather(self.call(messages(), "ses_a"), self.call(messages(), "ses_b"))
        self.assertEqual(len(self.browser.pages), 2)
        self.assertIsNot(self.sent[0][0], self.sent[1][0])

    async def test_title_and_main_streams_are_isolated(self):
        await self.call(messages(), "ses_a")
        await self.call(messages(instructions="Title generator\nRULE_A"), "ses_a")
        self.assertEqual(len(self.browser.pages), 2)

    async def test_history_compaction_replaces_the_page(self):
        await self.call(messages())
        old = self.browser.pages[0]
        await self.call(messages("COMPACTED_SUMMARY"))
        self.assertTrue(old.closed)
        self.assertEqual(len(self.browser.pages), 2)
        self.assertIn("RULE_A", self.sent[-1][1])

    async def test_rule_change_with_same_channel_rebuilds(self):
        await self.call(messages())
        await self.call(messages(instructions="OpenCode main\nRULE_B"))
        self.assertTrue(self.browser.pages[0].closed)
        self.assertIn("RULE_B", self.sent[-1][1])

    async def test_closed_or_manually_changed_pages_rebuild(self):
        initial = messages()
        first = await self.call(initial)
        history = initial + [first["choices"][0]["message"], {"role": "user", "content": "QUESTION_B"}]
        self.browser.pages[0].replies.append("MANUAL_CHANGE")
        second = await self.call(history)
        self.assertEqual(len(self.browser.pages), 2)
        self.browser.pages[-1].closed = True
        await self.call(history + [second["choices"][0]["message"], {"role": "user", "content": "QUESTION_C"}])
        self.assertEqual(len(self.browser.pages), 3)

    async def test_failed_send_does_not_advance_sync_state(self):
        self.responses = [RuntimeError("FAKE_SECRET"), "OK"]
        with self.assertRaises(server.HTTPException) as caught:
            await self.call(messages())
        self.assertEqual(caught.exception.status_code, 502)
        await self.call(messages())
        self.assertEqual(len(self.browser.pages), 2)
        self.assertNotIn("FAKE_SECRET", self.output.getvalue())

    async def test_client_without_session_id_is_stateless_and_does_not_leak_pages(self):
        request = server.ChatRequest(messages=messages())
        await server.chat_completions(request)
        await server.chat_completions(request)
        self.assertEqual(len(self.browser.pages), 2)
        self.assertTrue(all(page.closed for page in self.browser.pages))
        self.assertFalse(server.browser_sessions)

    async def test_chat_mode_filters_rules_and_updates_before_diffing(self):
        initial = messages() + [{"role": "user", "content": "<system-update>CONTROL</system-update>"}]
        first = await self.call(initial, model=server.CHAT_MODEL)
        self.assertEqual(self.sent[-1][1], "QUESTION_A")
        history = initial + [first["choices"][0]["message"], {"role": "user", "content": "QUESTION_B"}]
        await self.call(history, model=server.CHAT_MODEL)
        self.assertEqual(self.sent[-1][1], "QUESTION_B")

    async def test_tool_results_keep_call_binding_without_resending_code_or_schema(self):
        tools = [server.Tool(type="function", function=server.FunctionDef(name="shell", description="SCHEMA_MARKER",
                  parameters={"type": "object", "properties": {"command": {"type": "string"}}}))]
        self.responses = [json.dumps({"tool_calls": [{"name": "shell", "arguments": {"command": "CMD_A"}},
                                                      {"name": "shell", "arguments": {"command": "CMD_B"}}]}), "DONE"]
        initial = messages()
        first = await self.call(initial, tools=tools)
        answer = first["choices"][0]["message"]
        calls = answer["tool_calls"]
        results = [{"role": "tool", "tool_call_id": calls[1]["id"], "content": "RESULT_B"},
                   {"role": "tool", "tool_call_id": calls[0]["id"], "content": "RESULT_A"}]
        await self.call(initial + [answer] + results, tools=tools)
        prompt = self.sent[-1][1]
        self.assertIn("request 2: shell", prompt)
        self.assertIn("request 1: shell", prompt)
        self.assertIn("RESULT_A", prompt)
        self.assertNotIn("CMD_A", prompt)
        self.assertNotIn("SCHEMA_MARKER", prompt)
        self.assertNotIn("RULE_A", prompt)

    async def test_catalog_changes_are_forwarded_once(self):
        tool = server.Tool(type="function", function=server.FunctionDef(name="read", description="OLD_SCHEMA"))
        initial = messages()
        first = await self.call(initial, tools=[tool])
        tool.function.description = "NEW_SCHEMA"
        history = initial + [first["choices"][0]["message"], {"role": "user", "content": "QUESTION_B"}]
        await self.call(history, tools=[tool])
        self.assertIn("NEW_SCHEMA", self.sent[-1][1])
        self.assertNotIn("OLD_SCHEMA", self.sent[-1][1])
        self.assertEqual(len(self.browser.pages), 1)

    async def test_tool_choice_none_and_reenable_do_not_resend_catalogue(self):
        tool = server.Tool(type="function", function=server.FunctionDef(name="read", description="SCHEMA_MARKER"))
        initial = messages()
        first = await self.call(initial, tools=[tool])
        history = initial + [first["choices"][0]["message"], {"role": "user", "content": "B"}]
        second = await self.call(history, tools=[tool], tool_choice="none")
        self.assertIn("Tools are disabled for this turn", self.sent[-1][1])
        self.assertNotIn("SCHEMA_MARKER", self.sent[-1][1])
        await self.call(history + [second["choices"][0]["message"], {"role": "user", "content": "C"}], tools=[tool])
        self.assertIn('Tools allowed for this turn: ["read"]', self.sent[-1][1])
        self.assertNotIn("SCHEMA_MARKER", self.sent[-1][1])

    async def test_explicit_reset_and_regeneration_create_fresh_pages(self):
        initial = messages()
        await self.call(initial)
        await self.call(initial)
        self.assertTrue(self.browser.pages[0].closed)
        await self.call(initial, reset_session=True)
        self.assertTrue(self.browser.pages[1].closed)

    async def test_same_session_waits_without_creating_duplicate_pages(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        async def slow_send(page, text, timeout):
            if not self.sent:
                entered.set()
                await release.wait()
            return await self.send(page, text, timeout)
        with patch.object(server, "send_to_gemini", slow_send):
            first = asyncio.create_task(self.call(messages()))
            await entered.wait()
            second_history = messages() + [{"role": "assistant", "content": "REPLY_1"}, {"role": "user", "content": "QUESTION_B"}]
            second = asyncio.create_task(self.call(second_history))
            await asyncio.sleep(0)
            self.assertEqual(len(self.browser.pages), 1)
            release.set()
            await asyncio.gather(first, second)
        self.assertEqual(len(self.browser.pages), 1)
        self.assertEqual(self.sent[-1][1], "QUESTION_B")

    async def test_delete_during_send_serializes_and_removes_state(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        async def slow_send(page, text, timeout):
            entered.set()
            await release.wait()
            return await self.send(page, text, timeout)
        with patch.object(server, "send_to_gemini", slow_send):
            request = asyncio.create_task(self.call(messages()))
            await entered.wait()
            key = next(iter(server.browser_sessions))
            deletion = asyncio.create_task(server.delete_session(key))
            await asyncio.sleep(0)
            self.assertFalse(self.browser.pages[0].closed)
            release.set()
            await asyncio.gather(request, deletion)
        self.assertFalse(server.browser_sessions)

    async def test_cancellation_marks_context_dirty(self):
        entered = asyncio.Event()
        async def paused_send(page, text, timeout):
            entered.set()
            await asyncio.Event().wait()
        with patch.object(server, "send_to_gemini", paused_send):
            request = asyncio.create_task(self.call(messages()))
            await entered.wait()
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
        await self.call(messages())
        self.assertEqual(len(self.browser.pages), 2)

    async def test_fresh_page_guard_refuses_restored_or_foreign_chat(self):
        async def not_fresh():
            page = FakePage()
            page.replies = ["OTHER_CONVERSATION"]
            self.browser.pages.append(page)
            return page
        with patch.object(self.browser, "new_page", not_fresh):
            with self.assertRaises(server.HTTPException) as caught:
                await self.call(messages())
        self.assertEqual(caught.exception.status_code, 503)
        self.assertFalse(self.sent)
        self.assertTrue(self.browser.pages[0].closed)

    async def test_long_conversation_transmits_linear_content(self):
        history = [{"role": "system", "content": "OpenCode main\n" + "X" * 4000}]
        questions = []
        full_replay_chars = 0
        for index in range(20):
            question = f"QUESTION_{index}:" + "Y" * 500
            questions.append(question)
            history.append({"role": "user", "content": question})
            full_replay_chars += len(server.format_conversation([server.Message(**message) for message in history]))
            response = await self.call(history)
            history.append(response["choices"][0]["message"])
        for index in range(1, 20):
            self.assertEqual(self.sent[index][1], questions[index])
        incremental_chars = sum(len(text) for _, text in self.sent)
        self.assertLess(incremental_chars * 5, full_replay_chars)
        self.assertEqual(len(self.browser.pages), 1)

    async def test_stream_usage_reports_retained_context(self):
        initial = messages()
        first = await self.call(initial)
        response = await self.call(initial + [first["choices"][0]["message"], {"role": "user", "content": "B"}],
                                   stream=True, stream_options={"include_usage": True})
        chunks = [chunk async for chunk in response.body_iterator]
        usage = next(json.loads(chunk[6:])["usage"] for chunk in chunks if '"usage"' in chunk)
        self.assertGreater(usage["prompt_tokens"], len(self.sent[-1][1]) // 4)

    async def test_delete_cleans_page_and_state(self):
        await self.call(messages())
        key = next(iter(server.browser_sessions))
        await server.delete_session(key)
        self.assertTrue(self.browser.pages[0].closed)
        self.assertFalse(server.browser_sessions)
        self.assertEqual((await server.health())["sessions"], 0)


if __name__ == "__main__":
    unittest.main()
