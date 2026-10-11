import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import server
import unittest
import session_sync_checks as sync
from session_sync_checks import messages


def tool(name, description):
    return server.Tool(type="function", function=server.FunctionDef(name=name, description=description,
                       parameters={"type": "object", "properties": {"value": {"type": "string"}}}))


class RepetitionChecks(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = sync.BrowserChecks.asyncSetUp
    asyncTearDown = sync.BrowserChecks.asyncTearDown
    send = sync.BrowserChecks.send
    call = sync.BrowserChecks.call

    async def test_normal_turns_do_not_repeat_tool_policy_or_protocol(self):
        tools = [tool("read", "READ_SCHEMA"), tool("shell", "SHELL_SCHEMA")]
        history = messages()
        for i in range(3):
            result = await self.call(history, tools=tools)
            history += [result["choices"][0]["message"], {"role": "user", "content": "NEXT_" + str(i)}]
        for _, text in self.sent[1:]:
            self.assertNotIn("Available tools", text)
            self.assertNotIn("Tool calls:", text)
            self.assertNotIn("READ_SCHEMA", text)
            self.assertNotIn("RULE_A", text)

    async def test_change_one_schema_preserves_other_catalogue_without_repeating_it(self):
        tools = [tool("read", "UNCHANGED_SCHEMA"), tool("shell", "OLD_SHELL")]
        history = messages()
        result = await self.call(history, tools=tools)
        tools[1] = tool("shell", "NEW_SHELL")
        await self.call(history + [result["choices"][0]["message"], {"role": "user", "content": "NEXT"}], tools=tools)
        text = self.sent[-1][1]
        self.assertIn("NEW_SHELL", text)
        self.assertNotIn("UNCHANGED_SCHEMA", text)
        self.assertNotIn("OLD_SHELL", text)
        self.assertNotIn("Tool calls: JSON", text)
        self.assertEqual(len(self.browser.pages), 1)

    async def test_removed_tools_and_added_tools_are_explicit(self):
        history = messages()
        result = await self.call(history, tools=[tool("read", "OLD_READ"), tool("shell", "UNCHANGED")])
        await self.call(history + [result["choices"][0]["message"], {"role": "user", "content": "NEXT"}],
                        tools=[tool("shell", "UNCHANGED"), tool("write", "NEW_WRITE")])
        text = self.sent[-1][1]
        self.assertIn('"remove":["read"]', text)
        self.assertIn("NEW_WRITE", text)
        self.assertNotIn("UNCHANGED", text)
        self.assertIn('Available tools: ["shell", "write"]', text)

    async def test_required_and_forced_tool_constraints_apply_on_each_turn(self):
        for choice in ("required", {"type": "function", "function": {"name": "read"}}):
            self.responses = [json.dumps({"tool_calls": [{"name": "read", "arguments": {}}]})] * 2
            history = messages()
            tools = [tool("read", "SCHEMA")]
            result = await self.call(history, tools=tools, tool_choice=choice, session=str(choice))
            call = result["choices"][0]["message"]
            await self.call(history + [call, {"role": "tool", "tool_call_id": call["tool_calls"][0]["id"], "content": "RESULT"}],
                            tools=tools, tool_choice=choice, session=str(choice))
            self.assertIn("Request a tool in JSON", self.sent[-1][1])
            self.assertNotIn("SCHEMA", self.sent[-1][1])

    async def test_disable_repeat_and_reenable(self):
        tools = [tool("read", "SCHEMA")]
        history = messages()
        for i, choice in enumerate((None, "none", "none", None)):
            result = await self.call(history, tools=tools, tool_choice=choice)
            history += [result["choices"][0]["message"], {"role": "user", "content": "NEXT" + str(i)}]
        self.assertIn("Tools are disabled", self.sent[1][1])
        self.assertNotIn("Tools are disabled", self.sent[2][1])
        self.assertIn("Available tools", self.sent[3][1])
        self.assertNotIn("SCHEMA", self.sent[3][1])

    async def test_new_context_keeps_complete_bootstrap(self):
        tools = [tool("read", "FULL_SCHEMA")]
        await self.call(messages(), tools=tools, session="one")
        await self.call(messages(), tools=tools, session="two")
        for _, text in self.sent:
            self.assertIn("FULL_SCHEMA", text)
            self.assertIn("RULE_A", text)
            self.assertIn("Tool calls: JSON", text)
            self.assertIn("external client, outside Gemini", text)

    async def test_duplicate_tool_names_are_rejected_before_send(self):
        with self.assertRaises(server.HTTPException):
            await self.call(messages(), tools=[tool("read", "ONE"), tool("read", "TWO")])
        self.assertEqual(len(self.sent), 0)

    async def test_catalogue_order_change_does_not_repeat_definitions(self):
        tools = [tool("read", "ONE"), tool("shell", "TWO")]
        history = messages()
        result = await self.call(history, tools=tools)
        await self.call(history + [result["choices"][0]["message"], {"role": "user", "content": "NEXT"}], tools=list(reversed(tools)))
        self.assertEqual(self.sent[-1][1], "NEXT")

    async def test_tool_removed_then_readded_gets_complete_definition(self):
        history = messages()
        for tools in ([tool("read", "READ"), tool("shell", "SHELL")], [tool("read", "READ")],
                      [tool("read", "READ"), tool("shell", "SHELL")]):
            result = await self.call(history, tools=tools)
            history += [result["choices"][0]["message"], {"role": "user", "content": "NEXT"}]
        self.assertIn("SHELL", self.sent[-1][1])
        self.assertIn('"parameters"', self.sent[-1][1])
