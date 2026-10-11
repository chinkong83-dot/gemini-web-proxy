import copy
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prompt_compaction as compact
import server


SHELL_DESCRIPTION = "Execute a shell command and return its output. Commands run on macOS using zsh. Quote file paths containing spaces or special characters. Prefer dedicated tools over shell commands when possible. When output is large, the full result is saved to a file and a truncated preview is returned. Rely on automatic truncation unless filtering the output is more useful. Commands accept an optional timeout, background commands have no timeout by default. Background commands return immediately, and you will be notified when they complete."
DOCX_DESCRIPTION = "Use this skill whenever the user wants to create, read, edit, or manipulate Word documents (.docx) or Word templates (.dotx). Triggers include: any mention of 'Word doc', 'word document', '.docx', '.dotx', or requests to produce professional documents with formatting like tables of contents, page numbers, or letterheads. Also use when extracting or reorganizing content from .docx or .dotx files, inserting or replacing images in documents, find-and-replace in Word files, working with tracked changes or comments, or converting content into a polished Word document. If the user asks for a 'report', 'memo', 'letter', 'template', or similar deliverable as a Word or .docx file (to download, email or print), use this skill. However, if they ask for a document, page, report, memo, or notes WITHOUT naming a file format and the session offers Codex's own dedicated document or page skill or connector, use that instead. Do NOT use for PDFs, spreadsheets, Google Docs, or coding unrelated to document generation."
CODE_MODE = '# Code Mode\nUse execute, not direct calls. Search synchronously for exact paths.\ntools.api.find({ limit?: number /** default: 5, maximum: 10 */ }): Promise<string[]>\n'


def skills(description=DOCX_DESCRIPTION, identifier="docx"):
    return f'<available_skills><skill><id>{identifier}</id><name>Word</name><description>{description}</description></skill></available_skills>'


def system_text():
    prose = '\n'.join('- ' + text for text in compact.HARNESS_REPLACEMENTS)
    return compact.HARNESS_HEADER + '\n# Harness\n' + prose + '\n' + CODE_MODE + skills()


def shell(description=SHELL_DESCRIPTION):
    return server.Tool(type="function", function=server.FunctionDef(name="shell", description=description,
        parameters={"type": "object", "required": ["command"], "additionalProperties": False,
            "properties": {"command": {"type": "string", "description": "Actual command; requires approval."},
                "mode": {"type": "string", "enum": ["read", "write"], "default": "read"},
                "timeout": {"type": "integer", "minimum": 0, "default": 120000},
                "nested": {"type": "array", "items": {"type": "object", "required": ["id"],
                    "properties": {"id": {"type": "string", "pattern": "^id_"}}}}}}))


class CompactionChecks(unittest.TestCase):
    def test_known_tool_shrinks_without_modifying_any_schema_field(self):
        tool = shell()
        original = copy.deepcopy(tool.model_dump())
        definitions = json.loads(server.format_tools_prompt([tool]).split('Tools: ', 1)[1])
        self.assertLess(len(definitions[0]['description']), len(SHELL_DESCRIPTION))
        self.assertEqual(definitions[0]['parameters'], original['function']['parameters'])
        self.assertEqual(tool.model_dump(), original)

    def test_custom_or_changed_tool_description_is_verbatim(self):
        for name, description in [('shell', SHELL_DESCRIPTION + ' Only operate in approved directories.'),
                                  ('custom_shell', SHELL_DESCRIPTION)]:
            definition = {"name": name, "description": description, "parameters": {"enum": ["safe"]}}
            self.assertEqual(compact.compact_tool_definition(definition), definition)

    def test_skill_ids_names_and_exclusions_survive(self):
        content = compact.compact_system_prompt(system_text())
        catalog = content[content.index('<available_skills>'):]
        entry = ET.fromstring(catalog).find('skill')
        self.assertEqual(entry.findtext('id'), 'docx')
        self.assertEqual(entry.findtext('name'), 'Word')
        description = entry.findtext('description')
        self.assertLess(len(description), len(DOCX_DESCRIPTION))
        for term in ('.docx', '.dotx', 'dedicated document/page', 'Google Docs', 'PDFs'):
            self.assertIn(term, description)

    def test_custom_skills_and_changed_restrictions_are_verbatim(self):
        for description, identifier in [(DOCX_DESCRIPTION + ' Require explicit permission.', 'docx'),
                                         (DOCX_DESCRIPTION, 'private-word-workflow')]:
            content = compact.HARNESS_HEADER + '\n' + skills(description, identifier)
            self.assertEqual(compact.compact_system_prompt(content), content)

    def test_code_mode_signatures_and_user_rules_are_byte_preserved(self):
        protected = '\nInstructions from: /fixture/AGENTS.md\nONLY WITH PERMISSION\n' + skills()
        protected += '\n<env>Working directory: /fixture</env>'
        content = compact.compact_system_prompt(system_text() + protected)
        self.assertIn(CODE_MODE, content)
        self.assertTrue(content.endswith(protected))
        env_only = compact.HARNESS_HEADER + '\n<env>' + skills() + '</env>'
        self.assertEqual(compact.compact_system_prompt(env_only), env_only)

    def test_custom_system_and_literal_user_tool_text_are_verbatim(self):
        literal = skills()
        custom = 'You are a custom assistant.\n' + literal
        self.assertEqual(compact.compact_system_prompt(custom), custom)
        for chat_only in (False, True):
            self.assertEqual(server.format_conversation([server.Message(role='user', content=literal)],
                                                       chat_only=chat_only), literal)
        result = server.format_conversation([server.Message(role='tool', content=literal)])
        self.assertIn(literal, result)

    def test_idempotence_and_required_tool_choice(self):
        content = compact.compact_system_prompt(system_text())
        self.assertEqual(compact.compact_system_prompt(content), content)
        self.assertIn('Call a listed tool in this response.', server.format_tools_prompt([shell()], 'required'))

    def test_large_malformed_catalog_is_preserved(self):
        partial_entry = '<skill><id>docx</id><name>Word</name><description>'
        for text in ('<available_skills>' * 5000,
                     '<available_skills>' + partial_entry * 5000 + '</available_skills>'):
            self.assertEqual(compact.compact_skill_catalogs(text), text)

    def test_incremental_turn_skips_compacted_metadata_and_keeps_catalog_updates(self):
        tool = shell()
        history = [server.Message(role='system', content=system_text()), server.Message(role='user', content='FIRST')]
        state = server.BrowserSession()
        request = server.ChatRequest(messages=history, tools=[tool])
        text, plan, catalog = server.prepare_prompt(request, state, False, [tool])
        self.assertNotIn(DOCX_DESCRIPTION, text)
        answer = server.Message(role='assistant', content='ANSWER')
        state.sync.commit(plan, server.message_fingerprint(answer), catalog, len(text))
        updated = history + [answer, server.Message(role='user', content='SECOND')]
        request = server.ChatRequest(messages=updated, tools=[tool])
        delta, plan, _ = server.prepare_prompt(request, state, False, [tool])
        self.assertFalse(plan.rebuild)
        self.assertNotIn('<available_skills>', delta)
        self.assertNotIn('parameters', delta)
        self.assertIn('SECOND', delta)
        changed = shell(SHELL_DESCRIPTION + ' Only use approved directories.')
        replacement, plan, _ = server.prepare_prompt(server.ChatRequest(messages=updated, tools=[changed]),
                                                    state, False, [changed])
        self.assertFalse(plan.rebuild)
        self.assertIn(changed.function.description, replacement)
        self.assertIn('Update the tool catalogue', replacement)

    def test_changed_raw_system_rules_still_rebuild_and_updates_remain_verbatim(self):
        history = [server.Message(role='system', content=system_text()), server.Message(role='user', content='FIRST')]
        state = server.BrowserSession()
        _, plan, catalog = server.prepare_prompt(server.ChatRequest(messages=history), state, False, [])
        answer = server.Message(role='assistant', content='ANSWER')
        state.sync.commit(plan, server.message_fingerprint(answer), catalog)
        changed = system_text().replace(DOCX_DESCRIPTION, DOCX_DESCRIPTION + ' Never access private files.')
        request = server.ChatRequest(messages=[server.Message(role='system', content=changed),
                                             *history[1:], answer, server.Message(role='user', content='NEXT')])
        text, plan, _ = server.prepare_prompt(request, state, False, [])
        self.assertEqual(plan.reason, 'instructions-changed')
        self.assertIn('Never access private files.', text)
        update = '<system-update>New Code Mode signatures:\n' + CODE_MODE + skills() + '</system-update>'
        forwarded = server.format_conversation([server.Message(role='user', content=update)])
        self.assertEqual(forwarded, update)
        chat = server.format_conversation([server.Message(role='user', content=update)], chat_only=True)
        self.assertEqual(chat, '')


if __name__ == '__main__':
    unittest.main()
