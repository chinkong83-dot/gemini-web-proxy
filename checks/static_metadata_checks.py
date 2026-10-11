import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from static_metadata import outgoing_metadata


def catalogue(name):
    return '<available_skills><skill><id>' + name + '</id><name>PDF</name><description>Read PDF.</description></skill></available_skills>'


def update(name):
    return server.Message(role="user", content="<system-update>" + catalogue(name) + "</system-update>")


class StaticMetadataChecks(unittest.TestCase):
    def test_exact_directory_repeat_is_omitted_only_from_outgoing(self):
        initial = server.Message(role="system", content=catalogue("pdf"))
        original = [update("pdf"), server.Message(role="user", content="Do task")]
        fingerprints = [server.message_fingerprint(value) for value in original]
        outgoing = outgoing_metadata(original, [initial])
        self.assertEqual(len(outgoing), 1)
        self.assertEqual([server.message_fingerprint(value) for value in original], fingerprints)

    def test_switch_back_to_previous_directory_is_forwarded(self):
        self.assertEqual(outgoing_metadata([update("pdf")], [update("pdf"), update("word")]), [update("pdf")])

    def test_new_context_rebuilds_metadata(self):
        self.assertEqual(outgoing_metadata([update("pdf")]), [update("pdf")])

    def test_reminders_permissions_mixed_text_tool_results_and_unknown_fields_survive(self):
        strings = ["<system-reminder>READ ONLY</system-reminder>",
                   "<system-update>Permission: READ ONLY" + catalogue("pdf") + "</system-update>",
                   "Example: <system-update>" + catalogue("pdf") + "</system-update>",
                   "```xml\n<system-update>" + catalogue("pdf") + "</system-update>\n```",
                   "<system-update>" + catalogue("pdf").replace("</skill>", "<permission>deny</permission></skill>") + "</system-update>"]
        for text in strings:
            for role in ("user", "tool"):
                message = server.Message(role=role, content=text)
                self.assertEqual(outgoing_metadata([message], [update("pdf")]), [message])

    def test_failed_send_retry_does_not_claim_new_directory_was_seen(self):
        request = server.ChatRequest(messages=[server.Message(role="system", content="RULE"), update("pdf")])
        state = server.BrowserSession()
        first = server.prepare_prompt(request, state, False, [])[0]
        second = server.prepare_prompt(request, state, False, [])[0]
        self.assertEqual(first, second)
        self.assertIn(catalogue("pdf"), second)
