import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from request_guard import RequestGuard
from playwright.async_api import async_playwright


class BoundaryChecks(unittest.IsolatedAsyncioTestCase):
    async def test_chinese_send_is_single_click_and_never_enter_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            guard = RequestGuard(8, Path(folder) / 'ledger.json')
            keys, selectors, clicks = [], [], []
            class Control:
                async def click(self): clicks.append('click')
            class Page:
                async def wait_for_selector(self, selector, **kwargs):
                    selectors.append(selector)
                    return Control()
                async def query_selector_all(self, selector): return []
                async def evaluate(self, script, *args):
                    return {'editorChars': 3, 'controls': []} if not args else None
            async def press(key): keys.append(key)
            async def complete(*args):
                return {'html': '<p>OK</p>', 'text': 'OK'}
            async def sleep(*args): pass
            page = Page();page.keyboard = SimpleNamespace(press=press)
            with patch.object(server, 'request_guard', guard), patch.object(server.asyncio, 'sleep', sleep), patch.object(server, 'wait_for_complete_response', complete):
                self.assertEqual(await server.send_monitored_prompt(page, 'ABC'), 'OK')
            self.assertIn('傳送訊息', selectors[-1])
            self.assertEqual(guard.count, 1)
            self.assertEqual(clicks, ['click', 'click'])
            self.assertNotIn('Enter', keys)

    async def test_ui_notice_snapshot_uses_real_dom_and_ignores_quoted_user_quota(self):
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(channel='chrome', headless=True)
            try:
                page = await browser.new_page()
                await page.set_content('<user-query><span>You have reached your limit</span></user-query><div role="alert">You have reached your limit for Thinking</div>')
                snapshot = await server.read_response_snapshot(page)
                self.assertEqual(snapshot['notice'], 'You have reached your limit for Thinking\nYou have reached your limit for Thinking')
                await page.set_content('<user-query><span>You have reached your limit</span></user-query>')
                self.assertEqual((await server.read_response_snapshot(page))['notice'], '')
            finally:
                await browser.close()
