from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import json
import asyncio
import time
import re
import os
from uuid import uuid4
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright, BrowserContext, Page
from markdownify import markdownify as md
from conversation_sync import ConversationSync, fingerprint
from prompt_compaction import compact_tool_definition, compact_system_prompt
from request_guard import RequestGuard, GuardStopped, upstream_notice
from static_metadata import outgoing_metadata

app = FastAPI()

# Service profile
SERVICE_DIR = Path.home() / ".gemini-service"
PROFILE_DIR = SERVICE_DIR / "chrome-profile"
LOGIN_FLAG = SERVICE_DIR / "logged-in"
SHOW_BROWSER = os.environ.get("GEMINI_SHOW_BROWSER", "0") == "1"
CHAT_MODEL = "00bx-gemini-chat"
TOOLS_MODEL = "00bx-gemini-web"

# Global state
context: BrowserContext = None
playwright_instance = None
is_ready = False
model_request_lock = asyncio.Lock()
send_stage = "idle"
last_ui_probe = {}
stop_on_failure = os.environ.get("GEMINI_STOP_ON_ERROR", "0") == "1"
request_guard = RequestGuard(
    int(os.environ["GEMINI_REQUEST_LIMIT"]) if os.environ.get("GEMINI_REQUEST_LIMIT") else None,
    os.environ.get("GEMINI_REQUEST_LEDGER"),
)


@dataclass
class BrowserSession:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sync: ConversationSync = field(default_factory=ConversationSync)
    page: Optional[Page] = None
    marker: Optional[tuple] = None
    users: int = 0
    tool_policy: Optional[str] = None
    tool_definitions: Dict[str, Dict] = field(default_factory=dict)
    last_response: Optional[str] = None
    repeated_responses: int = 0


browser_sessions: Dict[str, BrowserSession] = {}


class Message(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


class FunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict] = None


class Tool(BaseModel):
    type: str
    function: FunctionDef


class ChatRequest(BaseModel):
    model: Optional[str] = "00bx-gemini-web"
    messages: List[Message]
    stream: Optional[bool] = False
    timeout: Optional[int] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Any] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    session_id: Optional[str] = None
    reset_session: bool = False
    stream_options: Optional[Dict] = None

def get_content_text(content: Any) -> str:
    """Helper to extract text from string or list content"""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join([
            c.get("text", "") 
            for c in content 
            if isinstance(c, dict) and c.get("type") == "text"
        ])
    return ""

async def check_logged_in(page: Page) -> bool:
    try:
        await asyncio.sleep(3)
        url = urlparse(page.url)
        if url.scheme != "https" or url.hostname != "gemini.google.com":
            return False

        # Guest mode also has an editor; require the signed-in account control.
        account_control = page.locator(
            'a[href^="https://accounts.google.com/SignOutOptions"]:visible'
        ).first
        editor = page.locator('rich-textarea .ql-editor').first
        return (
            await account_control.is_visible()
            and await editor.is_visible()
            and await editor.get_attribute("contenteditable") == "true"
        )
    except Exception:
        return False


async def init_browser():
    global context, playwright_instance, is_ready
    is_ready = False
    
    SERVICE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    
    first_time = not LOGIN_FLAG.exists()
    
    if first_time:
        print("\n" + "="*50)
        print("  FIRST TIME SETUP - Please log into Google")
        print("="*50 + "\n")
    else:
        print("🚀 Starting service...")
    
    playwright_instance = await async_playwright().start()
    
    headless_args = [] if first_time or SHOW_BROWSER else ["--headless=new"]
    
    context = await playwright_instance.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        headless=False,
        channel="chrome",
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-extensions",
            *headless_args
        ],
        viewport={"width": 1280, "height": 900},
    )
    
    page = await context.new_page()
    await page.goto("https://gemini.google.com/app")
    
    if first_time:
        print("📌 Browser opened - please log into your Google account")
        print("   Waiting for login...\n")
        
        for i in range(150):
            if await check_logged_in(page):
                LOGIN_FLAG.write_text("ok")
                if SHOW_BROWSER:
                    print("\n✅ Login saved! Keeping browser visible.\n")
                    break
                print("\n✅ Login saved! Restarting in headless mode...\n")
                await page.close()
                await context.close()
                await playwright_instance.stop()
                return await init_browser()
            await asyncio.sleep(2)
            if i % 15 == 0 and i > 0:
                print(f"   Still waiting... ({i*2}s)")
        
        else:
            print("❌ Login timeout")
            return
    
    if not await check_logged_in(page):
        print("❌ Saved session is not authenticated. Opening browser for login...")
        LOGIN_FLAG.unlink(missing_ok=True)
        await page.close()
        await context.close()
        await playwright_instance.stop()
        return await init_browser()
    
    if SHOW_BROWSER:
        await page.bring_to_front()
        print("📌 Browser visible: use Gemini's model menu to choose a model.")
    else:
        await page.close()
    is_ready = True
    print("✅ Service ready!")
    print("🎯 API: http://localhost:8080/v1/chat/completions\n")


@asynccontextmanager
async def session_scope(key: Optional[str]):
    session = browser_sessions.get(key) if key is not None else None
    if session is None:
        session = BrowserSession()
        if key is not None:
            browser_sessions[key] = session
    session.users += 1
    try:
        async with session.lock:
            yield session
    finally:
        session.users -= 1
        if key is None:
            if session.page and not session.page.is_closed():
                await session.page.close()
        elif session.users == 0 and session.page is None and browser_sessions.get(key) is session:
            browser_sessions.pop(key, None)


async def page_marker(page: Page) -> tuple:
    observation = await page.evaluate("""() => {
        const replies = document.querySelectorAll('div[id^="model-response-message-content"]');
        const last = replies.length ? replies[replies.length - 1] : null;
        return {url: location.href, count: replies.length, last: last ? last.innerText : ''};
    }""")
    return observation["url"], observation["count"], fingerprint(observation["last"])


async def replace_session_page(session: BrowserSession) -> Page:
    session.sync.invalidate()
    session.marker = None
    session.tool_policy = None
    session.tool_definitions = {}
    if session.page and not session.page.is_closed():
        await session.page.close()
    session.page = None
    page = await context.new_page()
    try:
        await page.goto('https://gemini.google.com/app')
        await page.wait_for_selector('rich-textarea .ql-editor', timeout=15000)
        marker = await page_marker(page)
        url = urlparse(marker[0])
        if url.scheme != "https" or url.hostname != "gemini.google.com" or url.path.rstrip('/') != '/app' or marker[1] != 0:
            raise HTTPException(503, "Gemini did not open a fresh conversation")
    except BaseException:
        if not page.is_closed():
            await page.close()
        raise
    session.page = page
    return page


def select_tools(tools: Optional[List[Tool]], tool_choice: Any = None) -> List[Tool]:
    if tool_choice == "none":
        return []
    if tool_choice not in (None, "auto", "required") and not isinstance(tool_choice, dict):
        raise HTTPException(400, "Unsupported tool_choice")
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function" or not isinstance(tool_choice.get("function"), dict):
            raise HTTPException(400, "Unsupported tool_choice")
        name = tool_choice.get("function", {}).get("name")
        selected = [tool for tool in tools or [] if tool.function.name == name]
        if not selected:
            raise HTTPException(400, "Requested tool is not available")
        return selected
    if tool_choice == "required" and not tools:
        raise HTTPException(400, "tool_choice requires an available tool")
    return tools or []


def format_tools_prompt(tools: List[Tool], tool_choice: Any = None) -> str:
    if not tools:
        return ""
    instruction = "Use a listed tool only when needed; otherwise answer normally."
    if tool_choice == "required" or isinstance(tool_choice, dict):
        instruction = "Call a listed tool in this response."
    definitions = [compact_tool_definition(tool.function.model_dump(exclude_none=True)) for tool in tools]
    return (
        "The tools below run in an external client, outside Gemini. Produce JSON requests; do not execute commands or use built-in Gemini tools. Results arrive in the next message.\n"
        f"{instruction}\n"
        'Tool calls: JSON only, no prose/fences: {"tool_calls":[{"name":"tool_name","arguments":{}}]}. '
        "Use real arguments matching the supplied schemas.\n"
        f"Tools: {json.dumps(definitions, ensure_ascii=False, separators=(',', ':'))}"
    )


def get_chat_content(content: Any) -> str:
    parts = [content] if isinstance(content, str) else [
        item.get("text", "") for item in content or []
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return "\n".join(part for part in parts if not any(
        part.strip().startswith(f"<{tag}>") and part.strip().endswith(f"</{tag}>")
        for tag in ("system-update", "system-reminder")
    ))


def format_conversation(messages: List[Message], tools: Optional[List[Tool]] = None,
                        tool_choice: Any = None, chat_only: bool = False,
                        bindings: Optional[Dict[str, tuple]] = None) -> str:
    tools = [] if chat_only else select_tools(tools, tool_choice)
    forwarded = []
    for msg in messages:
        if chat_only and msg.role not in ("user", "assistant"):
            continue
        content = get_chat_content(msg.content) if chat_only else get_content_text(msg.content)
        if not chat_only and msg.role in ("system", "developer"):
            content = compact_system_prompt(content)
        if not chat_only and msg.role == "assistant" and msg.tool_calls:
            calls = [{"id": call.get("id"), "name": call.get("function", {}).get("name"),
                      "arguments": json.loads(call.get("function", {}).get("arguments", "{}"))}
                     for call in msg.tool_calls]
            content = json.dumps({"tool_calls": calls}, ensure_ascii=False, separators=(",", ":"))
        if content or msg.role == "tool":
            label = f"tool ({msg.name or 'tool'})" if msg.role == "tool" else msg.role
            if msg.role == "tool" and msg.tool_call_id:
                label += f" [{msg.tool_call_id}]"
                if bindings and msg.tool_call_id in bindings:
                    index, name = bindings[msg.tool_call_id]
                    label += f" (request {index + 1}: {name})"
            forwarded.append((label, content))
    if len(forwarded) == 1 and forwarded[0][0] == "user":
        conversation = forwarded[0][1]
    else:
        conversation = "\n\n".join(f"{role}: {content}" for role, content in forwarded)
    if tools and not chat_only:
        return format_tools_prompt(tools, tool_choice) + "\n\n" + conversation
    return conversation


def normalize_message(message: Message, chat_only: bool) -> Optional[Message]:
    if chat_only and message.role not in ("user", "assistant"):
        return None
    content = get_chat_content(message.content) if chat_only else get_content_text(message.content)
    calls = None
    if not chat_only and message.tool_calls:
        calls = []
        for call in message.tool_calls:
            function = call.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise HTTPException(400, "Invalid tool-call history")
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    raise HTTPException(400, "Invalid tool-call arguments")
            if not isinstance(arguments, dict):
                raise HTTPException(400, "Invalid tool-call arguments")
            calls.append({"id": call.get("id"), "type": "function", "function": {
                "name": function["name"], "arguments": json.dumps(arguments, sort_keys=True, ensure_ascii=False)
            }})
    if not content and not calls and message.role != "tool":
        return None
    return Message(role=message.role, content=content, tool_calls=calls,
                   tool_call_id=message.tool_call_id, name=message.name)


def message_fingerprint(message: Message) -> str:
    value = message.model_dump(exclude_none=True)
    for call in value.get("tool_calls") or []:
        call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return fingerprint(value)


def session_key(request: ChatRequest, http_request: Optional[Request], chat_only: bool) -> Optional[str]:
    headers = http_request.headers if http_request else {}
    # The affinity header can refer to a parent; prefer the actual UI session ID.
    explicit = headers.get("x-opencode-session-id")
    if explicit is not None and request.session_id is not None and explicit != request.session_id:
        raise HTTPException(400, "Conflicting session IDs")
    identity = explicit if explicit is not None else request.session_id
    if identity is None:
        identity = headers.get("x-opencode-session")
    if identity is None:
        identity = headers.get("x-session-id")
    if identity is None:
        return None
    if not identity.strip() or len(identity) > 256:
        raise HTTPException(400, "Invalid session ID")
    channel = headers.get("x-gemini-channel")
    if channel is not None and (not channel.strip() or len(channel) > 256):
        raise HTTPException(400, "Invalid session channel")
    if channel is None:
        channel = next((line.strip() for message in request.messages
                        if message.role in ("system", "developer")
                        for line in get_content_text(message.content).splitlines() if line.strip()), "default")
    mode = "chat" if chat_only else "tools"
    return mode + ":" + fingerprint([identity, channel])


def prepare_prompt(request: ChatRequest, session: BrowserSession, chat_only: bool,
                   active_tools: List[Tool]):
    messages = [value for message in request.messages
                if (value := normalize_message(message, chat_only)) is not None]
    instructions = [message for message in messages if message.role in ("system", "developer")]
    history = [message for message in messages if message.role not in ("system", "developer")]
    plan = session.sync.plan(fingerprint([message_fingerprint(message) for message in instructions]),
                             tuple(message_fingerprint(message) for message in history), request.reset_session)
    forwarded = messages if plan.rebuild else history[plan.start:]
    if not chat_only:
        previous = () if plan.rebuild else [*instructions, *history[:plan.start]]
        forwarded = outgoing_metadata(forwarded, previous)
    bindings = {call["id"]: (index, call["function"]["name"])
                for message in history for index, call in enumerate(message.tool_calls or []) if call.get("id")}
    text = format_conversation(forwarded, chat_only=chat_only, bindings=bindings)
    catalog = None if plan.rebuild else session.sync.catalog
    previous_definitions = {} if plan.rebuild else session.tool_definitions
    if active_tools:
        definitions = request.tools or active_tools
        if len({tool.function.name for tool in definitions}) != len(definitions):
            raise HTTPException(400, "Duplicate tool names")
        current_definitions = {tool.function.name: tool.function.model_dump(exclude_none=True)
                               for tool in definitions}
        current_catalog = fingerprint(current_definitions)
        if current_catalog != catalog:
            if catalog is None:
                protocol = format_tools_prompt(definitions, request.tool_choice)
            else:
                changed = [compact_tool_definition(value) for name, value in current_definitions.items()
                           if previous_definitions.get(name) != value]
                removed = [name for name in previous_definitions if name not in current_definitions]
                protocol = "Update the tool catalogue: " + json.dumps(
                    {"replace": changed, "remove": removed}, ensure_ascii=False, separators=(',', ':'))
            text = protocol + "\n\n" + text
            catalog = current_catalog
        policy = fingerprint([sorted(tool.function.name for tool in active_tools), request.tool_choice])
        if (not plan.rebuild and policy != session.tool_policy) or isinstance(request.tool_choice, dict):
            names = json.dumps([tool.function.name for tool in active_tools])
            required = request.tool_choice == "required" or isinstance(request.tool_choice, dict)
            instruction = "Request a tool in JSON." if required else "Use tools only if needed; otherwise reply normally."
            text = f"Available tools: {names}. {instruction}\n\n" + text
        elif request.tool_choice == "required" and not plan.rebuild:
            text = "Request a tool in JSON for this turn.\n\n" + text
    elif not chat_only and catalog is not None:
        policy = fingerprint([[], request.tool_choice])
        if policy != session.tool_policy:
            text = "Tools are disabled; reply normally until tools are enabled again.\n\n" + text
    return text, plan, catalog


async def read_response_snapshot(page: Page) -> Dict[str, Any]:
    return await page.evaluate(r'''() => {
        const replies = [...document.querySelectorAll('div[id^="model-response-message-content"]')];
        const reply = replies[replies.length - 1];
        const notices = [...document.querySelectorAll('[role="alert"], [role="dialog"], snack-bar-container')]
            .filter(el => el.getClientRects().length > 0).map(el => el.innerText);
        const limitText = /(?:reached|hit|exceeded).{0,80}(?:limit|quota)|too many requests|quota exceeded|resource_exhausted|(?:已達|已超過|達到).{0,30}(?:使用上限|用量上限|配額|要求上限)/i;
        for (const el of document.querySelectorAll('body *')) {
            if (!el.children.length && el.getClientRects().length
                && !el.closest('user-query, model-response, .user-query-container')
                && limitText.test(el.innerText || '')) notices.push(el.innerText);
        }
        const notice = notices.join('\n');
        if (!reply) return {count: 0, finished: false, notice};
        const container = reply.closest('model-response');
        const hasAction = name => !!container && [...container.querySelectorAll(
            `button [data-mat-icon-name="${name}"]`
        )].some(icon => icon.closest('button').getClientRects().length > 0);
        return {
            count: replies.length,
            notice,
            id: reply.id,
            html: reply.innerHTML,
            text: reply.innerText || reply.textContent || '',
            finished: reply.getAttribute('aria-busy') === 'false'
                && hasAction('thumb_up') && hasAction('copy')
        };
    }''')


async def wait_for_complete_response(page: Page, previous_count: int, timeout: int) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    previous = None
    stable_count = 0
    while time.monotonic() < deadline:
        try:
            snapshot = await read_response_snapshot(page)
        except Exception:
            previous = None
            stable_count = 0
        else:
            notice = upstream_notice(snapshot.get('notice', ''))
            if snapshot['count'] > previous_count:
                notice = notice or upstream_notice(snapshot.get('text', ''))
            if notice:
                request_guard.stop(notice)
                raise GuardStopped(notice)
            if request_guard.stopped:
                raise GuardStopped(request_guard.stopped)
            # A pause in streaming is not a completed response.
            if snapshot['count'] > previous_count and snapshot['finished'] and snapshot.get('text', '').strip():
                current = (snapshot['id'], snapshot['html'], snapshot['text'])
                stable_count = stable_count + 1 if current == previous else 0
                previous = current
                if stable_count >= 2:
                    return snapshot
            else:
                previous = None
                stable_count = 0
        await asyncio.sleep(0.3)
    raise TimeoutError("Gemini response did not finish before the request timeout")


def observe_upstream_response(response):
    url = urlparse(response.url)
    if url.hostname == "gemini.google.com" and response.status == 429:
        request_guard.stop("upstream_http_429")


async def send_to_gemini(page: Page, text: str, timeout: Optional[int] = None) -> str:
    page.on("response", observe_upstream_response)
    try:
        return await send_monitored_prompt(page, text, timeout)
    finally:
        page.remove_listener("response", observe_upstream_response)


async def send_monitored_prompt(page: Page, text: str, timeout: Optional[int] = None) -> str:
    global send_stage, last_ui_probe
    send_stage = "editor"
    if timeout is None:
        timeout = 300  # 5 minute default
    
    await page.wait_for_selector('rich-textarea', timeout=10000)
    input_div = await page.wait_for_selector('rich-textarea .ql-editor', timeout=10000)
    
    # Count existing responses BEFORE sending
    existing_responses = await page.query_selector_all('div[id^="model-response-message-content"]')
    response_count_before = len(existing_responses)
    
    await input_div.click()
    await asyncio.sleep(0.1)
    await page.keyboard.press('Meta+A')
    await page.keyboard.press('Backspace')
    await asyncio.sleep(0.1)
    
    await page.evaluate('''(text) => {
        const editor = document.querySelector('rich-textarea .ql-editor');
        if (editor) {
            editor.focus();
            document.execCommand('insertText', false, text);
        }
    }''', text)
    await asyncio.sleep(0.3)
    
    send_stage = "send_control"
    last_ui_probe = await page.evaluate(r'''() => ({
        editorChars: (document.querySelector('rich-textarea .ql-editor')?.innerText || '').length,
        controls: [...document.querySelectorAll('button')].filter(button =>
            [...button.querySelectorAll('[data-mat-icon-name]')].some(icon => /send/i.test(icon.getAttribute('data-mat-icon-name')))
            || /send|傳送|发送/i.test(button.getAttribute('aria-label') || '')
        ).map(button => ({label: button.getAttribute('aria-label'), disabled: button.disabled,
            visible: !!button.getClientRects().length,
            icons: [...button.querySelectorAll('[data-mat-icon-name]')].map(icon => icon.getAttribute('data-mat-icon-name'))}))
    })''')
    print(f"🔎 Send stage:{send_stage} | UI:{json.dumps(last_ui_probe, ensure_ascii=False)}")
    send_button = await page.wait_for_selector(
        'button[aria-label="Send message"], button[aria-label="傳送訊息"], button[aria-label="发送消息"]',
        timeout=3000,
    )
    send_stage = "submit"
    request_guard.reserve()
    print(f"📊 Model requests:{request_guard.count} | limit:{request_guard.limit}")
    await send_button.click()
    
    await asyncio.sleep(1)
    
    send_stage = "response"
    completed = await wait_for_complete_response(page, response_count_before, timeout)
    send_stage = "complete"
    response_text = completed['html']
    text_content = completed['text'].strip()
    parsed_tools = parse_tool_calls(text_content)
    if parsed_tools:
        return json.dumps({"tool_calls": parsed_tools}, ensure_ascii=False)

    # No tool calls - convert HTML to markdown
    response_text = md(response_text, heading_style="ATX", code_language_callback=lambda el: next(
        (name[len('language-'):] for name in el.get('class', []) if name.startswith('language-')), ''
    ))
    
    return response_text.strip()


def parse_tool_calls(response: str) -> Optional[List[Dict]]:
    text = response.strip()
    if text.startswith("JSON\n"):
        text = text[5:].lstrip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    calls = data.get("tool_calls") if isinstance(data, dict) else None
    if not isinstance(calls, list) or not calls:
        return None
    if not all(isinstance(call, dict) and isinstance(call.get("name"), str)
               and isinstance(call.get("arguments"), dict) for call in calls):
        return None
    return [{"name": call["name"], "arguments": call["arguments"]} for call in calls]


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, http_request: Request = None):
    async with model_request_lock:
        try:
            request_guard.check()
            return await guarded_chat_completions(request, http_request)
        except GuardStopped as error:
            raise HTTPException(400, f"Gemini requests stopped: {error}")
        except BaseException as error:
            if request_guard.limit is not None or stop_on_failure:
                request_guard.stop(type(error).__name__)
                if isinstance(error, Exception):
                    raise HTTPException(400, "Gemini run stopped after a request failure") from error
            raise


async def guarded_chat_completions(request: ChatRequest, http_request: Request = None):
    if not is_ready:
        raise HTTPException(503, "Service initializing...")
    if not request.messages:
        raise HTTPException(400, "Messages required")
    chat_only = request.model == CHAT_MODEL
    active_tools = [] if chat_only else select_tools(request.tools, request.tool_choice)
    key = session_key(request, http_request, chat_only)
    async with session_scope(key) as session:
        if session.page is None or session.page.is_closed():
            session.sync.invalidate()
        elif session.marker != await page_marker(session.page):
            session.sync.invalidate()
        conversation, plan, catalog = prepare_prompt(request, session, chat_only, active_tools)
        if not conversation.strip():
            raise HTTPException(400, "No conversation content to forward")
        if plan.rebuild:
            await replace_session_page(session)
        page = session.page
        input_chars = (0 if plan.rebuild else session.sync.context_chars) + len(conversation)
        print(f"📥 Session:{key or 'isolated'} | sync:{plan.reason} | chars:{len(conversation)}")
        session.sync.invalidate()
        try:
            response = await send_to_gemini(page, conversation, request.timeout)
            response_hash = fingerprint(response.strip())
            session.repeated_responses = session.repeated_responses + 1 if response_hash == session.last_response else 1
            session.last_response = response_hash
            if (stop_on_failure or request_guard.limit is not None) and session.repeated_responses >= 3:
                request_guard.stop("repeated_response_possible_usage_limit")
                raise GuardStopped(request_guard.stopped)
            print(f"📤 Session:{key or 'isolated'} | chars:{len(response)}")
            
            # Check for tool calls
            tool_calls = None
            finish_reason = "stop"
            
            if active_tools:
                parsed_tools = parse_tool_calls(response)
                if parsed_tools:
                    allowed_names = {tool.function.name for tool in active_tools}
                    if any(call["name"] not in allowed_names for call in parsed_tools):
                        raise HTTPException(502, "Gemini requested an unavailable tool")
                    tool_calls = []
                    for i, tc in enumerate(parsed_tools):
                        tool_calls.append({
                            "id": f"call_{uuid4().hex}",
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": json.dumps(tc.get("arguments", {}))
                            }
                        })
                    finish_reason = "tool_calls"
                    print(f"🔧 Tool calls detected: {len(tool_calls)}")
                elif '"tool_calls"' in response and response.lstrip().startswith(("{", "```json", "JSON\n")):
                    raise HTTPException(502, "Gemini returned malformed tool-call JSON")
                elif request.tool_choice == "required" or isinstance(request.tool_choice, dict):
                    raise HTTPException(502, "Gemini did not return the required tool call")
            
            # Build response message
            msg = {"role": "assistant"}
            if tool_calls:
                msg["tool_calls"] = tool_calls
                msg["content"] = None
            else:
                msg["content"] = response
            
            result = {
                "id": f"chatcmpl-{uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model or "gemini-pro",
                "choices": [{
                    "index": 0,
                    "message": msg,
                    "finish_reason": finish_reason
                }],
                "usage": {
                    "prompt_tokens": input_chars // 4,
                    "completion_tokens": len(response) // 4,
                    "total_tokens": input_chars // 4 + len(response) // 4
                }
            }
            
            session.marker = await page_marker(page)
            assistant = normalize_message(Message(**msg), chat_only)
            session.sync.commit(plan, message_fingerprint(assistant) if assistant else None,
                                catalog, input_chars + len(response))
            session.tool_policy = fingerprint([sorted(tool.function.name for tool in active_tools), request.tool_choice])
            if active_tools:
                session.tool_definitions = {tool.function.name: tool.function.model_dump(exclude_none=True)
                                            for tool in request.tools or active_tools}

            # Handle streaming
            if request.stream:
                async def generate():
                    chunk_id = result["id"]
                    created = int(time.time())
                    model = request.model or "gemini-pro"
                    
                    if tool_calls:
                        # First chunk with role
                        first_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": None},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(first_chunk)}\n\n"
                        
                        # Stream each tool call
                        for i, tc in enumerate(tool_calls):
                            tc_chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [{
                                            "index": i,
                                            "id": tc["id"],
                                            "type": "function",
                                            "function": {
                                                "name": tc["function"]["name"],
                                                "arguments": tc["function"]["arguments"]
                                            }
                                        }]
                                    },
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(tc_chunk)}\n\n"
                    else:
                        # Stream content - first chunk with role
                        first_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(first_chunk)}\n\n"
                        
                        # Content chunk
                        content_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": response},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(content_chunk)}\n\n"
                    
                    # Final chunk with finish_reason
                    done_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason
                        }]
                    }
                    yield f"data: {json.dumps(done_chunk)}\n\n"
                    if request.stream_options and request.stream_options.get("include_usage"):
                        usage_chunk = {"id": chunk_id, "object": "chat.completion.chunk",
                                       "created": created, "model": model, "choices": [], "usage": result["usage"]}
                        yield f"data: {json.dumps(usage_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                
                return StreamingResponse(generate(), media_type="text/event-stream")
            
            return result
            
        except (HTTPException, GuardStopped):
            raise
        except Exception as e:
            if request_guard.limit is not None or stop_on_failure:
                reason = "response_timeout_possible_usage_limit" if isinstance(e, TimeoutError) else "bridge_" + type(e).__name__
                request_guard.stop(reason)
            print(f"❌ Gemini request failed: {type(e).__name__}")
            raise HTTPException(502, "Gemini request failed; the context will be rebuilt on the next request")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": TOOLS_MODEL, "object": "model", "created": int(time.time()), "owned_by": "google"},
            {"id": CHAT_MODEL, "object": "model", "created": int(time.time()), "owned_by": "google"},
        ]
    }


@app.get("/v1/request-budget")
async def request_budget():
    return request_guard.status()


@app.get("/health")
async def health():
    return {"status": "ready" if is_ready else "initializing",
            "sessions": sum(session.page is not None for session in browser_sessions.values())}


@app.post("/reset")
async def reset():
    LOGIN_FLAG.unlink(missing_ok=True)
    return {"status": "Login cleared. Restart to login again."}


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in browser_sessions:
        raise HTTPException(404, "Session not found")
    async with session_scope(session_id) as session:
        if session.page is None:
            raise HTTPException(404, "Session not found")
        if not session.page.is_closed():
            await session.page.close()
        session.page = None
        session.marker = None
        session.sync = ConversationSync()
        return {"status": "deleted", "session_id": session_id}


@app.get("/v1/sessions")
async def list_sessions():
    keys = [key for key, session in browser_sessions.items() if session.page is not None]
    return {"sessions": keys, "count": len(keys)}


@app.on_event("startup")
async def startup():
    print("\n" + "="*50)
    print("  00BX GEMINI API SERVICE")
    print("  OpenAI-Compatible API for Gemini Web")
    print("="*50)
    await init_browser()


@app.on_event("shutdown")
async def shutdown():
    global context, playwright_instance, is_ready
    is_ready = False
    for session in browser_sessions.values():
        try:
            if session.page and not session.page.is_closed():
                await session.page.close()
        except Exception:
            pass
    browser_sessions.clear()
    try:
        if context:
            await context.close()
    finally:
        if playwright_instance:
            await playwright_instance.stop()
