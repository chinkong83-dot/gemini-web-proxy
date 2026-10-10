from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import json
import asyncio
import time
import re
import os
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright, BrowserContext, Page
from markdownify import markdownify as md

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
session_pages: Dict[str, Page] = {}
page_locks: Dict[str, asyncio.Lock] = {}
session_msg_count: Dict[str, int] = {}  # Track message count per session


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


async def get_or_create_session_page(session_id: str, start_new_chat: bool = False) -> Page:
    global session_pages, page_locks
    
    if session_id not in session_pages:
        print(f"  → New session: {session_id}")
        
        page = await context.new_page()
        await page.goto('https://gemini.google.com/app')
        await asyncio.sleep(2)
        
        await page.wait_for_selector('rich-textarea', timeout=15000)
        
        # Always start fresh for new session
        try:
            new_chat_btn = await page.query_selector('side-nav-action-button[data-test-id="new-chat-button"]')
            if new_chat_btn:
                await new_chat_btn.click()
                await asyncio.sleep(1)
        except:
            pass
        
        await page.wait_for_selector('rich-textarea', timeout=10000)
        session_pages[session_id] = page
        page_locks[session_id] = asyncio.Lock()
        print(f"  ✓ Session {session_id} ready")
    else:
        # Check if page is still valid
        try:
            _ = session_pages[session_id].url
        except:
            del session_pages[session_id]
            if session_id in page_locks:
                del page_locks[session_id]
            return await get_or_create_session_page(session_id, start_new_chat)
        
        # Start new chat only if explicitly requested (new conversation)
        if start_new_chat:
            print(f"  → Starting new chat in session {session_id}")
            try:
                page = session_pages[session_id]
                new_chat_btn = await page.query_selector('side-nav-action-button[data-test-id="new-chat-button"]')
                if new_chat_btn:
                    await new_chat_btn.click()
                    await asyncio.sleep(1)
                    await page.wait_for_selector('rich-textarea', timeout=10000)
            except:
                pass
    
    return session_pages[session_id]


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
    definitions = [tool.function.model_dump(exclude_none=True) for tool in tools]
    return (
        "An external OpenCode client executes JSON tool requests and returns results. "
        "You only produce the request; do not execute commands yourself.\n"
        f"{instruction}\n"
        'For tool calls, return only JSON, without prose or Markdown: {"tool_calls":[{"name":"tool_name","arguments":{}}]}.\n'
        "Follow the supplied parameter schemas; put actual values in arguments.\n"
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
                        tool_choice: Any = None, chat_only: bool = False) -> str:
    tools = [] if chat_only else select_tools(tools, tool_choice)
    forwarded = []
    for msg in messages:
        if chat_only and msg.role not in ("user", "assistant"):
            continue
        content = get_chat_content(msg.content) if chat_only else get_content_text(msg.content)
        if not chat_only and msg.role == "assistant" and msg.tool_calls:
            calls = [{"id": call.get("id"), "name": call.get("function", {}).get("name"),
                      "arguments": json.loads(call.get("function", {}).get("arguments", "{}"))}
                     for call in msg.tool_calls]
            content = json.dumps({"tool_calls": calls}, ensure_ascii=False, separators=(",", ":"))
        if content:
            label = f"tool ({msg.name or 'tool'})" if msg.role == "tool" else msg.role
            if msg.role == "tool" and msg.tool_call_id:
                label += f" [{msg.tool_call_id}]"
            forwarded.append((label, content))
    if len(forwarded) == 1 and forwarded[0][0] == "user":
        conversation = forwarded[0][1]
    else:
        conversation = "\n\n".join(f"{role}: {content}" for role, content in forwarded)
    if tools and not chat_only:
        return format_tools_prompt(tools, tool_choice) + "\n\n" + conversation
    return conversation


async def send_to_gemini(page: Page, text: str, timeout: Optional[int] = None) -> str:
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
    
    try:
        send_button = await page.wait_for_selector('button[aria-label="Send message"]', timeout=3000)
        await send_button.click()
    except:
        await page.keyboard.press('Enter')
    
    await asyncio.sleep(1)
    
    response_text = None
    start_time = time.time()
    previous_length = 0
    stable_count = 0
    
    while (time.time() - start_time) < timeout:
        try:
            response_divs = await page.query_selector_all('div[id^="model-response-message-content"]')
            
            # Only process if we have a NEW response
            if len(response_divs) > response_count_before:
                # Get the newest response (last one)
                last_response = response_divs[-1]
                current_text = await last_response.evaluate('(el) => el.innerHTML')
                
                if current_text:
                    if len(current_text) == previous_length:
                        stable_count += 1
                        if stable_count >= 2:
                            response_text = current_text
                            break
                    else:
                        previous_length = len(current_text)
                        stable_count = 0
        except:
            pass
        await asyncio.sleep(0.3)
    
    if not response_text:
        raise Exception(f"Timeout after {timeout}s")
    
    # Preserve JSON text before converting ordinary replies to Markdown.
    extraction = await page.evaluate('''() => {
        const lastResponse = [...document.querySelectorAll('div[id^="model-response-message-content"]')].pop();
        if (!lastResponse) return { text: '' };
        const text = lastResponse.innerText || lastResponse.textContent || '';
        return { text };
    }''')
    
    text_content = extraction.get("text", "").strip()
    parsed_tools = parse_tool_calls(text_content)
    if parsed_tools:
        return json.dumps({"tool_calls": parsed_tools}, ensure_ascii=False)

    # No tool calls - convert HTML to markdown
    response_text = md(response_text, heading_style="ATX", code_language_callback=lambda el: el.get('class', [''])[0].replace('language-', '') if el.get('class') else '')
    
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
async def chat_completions(request: ChatRequest):
    if not is_ready:
        raise HTTPException(503, "Service initializing...")
    
    if not request.messages:
        raise HTTPException(400, "Messages required")

    chat_only = request.model == CHAT_MODEL
    active_tools = [] if chat_only else select_tools(request.tools, request.tool_choice)
    conversation = format_conversation(
        request.messages, active_tools, request.tool_choice, chat_only=chat_only
    )
    if not conversation.strip():
        raise HTTPException(400, "No conversation content to forward")
    
    # Use system prompt hash for session ID
    session_id = "default"
    for msg in request.messages:
        content_text = get_content_text(msg.content)
        if msg.role == "system" and content_text:
            session_id = str(hash(content_text[:100]))[:8]
            break
    session_id = ("chat:" if chat_only else "tools:") + session_id
    
    # Detect new conversation: compare current message count with what we've seen
    # New conversation = message count reset or decreased (OpenCode started fresh)
    current_msg_count = len(request.messages)
    prev_msg_count = session_msg_count.get(session_id, 0)
    
    # It's a new conversation if:
    # - We've never seen this session, OR
    # - Message count is less than before (conversation was reset)
    is_new_conversation = session_id not in session_msg_count or current_msg_count < prev_msg_count
    
    if is_new_conversation:
        print(f"  🆕 New conversation detected (msgs: {current_msg_count}, prev: {prev_msg_count})")
    
    # Update tracked count
    session_msg_count[session_id] = current_msg_count
    
    page = await get_or_create_session_page(session_id, start_new_chat=is_new_conversation)
    
    async with page_locks[session_id]:
        print(f"\n📥 [{time.strftime('%H:%M:%S')}] Session:{session_id} | msgs:{current_msg_count} | {conversation[:60]}...")
        
        try:
            response = await send_to_gemini(page, conversation, request.timeout)
            print(f"📤 [{time.strftime('%H:%M:%S')}] {response[:80]}...")
            
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
                            "id": f"call_{int(time.time())}_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": json.dumps(tc.get("arguments", {}))
                            }
                        })
                    finish_reason = "tool_calls"
                    print(f"🔧 Tool calls detected: {[tc['function']['name'] for tc in tool_calls]}")
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
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model or "gemini-pro",
                "choices": [{
                    "index": 0,
                    "message": msg,
                    "finish_reason": finish_reason
                }],
                "usage": {
                    "prompt_tokens": len(conversation) // 4,
                    "completion_tokens": len(response) // 4,
                    "total_tokens": (len(conversation) + len(response)) // 4
                }
            }
            
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
                    yield "data: [DONE]\n\n"
                
                return StreamingResponse(generate(), media_type="text/event-stream")
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            raise HTTPException(500, str(e))


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": TOOLS_MODEL, "object": "model", "created": int(time.time()), "owned_by": "google"},
            {"id": CHAT_MODEL, "object": "model", "created": int(time.time()), "owned_by": "google"},
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ready" if is_ready else "initializing", "sessions": len(session_pages)}


@app.post("/reset")
async def reset():
    LOGIN_FLAG.unlink(missing_ok=True)
    return {"status": "Login cleared. Restart to login again."}


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in session_pages:
        try:
            await session_pages[session_id].close()
        except:
            pass
        del session_pages[session_id]
        if session_id in page_locks:
            del page_locks[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(404, "Session not found")


@app.get("/v1/sessions")
async def list_sessions():
    return {"sessions": list(session_pages.keys()), "count": len(session_pages)}


@app.on_event("startup")
async def startup():
    print("\n" + "="*50)
    print("  00BX GEMINI API SERVICE")
    print("  OpenAI-Compatible API for Gemini Web")
    print("="*50)
    await init_browser()


@app.on_event("shutdown")
async def shutdown():
    global context, playwright_instance
    for page in session_pages.values():
        try:
            await page.close()
        except:
            pass
    if context:
        await context.close()
    if playwright_instance:
        await playwright_instance.stop()
