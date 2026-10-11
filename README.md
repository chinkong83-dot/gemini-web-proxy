# Gemini Web Proxy for OpenCode

A local browser automation proxy that exposes Gemini's web interface as an OpenAI-compatible API for OpenCode. It uses your existing Google login instead of a Google API key. Your account's Gemini availability and usage limits still apply.

**Educational Purpose**: This project is created for learning purposes to demonstrate how to build custom providers for OpenCode using browser automation.

## What This Does

- Converts Gemini web interface into an OpenAI-compatible API
- Uses Gemini in OpenCode without a Google API key
- Translates tool calls for OpenCode to execute (experimental)
- Maintains persistent login sessions
- Runs the proxy locally; conversation content is sent to Google's Gemini service

## How It Works

1. Uses Playwright to automate the Gemini web interface
2. Translates OpenCode requests to Gemini web format
3. Extracts responses and converts them back to OpenAI format
4. Uses a short JSON protocol for external tool execution in Tools mode
5. Maintains session state for continuous conversations

## Requirements

- Python 3.11 (tested and recommended for the pinned dependencies)
- Google account with Gemini access
- Chrome browser installed
- OpenCode installed

## Installation

### Quick Setup (Recommended)

```bash
git clone <repository-url>
cd gemini-web-proxy
chmod +x setup.sh
python3.11 -m venv venv
PYTHON=venv/bin/python ./setup.sh
```

If your Python 3.11 executable has another name, use that executable to create
the virtual environment. Check it with `python --version` or `python3 --version`;
these commands may point to different installations. The `PYTHON` variable
selects the interpreter used by setup. Without it, setup uses `python3`.
The minimum-version check does not guarantee that every newer Python version
is compatible with the pinned dependencies. Setup stops if an install fails.

The `venv/` directory is ignored by Git and contains only the local environment.

### Manual Setup

### Step 1: Clone and Setup

```bash
git clone <repository-url>
cd gemini-web-proxy
```

### Step 2: Install Dependencies

```bash
# Create an isolated environment and install Python dependencies
python3.11 -m venv venv
venv/bin/python -m pip install -r requirements.txt

# Install Playwright browsers
venv/bin/python -m playwright install chromium
```

### Step 3: First Run and Login

```bash
venv/bin/python run.py
```

On first run:
1. Chrome browser will open automatically
2. Log in to your Google account
3. Navigate to Gemini if not redirected automatically
4. Wait for "Login saved" message
5. Browser will restart in headless mode

Gemini also offers a guest editor, so seeing an input box does not prove you
are signed in. The proxy requires a visible signed-in Google account control
and an editable chat box before saving login. If a saved session fails this
check, it opens the login browser again without deleting the Chrome profile.

### Step 4: Configure OpenCode

Add this configuration to your OpenCode config file:

**Location**: `~/.config/opencode/opencode.json` or your project's `opencode.json`

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "00bx-gemini": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Gemini Web (local proxy)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "00bx-gemini-web": {
          "name": "Gemini Web Tools",
          "tool_call": true,
          "modalities": {"input": ["text"], "output": ["text"]}
        },
        "00bx-gemini-chat": {
          "name": "Gemini Web Chat",
          "tool_call": false,
          "modalities": {"input": ["text"], "output": ["text"]}
        }
      }
    }
  }
}
```

Merge the `provider` entry into any existing configuration instead of replacing
your other providers or plugins. OpenCode uses the singular key `provider`,
not `providers`. See the [official custom provider documentation](https://opencode.ai/docs/providers/#custom-provider).

`not-needed` is a placeholder, not a Google credential. Do not enter a Google
password, cookie, or token here. This proxy currently does not validate API keys.

Restart OpenCode after editing its configuration, then select the provider with:

```bash
opencode -m 00bx-gemini/00bx-gemini-web
```

You can also use `/models` in OpenCode. The configured model ID is a proxy alias;
it does not select a Gemini model. The browser's current Gemini model selection
determines which model answers.

Choose a mode explicitly; the proxy does not infer it from your topic:

- `00bx-gemini/00bx-gemini-chat` (**Gemini Web Chat**) forwards only user and
  assistant text. It omits system/developer instructions, tool definitions,
  tool results, and whole text parts enclosed in OpenCode's `system-update` or
  `system-reminder` control tags. It never returns executable tool calls.
- `00bx-gemini/00bx-gemini-web` (**Gemini Web Tools**) forwards OpenCode's
  instructions and conversation. When tools are enabled, it adds a short JSON
  tool-call protocol and the supplied function schemas. `tool_choice: "none"`
  disables that protocol for the current request.

Start a new conversation after changing modes. Plain single-user requests are
forwarded unchanged; multi-message conversations keep short role labels.

## Usage

### Starting the Proxy

```bash
venv/bin/python run.py
```

The proxy will start on `http://127.0.0.1:8080` and listen only on `127.0.0.1`.
Start one proxy process; leave it running while using OpenCode.
The Chrome process must keep running, but its window does not need to be visible.
The default mode hides the browser after login. Do not manually close the proxy's
browser or session tabs while requests are being processed.

### Showing or Hiding Chrome

To temporarily hide a visible window without restarting the proxy, click the
yellow minimize button in the Chrome window on macOS. The browser keeps running;
do not close its session tabs or quit Chrome.

To change between a visible browser and headless operation, stop the proxy with
`Ctrl+C` in the terminal where it is running, then use one of these commands:

```bash
# Show the proxy's Chrome window
GEMINI_SHOW_BROWSER=1 venv/bin/python run.py

# Hide the window and run Chrome in the background
GEMINI_SHOW_BROWSER=0 venv/bin/python run.py
```

With `GEMINI_SHOW_BROWSER` unset, the default is headless operation after login.
This setting is read at startup; it cannot be toggled while the proxy is running.
Restarting preserves the Chrome login profile but interrupts in-flight requests
and clears the proxy's in-memory session mapping. If login is missing or expired,
the login window still opens so you can sign in.

### Choosing the Gemini Model

Start the proxy with `GEMINI_SHOW_BROWSER=1` as shown above.
In that Chrome window, click the model name inside Gemini's input box and select
an available model. Use the proxy's window, not a different Chrome profile.
The window stays visible after login. Existing proxy session pages have their
own model selection; change it in the page handling that conversation, or start
a new OpenCode conversation. Verify the selection in any new Gemini page before
sending a prompt, since this proxy does not enforce a model across pages.

Gemini's actual context capacity is controlled by Google and your account's
plan. Changing OpenCode's model metadata does not change that capacity.

### Testing the Connection

```bash
curl http://127.0.0.1:8080/health

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "00bx-gemini-web",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Using with OpenCode

For a simple connection test, send a plain-text request before testing tools:

```bash
opencode run -m 00bx-gemini/00bx-gemini-web "Reply with exactly OK. Do not use tools."
```

Requests using this provider are sent through the proxy to Gemini's web interface.
Other OpenCode providers keep their own routes.

Verified on macOS with Python 3.11.5:

- OpenCode CLI 1.1.25: provider discovery and an isolated plain-text reply `OK`.
- OpenCode 2.0.26 (the CLI bundled with the desktop app): Gemini returned two
  shell tool calls, OpenCode executed `gh --version` and `aws --version` with
  exit code 0, and Gemini correctly reported the returned version strings.
- Offline checks cover explicit Chat mode, `tool_choice`, complete parameter
  schemas, JSON/code argument preservation, streaming tool calls, and rejection
  of unavailable tools.

The tool test used a temporary workspace, allowed only those two version
commands, disabled sharing and title generation, and bounded the live requests.
It did not call GitHub/AWS resource APIs, validate every tool, or test the user's
other plugins. A tool run normally needs a model request to choose tools and
another model request to read their results.

## Features

- **Google Login**: Uses the browser session instead of a Google API key
- **Tool Calling**: Experimental translation of Gemini output into OpenCode tool calls
- **Persistent Sessions**: Reuses saved login until Google expires or revokes it
- **Local Proxy**: Browser automation runs locally; inference runs on Google's service
- **OpenAI Compatible**: Works with any OpenAI-compatible client

## Troubleshooting

### Python Version Check

The setup script compares Python's version tuple, not decimal numbers. Python
3.10 and later must not be compared as decimals against `3.8`. Installation
and startup should use the same virtual environment interpreter.

### Login Issues

If login fails or expires:

```bash
# Stop the proxy first, then remove its local login profile
rm -rf ~/.gemini-service
venv/bin/python run.py
```

### Connection Problems

1. Ensure Chrome is installed and accessible
2. Check that port 8080 is not in use
3. Verify your Google account has Gemini access and `/health` reports `ready`
4. Try restarting the proxy

To keep logs available from a shell, start the proxy with output redirected:

```bash
mkdir -p ~/.gemini-service
venv/bin/python -u run.py >> ~/.gemini-service/proxy.log 2>&1
```

Read them with `tail -n 50 ~/.gemini-service/proxy.log`. Request metadata includes
`Session`, `sync`, and `chars`; `sync:delta` means the existing conversation was
reused. Output sent only to a terminal is not automatically saved to a log file.

### OpenCode Configuration

Make sure your OpenCode configuration matches exactly:
- Provider name: `00bx-gemini`
- Model name: `00bx-gemini-web`
- Base URL: `http://127.0.0.1:8080/v1`
- Configuration key: `provider` (singular)

## Project Structure

```
gemini-web-proxy/
├── server.py          # Main proxy server
├── conversation_sync.py # Incremental context planner
├── prompt_compaction.py # Guarded metadata-description compaction
├── checks/           # Offline synchronization checks
├── run.py            # Startup script
├── requirements.txt  # Python dependencies
├── README.md        # This file
└── .gitignore       # Git ignore rules
```

## Technical Details

### Architecture

```
OpenCode → HTTP Request → Gemini Proxy → Browser Automation → Gemini Web
                                      ← Response Processing ←
```

### Session Management

The first request creates a Gemini conversation with the supplied instructions,
history, and tool definitions. Subsequent requests for the same OpenCode session
send only new messages or tool results. Unchanged system instructions (including
AGENTS.md) and tool definitions are not repeated. OpenCode still sends its full
snapshot to the local proxy; the proxy computes the delta before sending to Google.

Session identity comes from `x-opencode-session-id`, then the JSON `session_id`
field, then `x-opencode-session` or `x-session-id`. Conflicting explicit header
and body IDs are rejected. OpenCode 2.0.26 supplies the actual session header
automatically, including for custom providers. Different logical sessions get
separate Gemini tabs; multiple UI tabs showing the same session reuse its context.
The proxy does not infer session identity from the number of UI tabs.

Chat and Tools modes use separate contexts. Background tasks with a different
first system-instruction line, such as title generation, use a separate channel.
Custom clients can set `x-gemini-channel` to distinguish parallel agents or tasks.
Do not reuse one session ID/channel for unrelated conversations. Without a session
ID, each request uses a fresh Gemini conversation and forwards the full snapshot;
older clients may need an upgrade to benefit from incremental forwarding.

Compaction, edits/undo, changed system instructions, explicit reset, a failed or
cancelled request, and observed changes to the owned Gemini page cause a fresh
conversation to be bootstrapped from the current snapshot. The proxy does not
automatically retry model requests. Same-session requests are serialized. Tool
results retain call IDs and request order without replaying Gemini's tool-call JSON.

For direct clients, keep `session_id` stable and supply the full current snapshot:

```json
{
  "model": "00bx-gemini-web",
  "session_id": "my-chat-1",
  "messages": [{"role": "user", "content": "Hello!"}]
}
```

Add `"reset_session": true` to rebuild that context on the next request. Runtime
session mappings are kept in memory: restarting the proxy bootstraps them again.
The login profile at `~/.gemini-service/` persists between runs. To close a runtime
tab, use its opaque key from `GET /v1/sessions` with `DELETE /v1/sessions/{key}`.

Offline checks (no Gemini requests):

```bash
venv/bin/python -m unittest discover -s checks -p '*_checks.py' -v
```

A local mock integration with the desktop's native OpenCode 2.0.26 CLI verified
first turn, session resume, a separate new session, and a shell-tool round trip.
Forwarded prompt lengths were 4,026 then 108 characters; returning the result of
`printf` needed 181 characters. AGENTS.md and unchanged tool definitions appeared
on the first turn only. OpenCode CLI 1.18.32 also passed the mock round trip.
These checks send no requests to Gemini; live long-context retention is unvalidated.

### Complete Replies

The proxy waits for the newest response's `aria-busy="false"` and visible copy
and feedback controls, then confirms the complete HTML and text remain stable.
A pause during generation is not treated as completion. HTML and text are read
from one snapshot so tool-call JSON and ordinary replies use the same finished
response. If completion cannot be confirmed before the request timeout, the
proxy returns an error without forwarding a partial answer or automatically
retrying the model request.

Prematurely reading a paused response can truncate a list or tool-call JSON.
If Gemini subsequently finishes that response, the page fingerprint changes
and the next request rebuilds the context. Completion checks prevent that
failure path while preserving detection of manual page changes. OpenCode title
generation still uses a separate Gemini conversation; it is an additional
background model request, not a second main conversation.

A bounded live check with OpenCode 2.0.26 received all five entries of a supplied
repository-name list. A separate tool/error-result round trip used `sync:delta`
with the same session key. Together these checks used three Gemini requests.
The live repository lookup could not authenticate in the isolated test environment:
temporary `XDG_CONFIG_HOME` also isolated gh's settings. The list check used names
already retrieved by the user, so it did not revalidate fresh GitHub access.

### Tool Calling

The proxy prompts Gemini to produce tool-call JSON, then translates the output
into OpenAI-compatible responses. OpenCode executes the tools according to its
permissions; the proxy itself does not execute their commands. Translation
does not guarantee that every tool call is correct.

The bridge preserves complete parameter schemas, JSON argument values, and
call IDs linking previous tool requests to their results. It
does not add access/permission claims, force tools for ordinary conversation,
or rewrite write/edit arguments with code-block placeholders. Only structured
tool-call JSON for currently available tools is returned as an executable call.
Complete parameter schemas and user rules remain unchanged in Tools mode.
The protocol identifies OpenCode as the external executor; Gemini is asked to
produce JSON requests rather than execute commands inside the web app. Actual
prompt size also includes client-supplied schemas and instructions.

### Prompt Compaction

The proxy shortens recognized OpenCode tool descriptions, two repeated Harness
instructions, and recognized descriptions in the initial skills directory.
Matching uses tool/skill IDs and fingerprints of known description text.
Unknown or customized descriptions pass through verbatim. Every parameter-schema
field is preserved, including descriptions, required fields, enums, defaults,
bounds and nested types. Code Mode instructions/signatures remain unchanged.

In Tools mode, skills keep their IDs, names and concise purposes/triggers.
Gemini can use that initial directory to select a skill on later turns; the
`skill` tool loads its full instructions, which are forwarded without compaction.
The directory is sent again when the context is rebuilt. User AGENTS.md sections,
environment details, user messages, tool results and control updates are preserved.
Chat mode continues to forward only conversation text with tools disabled.

For a sample with twelve tools and twelve skills, formatted bootstrap size
decreased from 25,977 to 19,023 characters: 6,954 fewer (26.8%). A native OpenCode
2.0.26 mock integration replaying that metadata verified session resume/isolation
and a `printf` tool round trip. Its first turn used 19,045 characters, the next
turn 218, and the tool-result turn 291; unchanged directories/schemas were not
repeated. Chat forwarded just its 13-character user message. These are character
counts, not tokenizer counts. The checks added no live Gemini requests; model
selection quality after compaction has not been evaluated with live Gemini.

Each tool round trip normally needs another Gemini response after OpenCode
returns the result. OpenCode may also request session titles or other background
responses. These requests consume the same account's usage; one user task does
not necessarily equal one Gemini prompt. Incremental forwarding reduces repeated
input transmission; Google still processes the retained conversation, so this
does not guarantee lower account usage.

For work accounts, check Gemini **Settings → Usage Limits** and the
[official Workspace limits](https://support.google.com/gemini/answer/14620100).
Limits depend on the license, prompt complexity, and conversation length, and
usage across devices shares the same allowance. This project cannot determine
company monitoring thresholds or guarantee that browser automation is approved
by your organization. Workspace activity retention is administrator-controlled.

### Security and Data Handling

- Google login is performed in the browser. The proxy uses a separate Chrome
  profile at `~/.gemini-service/chrome-profile` and saves a `logged-in` flag.
  The flag is not a credential; the profile contains sensitive browser state.
- Tools mode forwards user messages, system instructions, tool arguments, and
  tool-result text without secret redaction. Chat mode forwards the selected
  conversation text without secret redaction. If OpenCode reads a credential
  file in Tools mode, its output can be included in a prompt sent to Google.
- Request logs contain opaque session keys, synchronization reasons, character
  counts, and tool-call counts; prompt/response snippets are not logged.
- API routes have no authentication. Loopback binding limits network exposure,
  but other local clients can call the API using the saved Google session.
- Supplied session IDs separate logical conversations; they are not credentials
  and do not authenticate local clients. A caller using another client's ID and
  channel can access that runtime context.
- `/reset` removes only the `logged-in` flag. It does not clear cookies, close
  active browser sessions, or revoke the Google login.

## Limitations

- Requires active internet connection
- Dependent on Gemini web interface stability
- May need updates if Gemini changes their interface
- Rate limited by Gemini's web interface limits
- Session tabs remain open until explicitly deleted or the proxy stops; there
  is no automatic eviction
- History is replayed when rebuilding a context or when no session ID is supplied.
  Incremental forwarding reduces repeated input, but does not expand Google's
  context limit or prevent Gemini from forgetting earlier content.
- Usage is a rough character-count estimate of the retained browser context,
  not Google's tokenizer or billing. Streaming includes it when the client asks
  for `stream_options.include_usage`; it is not just the latest delta's size.
- Interrupted shutdown may log a Playwright driver-connection error even
  though the proxy process exits. Restarting the service worked in local tests.

## Contributing

This is an educational project. Feel free to fork and experiment.

## License

MIT License - See LICENSE file for details

## Disclaimer

This project is for educational purposes only. Users are responsible for complying with Google's terms of service. The authors are not responsible for any misuse or violations.
