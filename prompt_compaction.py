from hashlib import sha256
import re
from typing import Any, Dict


HARNESS_HEADER = "You are an AI agent running in OpenCode, a coding agent harness. Help the user accomplish their goals using the tools you have available."

# Match known prose; customized descriptions remain verbatim.
TOOL_PROFILES = {
    "edit": ("4194c06076efa8b8e16cae1b8d7b40a6cc1fb3ba2721a28ba3f179ac2b3006eb",
             "Edit by exact text replacement. Preserve indentation; omit Read line-number prefixes in old/new text. Missing oldString fails. Match must be unique unless replaceAll=true; add context or use replaceAll for all occurrences."),
    "grep": ("7f673e0439d39f3baf03353a068df91bde20a9c189cc4e1625c18e2fd7c92400",
             "Search file contents with ripgrep regex/literal matching. Narrow with path/include. Return matching paths, line numbers and previews."),
    "question": ("55a1a0d541e626fc23d4e2e668a6d32afc3eb55731d42b858f20a0b2fba01e99",
                 "Ask for missing information, preferences or decisions. UI adds free text; do not add that option. Use multiple=true for multi-select. Put the recommendation first and append '(Recommended)' to its label."),
    "read": ("e6ecb6c162129bb9af6662e49fe1bf8c717fe73fce0b70231426355f364a0fc8",
             "Read files/directories, including text/images/PDFs; images/PDFs reach the model directly. Text uses 1-based '<line>: <content>' prefixes for reference only; directory entries use one line each. Use offset/limit; prefer one larger read or grep for large files."),
    "shell": ("4e87032485a0f8e19eb99a45b184721ab5b681ca4a1c13148e967b545f859f78",
              "Run commands on macOS/zsh; quote special/space-containing paths. Prefer dedicated tools. Large output is saved with a truncated preview; normally rely on truncation. Optional timeout; background has no default timeout, returns immediately and notifies on completion. Do not poll."),
    "skill": ("b45ee59fd180f1b01404677711795737a5f071425183340ed989caee37897628",
              "Load skill instructions/resources when the task matches. Use an available or explicitly referenced skill ID."),
    "subagent": ("c6406f07cecff85dd38da7ef321e959f130a8e651adf46fbe85cac497d7a6924",
                 "Spawn a child session; returned sessionID continues it. Include context/instructions for new sessions. Foreground (default) waits; background independent work returns immediately and notifies (do not sleep/poll/check). Agents: explore=codebase discovery with quick/medium/very thorough depth; general=research and multi-step tasks, including parallel independent work. A user-requested model goes in model, after looking up its exact ID; otherwise omit it."),
    "webfetch": ("bc5173cc7d22ac32926e4b90d0914831d453dd14ef1bcfb7c96bb96ccaf76e05",
                 "Read HTTP(S) as text/Markdown/HTML (default Markdown). Prefer targeted tools. Large responses may use a preview with complete content in managed storage."),
    "write": ("347c290331a543b24c5256bfdc4e2b0faa547219aee445e8aaf4ba8e860f0ba4",
              "Write/overwrite an entire file, creating missing parent directories. Use edit for partial changes."),
    "execute": ("8438f87eae16bf74441b8354b1583c8ebd942508df36e59476c6f4ab4534be39",
                "Run confined Code Mode JavaScript: fetch for HTTP; no imports, direct filesystem or timers. Only use tools in its catalog or search results; ignore tools outside Code Mode. Use exact paths/signatures; never infer/normalize names; preserve tools.<namespace>[\"tool-name\"](input). Prefer explicit return; otherwise return the final top-level expression. Await needed calls; pending calls stop at runtime end. Parallel independent calls with Promise.allSettled."),
}

SKILL_PROFILES = {
    "docx": ("bce58846750f6765dbc7257190c1f37cc57db5fe5495ca0ba2d49e4a2ffca29a",
             "Word documents/templates (.docx/.dotx): create/read/edit/format/redline/extract/convert, including professional reports/letters. For generic documents without Word/file format requested, prefer dedicated document/page tools if available. Exclude PDFs, spreadsheets, Google Docs and unrelated coding."),
    "ec-release": ("a75a691f6785cb6d56a69ee3d123a3502be45e04a7ed6149ad22ea5aceb15891",
                   "Approved EC PR squash-merge, mywork alignment, monitor through production and verify prod health. Only with explicit user authorization to merge and release through prod; never review-only or unapproved production."),
    "google-workspace": ("8559ca03435d22b6f901d46d09f3b8cf456f80715fd8f0f0bb2f317224e40152",
                         "Read before the first Drive/Docs/Sheets/Slides connector call creating/changing Google files, including follow-up edits. Use for explicit Google files/links and create/edit/format/copy/rename requests. For generic documents or Google files used only as source, use the assistant's own output type. Exclude read-only and Word/Excel/PowerPoint/PDF files."),
    "morning": ("209bb075ea5287ab9e7380ad1a2dabafa78b62ecc1f6dd584bd16855ac4e1ff3",
                "Morning brief as styled HTML or recurring weekday task; only explicit run/view/setup requests or /morning. General day/schedule/calendar questions do not trigger it."),
    "opencode": ("1323e63afc725dcfcf17fc730606e0e0d2d0363630cb426cafadd4db9c3ebeea",
                 "Any OpenCode question: behavior/use/configuration, V1/V2 migration, troubleshooting, SDK/clients/server/API/plugins/integrations, agents/commands/skills/tools/permissions/MCP/providers/models/themes/keybinds/formatters, CLI/TUI/desktop/web and contributing."),
    "pdf": ("d1906e2fc005cb80c70b054ca9a61cbcaa67f14c348463f18c7c3bd41e682d02",
            "Any PDF/.pdf work: read/extract text or tables, merge/split/rotate/watermark/create/fill/encrypt/decrypt, extract images/OCR. Trigger on a PDF mention or requested PDF output."),
    "pptx": ("897e8328dbb85f8e917aaf1132b5de3a15791e1b301d817ab486177c0aca60d5",
             "Use whenever .pptx/.potx is input/output: create/read/edit/format/merge/split presentations/templates, notes/comments/layouts; also deck/slides/presentation/PowerPoint requests and follow-ups using parsed content. For generic slide requests without a file type, prefer dedicated slide artifacts/skills if available."),
    "skill-creator": ("dc3522ad3e3e46453a411f9d4f55faa15828e312933e722c1be9e8e3a7712cab",
                      "Create/update/improve skills; test/evaluate performance, benchmark variants and optimize triggering accuracy."),
    "xlsx": ("efd008c86ccd7afe9019af48f04fcd5192414bef7f1f5c44d8454348db04ed2f",
             "Use when a spreadsheet file is main input/output (.xlsx/.xlsm/.csv/.tsv): create/read/edit/fix/clean/format/analyze/restructure/convert, including messy tables. Output must be a spreadsheet. Exclude Word/PDF/HTML reports, standalone scripts, database pipelines and Google Sheets integrations, even with tables."),
}

HARNESS_REPLACEMENTS = {
    "Use the write tool to create files or completely replace their content. Prefer using the edit tool for targeted changes.":
        "Use write to create/overwrite files; edit for partial changes.",
    "Use the edit tool for targeted changes to existing text files. It replaces the exact text in `oldString` with `newString`, and the values must differ. By default, `oldString` must occur exactly once. If it occurs multiple times, include more surrounding context to make it unique or set `replaceAll` to true to replace every occurrence.":
        "Edit replaces exact oldString with differing newString. Require one match; add context or replaceAll=true for all occurrences.",
}


def description_hash(text: str) -> str:
    return sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def compact_tool_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(definition)
    description = result.get("description")
    profile = TOOL_PROFILES.get(result.get("name"))
    if isinstance(description, str) and profile and description_hash(description) == profile[0]:
        result["description"] = profile[1]
    return result


def compact_skill_catalogs(text: str) -> str:
    def entry(match):
        identifier, description = match.group(2), match.group(3)
        profile = SKILL_PROFILES.get(identifier)
        if profile and description_hash(description) == profile[0]:
            return match.group(1) + profile[1] + match.group(4)
        return match.group(0)

    pattern = re.compile(
        r'(<skill>\s*<id>([^<>]+)</id>\s*<name>[^<>]+</name>\s*<description>)([^<]*)(</description>\s*</skill>)',
        re.DOTALL
    )
    opening, closing = '<available_skills>', '</available_skills>'
    output = []
    cursor = 0
    while (start := text.find(opening, cursor)) >= 0:
        end = text.find(closing, start + len(opening))
        if end < 0:
            break
        end += len(closing)
        output.extend((text[cursor:start], pattern.sub(entry, text[start:end])))
        cursor = end
    output.append(text[cursor:])
    return ''.join(output)


def compact_system_prompt(text: str) -> str:
    if not text.startswith(HARNESS_HEADER):
        return text
    boundary = min((position for marker in ("Instructions from:", "<env>")
                    if (position := text.find(marker)) >= 0), default=len(text))
    prefix, protected = text[:boundary], text[boundary:]
    before_code, separator, code = prefix.partition("# Code Mode")
    harness = re.search(r'(?ms)^# Harness\s*\n(.*?)(?=^# |\Z)', before_code)
    if harness:
        body = harness.group(1)
        for original, concise in HARNESS_REPLACEMENTS.items():
            body = re.sub(r'(?m)^- ' + re.escape(original) + r'$', '- ' + concise, body)
        before_code = before_code[:harness.start(1)] + body + before_code[harness.end(1):]
    return compact_skill_catalogs(before_code + separator + code) + protected
