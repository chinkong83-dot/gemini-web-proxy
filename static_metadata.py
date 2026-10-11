import re
import xml.etree.ElementTree as ET


CATALOG = re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL)
UPDATE = re.compile(r"\s*<system-update>\s*(<available_skills>.*?</available_skills>)\s*</system-update>\s*", re.DOTALL)


def skill_catalog(text):
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    if root.tag != "available_skills" or root.attrib or (root.text or "").strip():
        return None
    for entry in root:
        if entry.tag != "skill" or entry.attrib or (entry.text or "").strip() or (entry.tail or "").strip() or [value.tag for value in entry] != ["id", "name", "description"]:
            return None
        if any(value.attrib or len(value) or (value.tail or "").strip() for value in entry):
            return None
    return text


def outgoing_metadata(messages, previous=()):
    latest = None
    for message in previous:
        if message.role in ("system", "developer"):
            for match in CATALOG.finditer(message.content or ""):
                latest = skill_catalog(match.group()) or latest
        elif message.role == "user" and (match := UPDATE.fullmatch(message.content or "")):
            latest = skill_catalog(match.group(1)) or latest
    outgoing = []
    for message in messages:
        if message.role in ("system", "developer"):
            for match in CATALOG.finditer(message.content or ""):
                latest = skill_catalog(match.group()) or latest
        elif message.role == "user" and (match := UPDATE.fullmatch(message.content or "")):
            current = skill_catalog(match.group(1))
            if current is not None and current == latest:
                continue
            latest = current or latest
        outgoing.append(message)
    return outgoing if outgoing or not messages else [messages[-1]]
