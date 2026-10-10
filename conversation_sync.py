from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Optional, Tuple


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SyncPlan:
    instructions: str
    history: Tuple[str, ...]
    start: int
    rebuild: bool
    reason: str


@dataclass
class ConversationSync:
    instructions: Optional[str] = None
    history: Tuple[str, ...] = ()
    catalog: Optional[str] = None
    context_chars: int = 0
    dirty: bool = True

    def plan(self, instructions: str, history: Tuple[str, ...], reset: bool = False) -> SyncPlan:
        if reset:
            reason = "explicit-reset"
        elif self.instructions is None:
            reason = "new-session"
        elif self.dirty:
            reason = "unsynchronized-browser"
        elif instructions != self.instructions:
            reason = "instructions-changed"
        elif len(history) < len(self.history) or history[:len(self.history)] != self.history:
            reason = "history-rewritten"
        elif len(history) == len(self.history):
            reason = "regeneration"
        else:
            return SyncPlan(instructions, history, len(self.history), False, "delta")
        return SyncPlan(instructions, history, 0, True, reason)

    def commit(self, plan: SyncPlan, assistant: Optional[str], catalog: Optional[str], context_chars: int = 0) -> None:
        self.instructions = plan.instructions
        self.history = plan.history + ((assistant,) if assistant is not None else ())
        self.catalog = catalog
        self.context_chars = context_chars
        self.dirty = False

    def invalidate(self) -> None:
        self.dirty = True
