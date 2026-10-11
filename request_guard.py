import json
import os
import re
from pathlib import Path


class GuardStopped(Exception):
    pass


class RequestGuard:
    def __init__(self, limit=None, ledger=None):
        if limit is not None and limit < 1:
            raise ValueError("Request limit must be positive")
        if limit is not None and ledger is None:
            raise ValueError("A bounded run requires a persistent ledger")
        self.limit = limit
        self.ledger = Path(ledger) if ledger else None
        self.count = 0
        self.stopped = None
        if self.ledger and self.ledger.exists():
            state = json.loads(self.ledger.read_text())
            if state["limit"] != limit:
                raise ValueError("Ledger limit differs; use a new authorized run ledger")
            self.count = state["count"]
            self.stopped = state["stopped"]
            if not isinstance(self.count, int) or self.count < 0 or (limit is not None and self.count > limit):
                raise ValueError("Invalid ledger count")

    def status(self):
        return {"limit": self.limit, "count": self.count, "stopped": self.stopped}

    def persist(self):
        if self.ledger:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            temp = self.ledger.with_suffix(self.ledger.suffix + ".tmp")
            temp.write_text(json.dumps(self.status()) + "\n")
            os.chmod(temp, 0o600)
            temp.replace(self.ledger)

    def check(self):
        if self.stopped:
            raise GuardStopped(self.stopped)
        if self.limit is not None and self.count >= self.limit:
            self.stop("request_budget_exhausted")
            raise GuardStopped(self.stopped)

    def reserve(self):
        self.check()
        self.count += 1
        self.persist()

    def stop(self, reason):
        self.stopped = self.stopped or reason
        self.persist()


def upstream_notice(text):
    text = text.strip().lower().replace("’", "'")
    patterns = (
        r"^(?:you(?:'ve| have) |your |i(?:'ve| have) )?(?:reached|hit|exceeded).{0,80}(?:limit|quota)",
        r"^(?:you(?:'ve| have) )?run out of.{0,40}(?:prompts|requests)",
        r"^(?:too many requests|quota exceeded|resource_exhausted|rate limit exceeded)",
        r"(?:已達|已超過|達到).{0,30}(?:使用上限|用量上限|配額|要求上限)",
    )
    return "upstream_usage_limit" if any(re.search(pattern, line) for line in text.splitlines() for pattern in patterns) else None
