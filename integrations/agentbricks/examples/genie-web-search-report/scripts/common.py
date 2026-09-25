from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / ".demo-state"
SETUP_STATE_PATH = STATE_DIR / "setup-state.json"
RESULT_PATH = STATE_DIR / "result.json"
EVIDENCE_DIR = STATE_DIR / "evidence"


class StateStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)


def redact(value: Any) -> Any:
    """Remove credential-shaped fields before persisting diagnostic evidence."""
    if isinstance(value, dict):
        return {
            str(key): redact(item)
            for key, item in value.items()
            if str(key).lower() not in {"authorization", "cookie", "headers", "token"}
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value
