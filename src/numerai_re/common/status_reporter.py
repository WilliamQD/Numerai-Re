from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeStatusReporter:
    logger: Any
    interval_seconds: float = 60.0
    name: str = "train"

    def __post_init__(self) -> None:
        self._last_emit = 0.0
        self._last_rendered_len = 0
        self._active = False
        self._interactive = bool(getattr(sys.stderr, "isatty", lambda: False)())

    def _format(self, phase: str, fields: dict[str, object]) -> str:
        keyvals = " ".join(f"{key}={value}" for key, value in fields.items())
        if keyvals:
            return f"[{self.name}] phase={phase} {keyvals}"
        return f"[{self.name}] phase={phase}"

    def update(self, phase: str, *, force: bool = False, **fields: object) -> None:
        now = time.monotonic()
        if not force and self._last_emit > 0 and (now - self._last_emit) < self.interval_seconds:
            return

        message = self._format(phase, fields)
        self._last_emit = now
        self._active = True

        if self._interactive:
            padded = message
            if self._last_rendered_len > len(message):
                padded += " " * (self._last_rendered_len - len(message))
            print(f"\r{padded}", end="", file=sys.stderr, flush=True)
            self._last_rendered_len = len(message)
            return

        self.logger.info("phase=status_update message=%s", message)

    def clear(self) -> None:
        if self._interactive and self._active:
            print(file=sys.stderr, flush=True)
        self._active = False
        self._last_rendered_len = 0

    def __enter__(self) -> "RuntimeStatusReporter":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.clear()
