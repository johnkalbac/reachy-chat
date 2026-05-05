"""Background timer service that announces itself when a timer fires.

The realtime model exposes `set_timer(seconds, label)`. The handler in
`realtime.py` enqueues the timer here. A single worker thread waits on a
`threading.Condition` until the next deadline, then plays the ready chime
and opens a one-shot realtime session via `announce_via_realtime` to
speak the label.

The service is created in `ReachyChat.run()` and stopped in its `finally`
clause; the singleton accessor is `get_service()`.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field

from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _Entry:
    deadline: float
    timer_id: int = field(compare=False)
    label: str = field(default="", compare=False)


class TimerService:
    def __init__(self, reachy_mini: ReachyMini, output_rate: int) -> None:
        self.reachy_mini = reachy_mini
        self.output_rate = output_rate
        self._cv = threading.Condition()
        self._heap: list[_Entry] = []
        self._next_id = 1
        self._stopped = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="timer-service", daemon=True)
        self._thread.start()
        logger.info("timer service started")

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("timer service stopped")

    def add_timer(self, seconds: float, label: str = "") -> int:
        deadline = time.monotonic() + max(0.0, float(seconds))
        with self._cv:
            timer_id = self._next_id
            self._next_id += 1
            heapq.heappush(self._heap, _Entry(deadline, timer_id, label))
            self._cv.notify_all()
        logger.info("timer #%d (%r) registered for +%.1fs", timer_id, label, seconds)
        return timer_id

    def list_timers(self) -> list[dict]:
        now = time.monotonic()
        with self._cv:
            entries = sorted(self._heap)
        return [
            {"id": e.timer_id, "label": e.label, "fires_in_seconds": max(0.0, e.deadline - now)}
            for e in entries
        ]

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stopped and (
                    not self._heap or self._heap[0].deadline > time.monotonic()
                ):
                    if self._heap:
                        wait = self._heap[0].deadline - time.monotonic()
                        if wait > 0:
                            self._cv.wait(timeout=wait)
                        else:
                            break
                    else:
                        self._cv.wait()
                if self._stopped:
                    return
                entry = heapq.heappop(self._heap)
            try:
                self._fire(entry)
            except Exception:
                logger.exception("firing timer #%d failed", entry.timer_id)

    def _fire(self, entry: _Entry) -> None:
        # Imported lazily to avoid a circular import (realtime.py imports timers
        # for the set_timer tool handler).
        from reachy_chat.realtime import announce_via_realtime, play_ready_chime, waggle_antennas

        logger.info("firing timer #%d (%r)", entry.timer_id, entry.label)
        try:
            play_ready_chime(self.reachy_mini, self.output_rate)
        except Exception:
            logger.exception("ready chime failed during timer fire")
        try:
            waggle_antennas(self.reachy_mini)
        except Exception:
            logger.exception("antenna waggle failed during timer fire")
        message = f"Timer {entry.label} is done." if entry.label else "Timer's up."
        announce_via_realtime(self.reachy_mini, self.output_rate, message)


# --- Module-level singleton accessor -------------------------------------

_active_service: TimerService | None = None
_active_lock = threading.Lock()


def set_service(service: TimerService | None) -> None:
    global _active_service
    with _active_lock:
        _active_service = service


def get_service() -> TimerService | None:
    with _active_lock:
        return _active_service
