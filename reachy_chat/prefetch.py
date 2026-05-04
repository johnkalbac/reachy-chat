"""Pre-fetch external assets so the first wake-word doesn't pay the download cost.

Run once after `pip install -e .`:

    /venvs/apps_venv/bin/reachy-chat-prefetch

Currently fetches the OpenAI Realtime emotions library (a HuggingFace dataset).
Safe to re-run — `RecordedMoves(...)` short-circuits on cache hit.
"""

from __future__ import annotations

import logging
import sys

from reachy_mini.motion.recorded_move import RecordedMoves

from reachy_chat.realtime import EMOTIONS_LIBRARY


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(f"Pre-fetching {EMOTIONS_LIBRARY} ...", flush=True)
    try:
        moves = RecordedMoves(EMOTIONS_LIBRARY)
        names = list(moves.list_moves())
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    print(f"OK. {len(names)} emotion clips cached:")
    print("  " + ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
