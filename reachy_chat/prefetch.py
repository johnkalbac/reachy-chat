"""Pre-fetch external assets so the first wake-word doesn't pay the download cost.

Run once after `pip install -e .`:

    /venvs/apps_venv/bin/reachy-chat-prefetch

Currently fetches both recorded-move libraries (emotions + dances) used as
realtime tools. Safe to re-run — `RecordedMoves(...)` short-circuits on
cache hit.
"""

from __future__ import annotations

import logging
import sys

from reachy_mini.motion.recorded_move import RecordedMoves

from reachy_chat.realtime import DANCES_LIBRARY, EMOTIONS_LIBRARY


def _fetch(library_id: str) -> int:
    print(f"Pre-fetching {library_id} ...", flush=True)
    moves = RecordedMoves(library_id)
    names = list(moves.list_moves())
    print(f"  OK. {len(names)} clips:")
    print("    " + ", ".join(names))
    return len(names)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rc = 0
    for library_id in (EMOTIONS_LIBRARY, DANCES_LIBRARY):
        try:
            _fetch(library_id)
        except Exception as e:
            print(f"  FAILED ({library_id}): {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
