"""Project-wide tunables loaded from `config.toml` at the repo root.

Single source of truth for the non-secret settings — wake word, provider,
model and voice names, and session timing. Secrets (OPENAI_API_KEY,
GEMINI_API_KEY) stay in the environment and are read directly by the
provider modules.

Resolution mirrors how `prompts/` is found in realtime.py: the file is
expected one directory up from this module, which assumes the editable
install layout (`pip install -e .`). If the file is missing or a value
fails to parse, the built-in default below is used so the app still runs.

Changes take effect on the next process start — the file is read once at
import time, not on every turn.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"

# Defaults are duplicated here (not just in config.toml) so the app still
# works when the file is missing — e.g. a non-editable wheel install or a
# half-finished checkout. Keep these in sync with config.toml.
_DEFAULTS: dict = {
    "wake_word": {"name": "hey_jarvis", "threshold": 0.5},
    "provider": {"name": "openai"},
    "openai": {
        "model": "gpt-realtime",
        "voice": "ballad",
        "web_search_model": "gpt-4o-mini",
        "web_search_timeout_s": 15.0,
    },
    "gemini": {"model": "gemini-3.1-flash-live-preview", "voice": "Iapetus"},
    "timing": {
        "announce_max_s": 15.0,
        "followup_window_s": 8.0,
        "max_session_s": 180.0,
        "max_turn_s": 60.0,
        "reset_to_neutral_duration_s": 0.6,
    },
}


def _load() -> dict:
    if not CONFIG_PATH.is_file():
        logger.warning("config file %s missing; using built-in defaults", CONFIG_PATH)
        return _DEFAULTS
    try:
        with CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        logger.exception("failed to parse %s; using built-in defaults", CONFIG_PATH)
        return _DEFAULTS
    merged: dict = {}
    for section, defaults in _DEFAULTS.items():
        section_data = dict(defaults)
        section_data.update(data.get(section, {}))
        merged[section] = section_data
    logger.info("loaded config from %s", CONFIG_PATH)
    return merged


_cfg = _load()

WAKE_WORD: str = str(_cfg["wake_word"]["name"])
WAKE_WORD_THRESHOLD: float = float(_cfg["wake_word"]["threshold"])

REACHY_CHAT_PROVIDER: str = str(_cfg["provider"]["name"]).strip().lower()

REALTIME_MODEL: str = str(_cfg["openai"]["model"])
REALTIME_VOICE: str = str(_cfg["openai"]["voice"])
WEB_SEARCH_MODEL: str = str(_cfg["openai"]["web_search_model"])
WEB_SEARCH_TIMEOUT_S: float = float(_cfg["openai"]["web_search_timeout_s"])

GEMINI_MODEL: str = str(_cfg["gemini"]["model"])
GEMINI_VOICE: str = str(_cfg["gemini"]["voice"])

ANNOUNCE_MAX_S: float = float(_cfg["timing"]["announce_max_s"])
FOLLOWUP_WINDOW_S: float = float(_cfg["timing"]["followup_window_s"])
MAX_SESSION_S: float = float(_cfg["timing"]["max_session_s"])
MAX_TURN_S: float = float(_cfg["timing"]["max_turn_s"])
RESET_TO_NEUTRAL_DURATION_S: float = float(_cfg["timing"]["reset_to_neutral_duration_s"])
