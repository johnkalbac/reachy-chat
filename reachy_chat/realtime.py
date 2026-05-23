"""Orchestration layer + shared infrastructure for realtime voice turns.

This module is provider-agnostic. The per-provider wire protocol lives in:
- [reachy_chat.openai_realtime][] — OpenAI Realtime API backend
- [reachy_chat.gemini_realtime][] — Gemini Live API backend

`realtime_turn()` and `announce_via_realtime()` acquire a device-wide
session lock and dispatch to one of those backends based on the
`provider.name` value in `config.toml` (default `openai`).

Both backends share the tool registry (`TOOL_HANDLERS` + the `_tool_*`
handlers), the motion lock + antenna wave helpers, the volume scaling
state, the prompt loader, and the audio output path that live here.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from math import gcd
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.signal import resample_poly

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose

from reachy_chat.config import (
    REACHY_CHAT_PROVIDER,
    WEB_SEARCH_MODEL,
    WEB_SEARCH_TIMEOUT_S,
)

logger = logging.getLogger(__name__)

PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"

PROVIDER_AUDIO_RATE = 24_000  # Both OpenAI Realtime and Gemini Live deliver PCM16 mono @ 24 kHz.

WAVE_AMPLITUDE_DEG = 15.0  # peak antenna deflection while assistant is speaking.
WAVE_FREQ_HZ = 0.8         # full cycles per second.
WAVE_TICK_S = 0.04         # 25 Hz update rate for set_target — gentle on the motor bus.
LISTENING_WAVE_AMPLITUDE_DEG = 8.0  # smaller deflection while waiting for a follow-up.
LISTENING_WAVE_FREQ_HZ = 0.3        # slower than the speaking wave so they read as distinct.

WAGGLE_AMPLITUDE_DEG = 30.0  # wider than the speaking wave so it reads as an alert.
WAGGLE_FREQ_HZ = 3.0
WAGGLE_DURATION_S = 1.2

EMOTIONS_LIBRARY = "pollen-robotics/reachy-mini-emotions-library"
DANCES_LIBRARY = "pollen-robotics/reachy-mini-dances-library"
RECORDED_MOVE_GOTO_DURATION_S = 0.5

DOA_HEAD_TURN_DURATION_S = 0.4
DOA_YAW_LIMIT_RAD = float(np.pi / 2)

# --- Module state ---------------------------------------------------------

# Cache `RecordedMoves` instances across turns — construction does network IO
# + disk reads on first use, so we only want to pay it once per dataset id.
_recorded_libraries: dict[str, RecordedMoves] = {}
_recorded_lock = threading.Lock()

# Output-volume state. Realtime audio, the ready chime, espeak fallback, and
# the timer announcement all flow through `apply_output_volume`.
_volume_pct: int = 100
_muted: bool = False
_volume_lock = threading.Lock()

# Serializes realtime sessions device-wide so wake-word turns and timer
# announcements never run two simultaneous sessions over the same speaker.
_realtime_session_lock = threading.Lock()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_DEFAULT_INSTRUCTIONS = (
    "You are a helpful assistant living inside a small desktop robot named Reachy. "
    "Keep replies short and conversational — usually one or two sentences."
)


# --- Public: volume -------------------------------------------------------

def apply_output_volume(samples: np.ndarray) -> np.ndarray:
    """Scale float32 samples by the current volume / mute state.

    Cheap (in-place clip after multiply); safe to call from any thread.
    """
    with _volume_lock:
        if _muted:
            return np.zeros_like(samples)
        scale = _volume_pct / 100.0
    if scale >= 1.0:
        return samples
    return samples * scale


def set_output_volume(level: int) -> dict:
    global _volume_pct
    with _volume_lock:
        _volume_pct = max(0, min(100, int(level)))
        state = {"volume": _volume_pct, "muted": _muted}
    logger.info("volume set: %s", state)
    return state


def set_output_muted(muted: bool) -> dict:
    global _muted
    with _volume_lock:
        _muted = bool(muted)
        state = {"volume": _volume_pct, "muted": _muted}
    logger.info("mute set: %s", state)
    return state


def get_output_volume_state() -> dict:
    with _volume_lock:
        return {"volume": _volume_pct, "muted": _muted}


# --- Public: ready chime (shared with main.py and the timer service) -----

def play_ready_chime(reachy_mini: ReachyMini, output_rate: int) -> None:
    """Two-note ascending chime (A5 -> E6). Honors the volume/mute state."""
    note_s = 0.12
    gap_s = 0.03
    n = int(output_rate * note_s)
    t = np.arange(n, dtype=np.float32) / output_rate
    envelope = np.minimum(t / 0.01, np.exp(-(t - 0.01) / 0.05)).astype(np.float32)
    tone1 = (np.sin(2 * np.pi * 880.0 * t) * envelope * 0.3).astype(np.float32)
    tone2 = (np.sin(2 * np.pi * 1318.5 * t) * envelope * 0.3).astype(np.float32)
    gap = np.zeros(int(output_rate * gap_s), dtype=np.float32)
    chime = np.concatenate([tone1, gap, tone2])
    reachy_mini.media.push_audio_sample(apply_output_volume(chime).reshape(-1, 1))
    time.sleep(len(chime) / output_rate + 0.05)


def waggle_antennas(reachy_mini: ReachyMini, duration: float = WAGGLE_DURATION_S) -> None:
    """Short, snappy antisymmetric antenna waggle to mark a notable event.

    Distinct from the speaking-time `_wave_antennas`: wider amplitude, faster,
    runs synchronously, and returns the antennas to neutral when done.
    """
    amplitude = np.deg2rad(WAGGLE_AMPLITUDE_DEG)
    omega = 2 * np.pi * WAGGLE_FREQ_HZ
    neutral = create_head_pose()
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < duration:
            offset = float(amplitude * np.sin(omega * (time.monotonic() - t0)))
            reachy_mini.set_target(head=neutral, antennas=[offset, -offset])
            time.sleep(WAVE_TICK_S)
        reachy_mini.set_target(head=neutral, antennas=[0.0, 0.0])
    except Exception:
        logger.exception("antenna waggle failed")


# --- Public: warm caches at app startup ----------------------------------

def warm_libraries() -> None:
    """Pre-load the emotions + dances libraries from a background thread."""
    _get_recorded_moves(EMOTIONS_LIBRARY)
    _get_recorded_moves(DANCES_LIBRARY)


def _get_recorded_moves(library_id: str) -> RecordedMoves | None:
    """Return the cached `RecordedMoves` for `library_id`, lazy-loading once.

    Returns None if loading fails (e.g. no network on first install). The
    caller is responsible for degrading gracefully.
    """
    with _recorded_lock:
        if library_id in _recorded_libraries:
            return _recorded_libraries[library_id]
        try:
            logger.info("loading recorded-moves library %s", library_id)
            t0 = time.monotonic()
            moves = RecordedMoves(library_id)
            count = len(list(moves.list_moves()))
            logger.info("loaded %d clips from %s in %.1fs", count, library_id, time.monotonic() - t0)
            _recorded_libraries[library_id] = moves
            return moves
        except Exception:
            logger.exception("loading %s failed; tools using it will be disabled", library_id)
            return None


# --- Public: realtime turn ------------------------------------------------

def realtime_turn(reachy_mini: ReachyMini, stop_event: threading.Event, output_rate: int) -> None:
    """One full conversational turn: user speaks, model replies, session closes.

    Dispatches to the OpenAI Realtime or Gemini Live backend based on the
    `provider.name` value in `config.toml` (default `openai`). Builds
    `instructions` and `tools` once here so both backends see the same inputs.
    """
    if not _realtime_session_lock.acquire(timeout=5.0):
        logger.warning("could not acquire realtime lock within 5s; skipping turn")
        return
    try:
        instructions = _load_instructions()
        tools = _build_tools()
        if REACHY_CHAT_PROVIDER == PROVIDER_GEMINI:
            from reachy_chat import gemini_realtime
            gemini_realtime.run_gemini_turn(reachy_mini, stop_event, output_rate, instructions, tools)
        else:
            from reachy_chat import openai_realtime
            openai_realtime.run_openai_turn(reachy_mini, stop_event, output_rate, instructions, tools)
    finally:
        _realtime_session_lock.release()


# --- Public: announce a message via a one-shot realtime session ----------

def announce_via_realtime(reachy_mini: ReachyMini, output_rate: int, message: str) -> None:
    """Open a brief realtime session with a fixed instruction and play the reply.

    No mic input — the session is triggered with a fixed text input.
    Used by the timer service to speak when a timer fires. Will wait up to
    30s to grab the realtime lock if a wake-word turn is in flight.
    Dispatches to OpenAI or Gemini based on `provider.name` in `config.toml`.
    """
    if not _realtime_session_lock.acquire(timeout=30.0):
        logger.warning("could not acquire realtime lock for announcement; skipping")
        return
    try:
        if REACHY_CHAT_PROVIDER == PROVIDER_GEMINI:
            from reachy_chat import gemini_realtime
            gemini_realtime.run_gemini_announcement(reachy_mini, output_rate, message)
        else:
            from reachy_chat import openai_realtime
            openai_realtime.run_openai_announcement(reachy_mini, output_rate, message)
    finally:
        _realtime_session_lock.release()


# --- Internal: antenna wave (shared by both provider backends) -----------

def _wave_antennas(
    reachy_mini: ReachyMini,
    stop_flag: threading.Event,
    motion_lock: threading.Lock,
    amplitude_deg: float = WAVE_AMPLITUDE_DEG,
    freq_hz: float = WAVE_FREQ_HZ,
    antisymmetric: bool = True,
) -> None:
    """Sine wave on the antennas.

    Default params are the "speaking" wave: antisymmetric (one antenna up
    while the other is down) at 15° / 0.8 Hz. Pass `antisymmetric=False`
    to make both antennas move together for a simpler back-and-forth wave.
    """
    neutral = create_head_pose()
    amplitude = np.deg2rad(amplitude_deg)
    omega = 2 * np.pi * freq_hz
    t0 = time.monotonic()
    try:
        while not stop_flag.is_set():
            offset = float(amplitude * np.sin(omega * (time.monotonic() - t0)))
            pair = [offset, -offset] if antisymmetric else [offset, offset]
            with motion_lock:
                # Re-check inside the lock: while we were waiting for it (e.g.
                # a recorded move was running), the turn may have ended.
                # Without this, we'd snap the antennas to a non-zero sine
                # position right after the end-of-turn goto_target.
                if stop_flag.is_set():
                    break
                reachy_mini.set_target(head=neutral, antennas=pair)
            time.sleep(WAVE_TICK_S)
    except Exception:
        logger.exception("antenna wave thread crashed")


def _wave_antennas_listening(
    reachy_mini: ReachyMini,
    stop_flag: threading.Event,
    motion_lock: threading.Lock,
) -> None:
    """Slow, gentle back-and-forth wave while waiting for the follow-up.

    Symmetric (both antennas move together in phase) so it reads as a calm
    listening cue rather than the busier antisymmetric speaking wave.
    """
    _wave_antennas(
        reachy_mini,
        stop_flag,
        motion_lock,
        amplitude_deg=LISTENING_WAVE_AMPLITUDE_DEG,
        freq_hz=LISTENING_WAVE_FREQ_HZ,
        antisymmetric=False,
    )


def _push_realtime_audio(reachy_mini: ReachyMini, pcm_bytes: bytes, output_rate: int) -> None:
    """Decode PCM16 mono @ provider rate, resample, apply volume, push to speaker.

    Both OpenAI Realtime and Gemini Live deliver audio at `PROVIDER_AUDIO_RATE`,
    so this is shared between the two backends.
    """
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    floats = pcm.astype(np.float32) / 32767.0
    if PROVIDER_AUDIO_RATE != output_rate:
        g = gcd(PROVIDER_AUDIO_RATE, output_rate)
        floats = resample_poly(floats, up=output_rate // g, down=PROVIDER_AUDIO_RATE // g)
        floats = floats.astype(np.float32)
    reachy_mini.media.push_audio_sample(apply_output_volume(floats).reshape(-1, 1))


# --- Tool dispatcher ------------------------------------------------------

def _build_tools() -> list[dict]:
    """Assemble per-tool schemas for `session.update`.

    Tools whose backing resource is unavailable (no emotions library, etc.)
    are omitted so a partial environment still works.
    """
    schemas: list[dict] = []
    for builder in (
        _emotion_schema,
        _dance_schema,
        _set_volume_schema,
        _mute_schema,
        _unmute_schema,
        _who_called_me_schema,
        _web_search_schema,
        _set_timer_schema,
    ):
        try:
            schema = builder()
        except Exception:
            logger.exception("tool schema builder %s failed", builder.__name__)
            continue
        if schema is not None:
            schemas.append(schema)
    return schemas


def _dispatch_tool_call(
    reachy_mini: ReachyMini,
    name: str | None,
    raw_args: str | dict | None,
    ctx: dict[str, Any],
) -> dict:
    """Run the named tool from `TOOL_HANDLERS` and return its result dict.

    `raw_args` may be a JSON string (OpenAI) or an already-decoded dict (Gemini).
    Provider-specific code is responsible for shipping the returned dict back
    to the model.
    """
    if isinstance(raw_args, dict):
        args = raw_args
    elif isinstance(raw_args, str) and raw_args:
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            logger.exception("malformed tool arguments: %r", raw_args)
            args = {}
    else:
        args = {}

    handler = TOOL_HANDLERS.get(name) if name else None
    if handler is None:
        logger.warning("unknown tool %r requested by model", name)
        return {"status": "error", "reason": f"unknown_tool:{name}"}
    logger.info("tool %s(%s) requested", name, args)
    try:
        return handler(reachy_mini, args, ctx)
    except Exception as e:
        logger.exception("tool %s handler raised", name)
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}


# --- Tool: play_emotion ---------------------------------------------------

def _emotion_schema() -> dict | None:
    moves = _get_recorded_moves(EMOTIONS_LIBRARY)
    if moves is None:
        return None
    names = list(moves.list_moves())
    if not names:
        return None
    return {
        "type": "function",
        "name": "play_emotion",
        "description": (
            "Play a short physical animation on the robot's body and antennas to react "
            "expressively. Use sparingly — only when an emotion would visibly add to "
            "what you're saying. The clip plays asynchronously. Do not narrate or "
            "announce this action; just call the tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": names, "description": "Emotion clip identifier."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    }


def _tool_play_emotion(reachy_mini: ReachyMini, args: dict, ctx: dict) -> dict:
    name = args.get("name")
    if not name:
        return {"status": "error", "reason": "missing_name"}
    err = _validate_recorded_move(EMOTIONS_LIBRARY, name)
    if err is not None:
        return err
    threading.Thread(
        target=_execute_recorded_move,
        args=(reachy_mini, EMOTIONS_LIBRARY, name, ctx["motion_lock"]),
        name=f"emotion-{name}",
        daemon=True,
    ).start()
    return {"status": "started", "emotion": name}


# --- Tool: play_dance -----------------------------------------------------

def _dance_schema() -> dict | None:
    moves = _get_recorded_moves(DANCES_LIBRARY)
    if moves is None:
        return None
    names = list(moves.list_moves())
    if not names:
        return None
    return {
        "type": "function",
        "name": "play_dance",
        "description": (
            "Play a longer choreographed dance routine on the robot's body. Heavier "
            "than play_emotion — only call when the user explicitly asks the robot to "
            "dance, or when celebration is clearly warranted. Plays asynchronously. "
            "Do not narrate or announce this action; just call the tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": names, "description": "Dance clip identifier."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    }


def _tool_play_dance(reachy_mini: ReachyMini, args: dict, ctx: dict) -> dict:
    name = args.get("name")
    if not name:
        return {"status": "error", "reason": "missing_name"}
    err = _validate_recorded_move(DANCES_LIBRARY, name)
    if err is not None:
        return err
    threading.Thread(
        target=_execute_recorded_move,
        args=(reachy_mini, DANCES_LIBRARY, name, ctx["motion_lock"]),
        name=f"dance-{name}",
        daemon=True,
    ).start()
    return {"status": "started", "dance": name}


def _validate_recorded_move(library_id: str, name: str) -> dict | None:
    """Return an error dict if `name` is unknown in `library_id`, else None.

    The realtime model occasionally hallucinates clip names despite the
    schema enum, so validating here lets us tell the model it picked a
    bad name instead of failing silently in the worker thread.
    """
    moves = _get_recorded_moves(library_id)
    if moves is None:
        return {"status": "error", "reason": "library_unavailable"}
    if name not in set(moves.list_moves()):
        logger.warning("clip %r not in library %s; rejecting tool call", name, library_id)
        return {"status": "error", "reason": f"unknown_clip:{name}"}
    return None


def _execute_recorded_move(
    reachy_mini: ReachyMini,
    library_id: str,
    name: str,
    motion_lock: threading.Lock,
) -> None:
    moves = _get_recorded_moves(library_id)
    if moves is None:
        logger.warning("library %s unavailable; skipping %r", library_id, name)
        return
    try:
        move = moves.get(name)
    except Exception as e:
        # Should be unreachable: tool handlers validate `name` via
        # `_validate_recorded_move` before kicking off this thread.
        logger.warning("clip %r unexpectedly missing from %s: %s", name, library_id, e)
        return
    try:
        with motion_lock:
            reachy_mini.play_move(move, initial_goto_duration=RECORDED_MOVE_GOTO_DURATION_S)
    except Exception:
        logger.exception("playing clip %r failed", name)


# --- Tool: set_volume / mute / unmute -------------------------------------

def _set_volume_schema() -> dict:
    return {
        "type": "function",
        "name": "set_volume",
        "description": (
            "Set the robot's audio output level as a percentage. Affects your own "
            "voice, the ready chime, and timer announcements. Does not unmute — call "
            "unmute separately if the robot is muted. Do not narrate the change; the "
            "user will hear the new level on your next words."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Volume 0..100."},
            },
            "required": ["level"],
            "additionalProperties": False,
        },
    }


def _tool_set_volume(reachy_mini: ReachyMini, args: dict, ctx: dict) -> dict:
    level = args.get("level")
    if not isinstance(level, (int, float)):
        return {"status": "error", "reason": "missing_level"}
    return set_output_volume(int(level))


def _mute_schema() -> dict:
    return {
        "type": "function",
        "name": "mute",
        "description": (
            "Silence all audio output (your voice, chimes, announcements) until "
            "unmute is called. Do not narrate the action; the silence speaks for itself."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }


def _tool_mute(reachy_mini: ReachyMini, args: dict, ctx: dict) -> dict:
    return set_output_muted(True)


def _unmute_schema() -> dict:
    return {
        "type": "function",
        "name": "unmute",
        "description": "Restore audio output after a previous mute. Do not narrate the action.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }


def _tool_unmute(reachy_mini: ReachyMini, args: dict, ctx: dict) -> dict:
    return set_output_muted(False)


# --- Tool: who_called_me --------------------------------------------------

def _who_called_me_schema() -> dict:
    return {
        "type": "function",
        "name": "who_called_me",
        "description": (
            "Determine the direction of the most recent sound source (likely the "
            "speaker), turn the robot's head to face it, and return the angle in "
            "degrees. Useful when the user asks you to look at them or face the speaker."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }


def _tool_who_called_me(reachy_mini: ReachyMini, args: dict, ctx: dict) -> dict:
    try:
        result = reachy_mini.media.get_DoA()
    except Exception:
        logger.exception("get_DoA() raised")
        return {"status": "error", "reason": "doa_unavailable"}
    if result is None:
        return {"status": "no_doa"}

    angle_rad, speech_detected = result
    clamped = float(np.clip(angle_rad, -DOA_YAW_LIMIT_RAD, DOA_YAW_LIMIT_RAD))
    try:
        with ctx["motion_lock"]:
            reachy_mini.goto_target(
                create_head_pose(yaw=clamped),
                antennas=[0.0, 0.0],
                duration=DOA_HEAD_TURN_DURATION_S,
            )
    except Exception:
        logger.exception("turning head toward DOA failed")
    return {
        "angle_deg": round(float(np.rad2deg(angle_rad)), 1),
        "speech_detected": bool(speech_detected),
    }


# --- Tool: web_search (sync, bridges to Responses API) -------------------

def _web_search_schema() -> dict:
    return {
        "type": "function",
        "name": "web_search",
        "description": (
            "Search the web for up-to-date information you don't already know. The "
            "result is a short text answer with citations baked in by the search "
            "model. Use for current events, prices, sports scores, recent news, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }


def _tool_web_search(reachy_mini: ReachyMini, args: dict, ctx: dict) -> dict:
    query = (args.get("query") or "").strip()
    if not query:
        return {"status": "error", "reason": "missing_query"}
    client = ctx["client"]  # an `openai.OpenAI` instance (this tool only runs in the OpenAI backend).
    bounded = client.with_options(timeout=WEB_SEARCH_TIMEOUT_S)
    prompt = f"Search the web and answer concisely: {query}"
    # OpenAI has shipped both "web_search" and "web_search_preview" tool types.
    # Try the GA name first, fall back to the preview alias on a 400.
    last_err: Exception | None = None
    for tool_type in ("web_search", "web_search_preview"):
        try:
            resp = bounded.responses.create(
                model=WEB_SEARCH_MODEL,
                tools=[{"type": tool_type}],
                input=prompt,
            )
            answer = (getattr(resp, "output_text", "") or "").strip()
            return {"answer": answer or "(no answer)"}
        except Exception as e:  # BadRequestError, RateLimitError, APITimeoutError, ...
            last_err = e
            msg = str(e).lower()
            if "tool" in msg and ("type" in msg or "unknown" in msg):
                continue  # tool-name mismatch; try the alias
            break
    logger.exception("web_search failed", exc_info=last_err)
    return {"status": "error", "reason": f"{type(last_err).__name__}: {last_err}"}


# --- Tool: set_timer ------------------------------------------------------

def _set_timer_schema() -> dict:
    return {
        "type": "function",
        "name": "set_timer",
        "description": (
            "Set a countdown timer that announces itself when it fires. The user "
            "will hear a chime and a brief spoken announcement using the label. Do "
            "not narrate the setup; just call the tool. Returns the timer id and "
            "the seconds remaining."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "minimum": 1, "description": "Countdown duration in seconds."},
                "label": {"type": "string", "description": "Short label (e.g. 'pasta'). Optional."},
            },
            "required": ["seconds"],
            "additionalProperties": False,
        },
    }


def _tool_set_timer(reachy_mini: ReachyMini, args: dict, ctx: dict) -> dict:
    # Imported lazily to keep timers.py optional / break import cycles.
    from reachy_chat import timers
    service = timers.get_service()
    if service is None:
        return {"status": "error", "reason": "timer_service_not_running"}
    seconds = args.get("seconds")
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return {"status": "error", "reason": "invalid_seconds"}
    label = (args.get("label") or "").strip()
    timer_id = service.add_timer(float(seconds), label)
    return {"status": "started", "id": timer_id, "fires_in_seconds": float(seconds), "label": label}


# Wire the dispatcher last so all _tool_* functions exist.
TOOL_HANDLERS: dict[str, Callable[[ReachyMini, dict, dict], dict]] = {
    "play_emotion":  _tool_play_emotion,
    "play_dance":    _tool_play_dance,
    "set_volume":    _tool_set_volume,
    "mute":          _tool_mute,
    "unmute":        _tool_unmute,
    "who_called_me": _tool_who_called_me,
    "web_search":    _tool_web_search,
    "set_timer":     _tool_set_timer,
}


# --- Internal: prompt loading + audio utils ------------------------------

def _load_instructions() -> str:
    """Concatenate all `prompts/*.md` (sorted) into the realtime `instructions`."""
    if not PROMPTS_DIR.is_dir():
        logger.warning("prompts dir %s missing; using built-in default", PROMPTS_DIR)
        return _DEFAULT_INSTRUCTIONS

    fragments: list[str] = []
    used: list[str] = []
    for path in sorted(PROMPTS_DIR.glob("*.md")):
        if path.name.startswith(("_", ".")):
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            fragments.append(text)
            used.append(path.name)

    if not fragments:
        logger.warning("no usable prompts in %s; using built-in default", PROMPTS_DIR)
        return _DEFAULT_INSTRUCTIONS

    logger.info("loaded prompt fragments: %s", ", ".join(used))
    return "\n\n".join(fragments)


def _to_mono_int16(sample: np.ndarray) -> np.ndarray:
    if sample.ndim == 2:
        sample = sample.mean(axis=1)
    return (np.clip(sample, -1.0, 1.0) * 32767.0).astype(np.int16)


# --- Backwards-compatible aliases (old import names used by main.py) ----

# Old name kept so existing `from reachy_chat.realtime import warm_emotions`
# imports keep working through the transition.
warm_emotions = warm_libraries
