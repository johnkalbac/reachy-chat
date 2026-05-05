"""Realtime-API-driven conversational turns and shared audio output state.

`realtime_turn()` is called from the wake-word loop in main.py. It opens a
WebSocket session against `gpt-realtime`, streams the user's request up,
plays the assistant's audio reply back, and routes the model's function
tool calls (play_emotion, play_dance, set_volume, mute, unmute,
who_called_me, web_search, set_timer) through `TOOL_HANDLERS`.

`announce_via_realtime()` runs a one-shot realtime session with a fixed
instruction and no microphone — used by the timer service to speak when
a timer fires.

Auth: reads OPENAI_API_KEY from the environment. On the robot this must be
set in the daemon's systemd unit, not just an interactive shell.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from math import gcd
from pathlib import Path
from typing import Any, Callable

import numpy as np
from openai import OpenAI
from scipy.signal import resample_poly

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose

logger = logging.getLogger(__name__)

REALTIME_MODEL = "gpt-realtime"
REALTIME_RATE = 24_000  # OpenAI Realtime default: PCM16 mono @ 24 kHz, both directions.
REALTIME_VOICE = "ballad"
MAX_TURN_S = 30.0  # hard cap; an unresponsive session shouldn't pin the device.
SDK_INPUT_RATE = 16_000  # mic is 16 kHz.

WAVE_AMPLITUDE_DEG = 15.0  # peak antenna deflection while assistant is speaking.
WAVE_FREQ_HZ = 0.8         # full cycles per second.
WAVE_TICK_S = 0.04         # 25 Hz update rate for set_target — gentle on the motor bus.

WAGGLE_AMPLITUDE_DEG = 30.0  # wider than the speaking wave so it reads as an alert.
WAGGLE_FREQ_HZ = 3.0
WAGGLE_DURATION_S = 1.2

EMOTIONS_LIBRARY = "pollen-robotics/reachy-mini-emotions-library"
DANCES_LIBRARY = "pollen-robotics/reachy-mini-dances-library"
RECORDED_MOVE_GOTO_DURATION_S = 0.5

WEB_SEARCH_MODEL = "gpt-5-mini"
WEB_SEARCH_TIMEOUT_S = 15.0
DOA_HEAD_TURN_DURATION_S = 0.4
DOA_YAW_LIMIT_RAD = float(np.pi / 2)

ANNOUNCE_MAX_S = 15.0  # cap on a single timer-announcement realtime session.

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
    """One full conversational turn: user speaks, model replies, session closes."""
    if not _realtime_session_lock.acquire(timeout=5.0):
        logger.warning("could not acquire realtime lock within 5s; skipping turn")
        return
    try:
        _run_realtime_turn(reachy_mini, stop_event, output_rate)
    finally:
        _realtime_session_lock.release()


def _run_realtime_turn(reachy_mini: ReachyMini, stop_event: threading.Event, output_rate: int) -> None:
    client = OpenAI()
    deadline = time.monotonic() + MAX_TURN_S
    producer_done = threading.Event()
    wave_stop = threading.Event()
    wave_thread: threading.Thread | None = None
    motion_lock = threading.Lock()
    instructions = _load_instructions()
    tools = _build_tools()
    need_continuation = False

    ctx: dict[str, Any] = {
        "client": client,
        "motion_lock": motion_lock,
        "output_rate": output_rate,
    }

    try:
        with client.realtime.connect(model=REALTIME_MODEL) as conn:
            session: dict = {
                "type": "realtime",
                "model": REALTIME_MODEL,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {"turn_detection": {"type": "server_vad"}},
                    "output": {"voice": REALTIME_VOICE},
                },
                "instructions": instructions,
            }
            if tools:
                session["tools"] = tools
                session["tool_choice"] = "auto"
            conn.session.update(session=session)
            logger.info(
                "realtime session opened (model=%s, voice=%s, instructions=%d chars, tools=%d)",
                REALTIME_MODEL, REALTIME_VOICE, len(instructions), len(tools),
            )

            producer = threading.Thread(
                target=_pump_mic_to_realtime,
                args=(reachy_mini, conn, stop_event, producer_done),
                name="realtime-mic-producer",
                daemon=True,
            )
            producer.start()

            last_audio_at: float | None = None
            for event in conn:
                if stop_event.is_set():
                    logger.info("stop_event set; ending realtime turn")
                    break
                if time.monotonic() > deadline:
                    logger.warning("realtime turn exceeded %.0fs cap; ending", MAX_TURN_S)
                    break

                if event.type == "response.output_audio.delta":
                    if wave_thread is None:
                        wave_thread = threading.Thread(
                            target=_wave_antennas,
                            args=(reachy_mini, wave_stop, motion_lock),
                            name="realtime-antenna-wave",
                            daemon=True,
                        )
                        wave_thread.start()
                    pcm_bytes = base64.b64decode(event.delta)
                    _push_realtime_audio(reachy_mini, pcm_bytes, output_rate)
                    last_audio_at = time.monotonic()
                elif event.type == "response.function_call_arguments.done":
                    _handle_function_call(reachy_mini, conn, event, ctx)
                    need_continuation = True
                elif event.type == "response.done":
                    if need_continuation:
                        logger.info("response.done with pending tool result; continuing")
                        need_continuation = False
                        conn.response.create()
                    else:
                        logger.info("response.done received")
                        break
                elif event.type == "error":
                    logger.error("realtime error event: %s", getattr(event, "error", event))
                    break
                elif event.type == "input_audio_buffer.speech_started":
                    logger.info("user speech started")
                elif event.type == "input_audio_buffer.speech_stopped":
                    logger.info("user speech stopped")

            producer_done.set()
            producer.join(timeout=1.0)

            if last_audio_at is not None:
                tail_wait = max(0.0, 0.4 - (time.monotonic() - last_audio_at))
                if tail_wait:
                    time.sleep(tail_wait)
    except Exception:
        logger.exception("realtime turn failed")
    finally:
        producer_done.set()
        wave_stop.set()
        if wave_thread is not None:
            wave_thread.join(timeout=0.5)
            # Wait out any in-flight recorded move so we don't tug the antennas
            # away from its targets mid-clip; cap the wait so we never hang.
            if motion_lock.acquire(timeout=10.0):
                try:
                    reachy_mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=0.2)
                except Exception:
                    logger.exception("returning antennas to neutral failed")
                finally:
                    motion_lock.release()
            else:
                logger.warning("could not acquire motion_lock to reset antennas; skipping")


# --- Public: announce a message via a one-shot realtime session ----------

def announce_via_realtime(reachy_mini: ReachyMini, output_rate: int, message: str) -> None:
    """Open a brief realtime session with a fixed instruction and play the reply.

    No mic input — `conn.response.create()` triggers an immediate response.
    Used by the timer service to speak when a timer fires. Will wait up to
    30s to grab the realtime lock if a wake-word turn is in flight.
    """
    if not _realtime_session_lock.acquire(timeout=30.0):
        logger.warning("could not acquire realtime lock for announcement; skipping")
        return
    try:
        _run_announcement(reachy_mini, output_rate, message)
    finally:
        _realtime_session_lock.release()


def _run_announcement(reachy_mini: ReachyMini, output_rate: int, message: str) -> None:
    client = OpenAI()
    deadline = time.monotonic() + ANNOUNCE_MAX_S
    try:
        with client.realtime.connect(model=REALTIME_MODEL) as conn:
            conn.session.update(session={
                "type": "realtime",
                "model": REALTIME_MODEL,
                "output_modalities": ["audio"],
                "audio": {"output": {"voice": REALTIME_VOICE}},
                "instructions": f"Say exactly the following sentence and nothing else: {message}",
            })
            conn.response.create()
            logger.info("announcement session opened: %r", message)

            last_audio_at: float | None = None
            for event in conn:
                if time.monotonic() > deadline:
                    logger.warning("announcement exceeded %.0fs cap; ending", ANNOUNCE_MAX_S)
                    break
                if event.type == "response.output_audio.delta":
                    pcm_bytes = base64.b64decode(event.delta)
                    _push_realtime_audio(reachy_mini, pcm_bytes, output_rate)
                    last_audio_at = time.monotonic()
                elif event.type == "response.done":
                    break
                elif event.type == "error":
                    logger.error("announcement error: %s", getattr(event, "error", event))
                    break

            if last_audio_at is not None:
                tail_wait = max(0.0, 0.4 - (time.monotonic() - last_audio_at))
                if tail_wait:
                    time.sleep(tail_wait)
    except Exception:
        logger.exception("announcement failed")


# --- Internal: mic producer + antenna wave -------------------------------

def _pump_mic_to_realtime(
    reachy_mini: ReachyMini,
    conn,
    stop_event: threading.Event,
    producer_done: threading.Event,
) -> None:
    """Read mic samples, resample 16 kHz -> 24 kHz, send as base64 PCM16."""
    try:
        while not producer_done.is_set() and not stop_event.is_set():
            sample = reachy_mini.media.get_audio_sample()
            if sample is None or len(sample) == 0:
                time.sleep(0.005)
                continue

            mono16k = _to_mono_int16(sample)
            pcm24k = _resample_int16(mono16k, SDK_INPUT_RATE, REALTIME_RATE)
            payload = base64.b64encode(pcm24k.tobytes()).decode("utf-8")
            conn.input_audio_buffer.append(audio=payload)
    except Exception:
        logger.exception("mic producer thread crashed")


def _wave_antennas(
    reachy_mini: ReachyMini,
    stop_flag: threading.Event,
    motion_lock: threading.Lock,
) -> None:
    """Antisymmetric sine wave on the antennas while the assistant is speaking."""
    neutral = create_head_pose()
    amplitude = np.deg2rad(WAVE_AMPLITUDE_DEG)
    omega = 2 * np.pi * WAVE_FREQ_HZ
    t0 = time.monotonic()
    try:
        while not stop_flag.is_set():
            offset = float(amplitude * np.sin(omega * (time.monotonic() - t0)))
            with motion_lock:
                # Re-check inside the lock: while we were waiting for it (e.g.
                # a recorded move was running), the turn may have ended.
                # Without this, we'd snap the antennas to a non-zero sine
                # position right after the end-of-turn goto_target.
                if stop_flag.is_set():
                    break
                reachy_mini.set_target(head=neutral, antennas=[offset, -offset])
            time.sleep(WAVE_TICK_S)
    except Exception:
        logger.exception("antenna wave thread crashed")


def _push_realtime_audio(reachy_mini: ReachyMini, pcm_bytes: bytes, output_rate: int) -> None:
    """Decode PCM16 mono @ 24 kHz, resample, apply volume, push to speaker."""
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    floats = pcm.astype(np.float32) / 32767.0
    if REALTIME_RATE != output_rate:
        g = gcd(REALTIME_RATE, output_rate)
        floats = resample_poly(floats, up=output_rate // g, down=REALTIME_RATE // g)
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


def _handle_function_call(
    reachy_mini: ReachyMini,
    conn,
    event,
    ctx: dict[str, Any],
) -> None:
    """Dispatch one `response.function_call_arguments.done` event."""
    call_id = getattr(event, "call_id", None)
    name = getattr(event, "name", None)
    raw_args = getattr(event, "arguments", "") or ""
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        logger.exception("malformed tool arguments: %r", raw_args)
        args = {}

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        logger.warning("unknown tool %r requested by model", name)
        output: dict = {"status": "error", "reason": f"unknown_tool:{name}"}
    else:
        logger.info("tool %s(%s) requested", name, args)
        try:
            output = handler(reachy_mini, args, ctx)
        except Exception as e:
            logger.exception("tool %s handler raised", name)
            output = {"status": "error", "reason": f"{type(e).__name__}: {e}"}

    if call_id:
        try:
            conn.conversation.item.create(item={
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(output),
            })
        except Exception:
            logger.exception("sending function_call_output failed")


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
            "what you're saying. The clip plays asynchronously while you keep speaking."
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
            "dance, or when celebration is clearly warranted. Plays asynchronously."
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
    threading.Thread(
        target=_execute_recorded_move,
        args=(reachy_mini, DANCES_LIBRARY, name, ctx["motion_lock"]),
        name=f"dance-{name}",
        daemon=True,
    ).start()
    return {"status": "started", "dance": name}


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
    except Exception:
        logger.exception("clip %r not found in %s", name, library_id)
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
            "unmute separately if the robot is muted."
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
        "description": "Silence all audio output (your voice, chimes, announcements) until unmute is called.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }


def _tool_mute(reachy_mini: ReachyMini, args: dict, ctx: dict) -> dict:
    return set_output_muted(True)


def _unmute_schema() -> dict:
    return {
        "type": "function",
        "name": "unmute",
        "description": "Restore audio output after a previous mute.",
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
    client: OpenAI = ctx["client"]
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
                reasoning={"effort": "low"},
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
            "will hear a chime and a brief spoken announcement using the label. "
            "Returns the timer id and the seconds remaining."
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


def _resample_int16(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples
    g = gcd(src_rate, dst_rate)
    floats = samples.astype(np.float32) / 32767.0
    floats = resample_poly(floats, up=dst_rate // g, down=src_rate // g)
    return np.clip(floats * 32767.0, -32768, 32767).astype(np.int16)


# --- Backwards-compatible aliases (old import names used by main.py) ----

# Old name kept so existing `from reachy_chat.realtime import warm_emotions`
# imports keep working through the transition.
warm_emotions = warm_libraries
