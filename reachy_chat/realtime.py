"""One conversational turn against the OpenAI Realtime API.

Called from the wake-word loop in main.py. Opens a WebSocket session,
streams mic audio up, streams the assistant's audio reply back to the
speaker, then closes. Single-turn — no conversation memory across calls.

Auth: reads OPENAI_API_KEY from the environment (handled by the openai SDK).
On the robot this must be set in the daemon's systemd unit, not just an
interactive shell — see CLAUDE.md.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from math import gcd
from pathlib import Path

import numpy as np
from openai import OpenAI
from scipy.signal import resample_poly

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose

logger = logging.getLogger(__name__)

REALTIME_MODEL = "gpt-realtime"
REALTIME_RATE = 24_000  # OpenAI Realtime default: PCM16 mono @ 24 kHz, both directions.
REALTIME_VOICE = "coral"
MAX_TURN_S = 30.0  # hard cap; an unresponsive session shouldn't pin the device.
SDK_INPUT_RATE = 16_000  # matches main.SDK_SAMPLE_RATE; mic is 16 kHz.

WAVE_AMPLITUDE_DEG = 15.0  # peak antenna deflection while assistant is speaking.
WAVE_FREQ_HZ = 0.8         # full cycles per second.
WAVE_TICK_S = 0.02         # ~50 Hz update rate for set_target.

EMOTIONS_LIBRARY = "pollen-robotics/reachy-mini-emotions-library"
EMOTION_GOTO_DURATION_S = 0.5  # smoothing into an emotion clip's first pose.

# Cache the RecordedMoves dataset across turns — it does network IO + disk reads
# on first construction, so we only want to pay that once per process.
_emotions: RecordedMoves | None = None
_emotions_lock = threading.Lock()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_DEFAULT_INSTRUCTIONS = (
    "You are a helpful assistant living inside a small desktop robot named Reachy. "
    "Keep replies short and conversational — usually one or two sentences."
)


def realtime_turn(reachy_mini: ReachyMini, stop_event: threading.Event, output_rate: int) -> None:
    client = OpenAI()
    deadline = time.monotonic() + MAX_TURN_S
    producer_done = threading.Event()
    wave_stop = threading.Event()
    wave_thread: threading.Thread | None = None
    # Serializes the antenna-wave thread against play_move() so they don't fight
    # over motion targets. Wave acquires per-tick (microseconds); emotion holds
    # it for the duration of the clip.
    motion_lock = threading.Lock()
    instructions = _load_instructions()
    tools = _build_tools()
    need_continuation = False

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
                    _handle_function_call(reachy_mini, conn, event, motion_lock)
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

            # push_audio_sample is non-blocking; let the tail of the reply finish
            # before we hand control back to the wake-word loop.
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
            try:
                reachy_mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=0.2)
            except Exception:
                logger.exception("returning antennas to neutral failed")


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
        # The receiver loop will see the connection close or an error event.
        logger.exception("mic producer thread crashed")


def _wave_antennas(
    reachy_mini: ReachyMini,
    stop_flag: threading.Event,
    motion_lock: threading.Lock,
) -> None:
    """Antisymmetric sine wave on the antennas while the assistant is speaking.

    Holds `motion_lock` for each set_target call so an in-flight emotion
    `play_move` can take exclusive control of the body without fighting our
    targets.
    """
    neutral = create_head_pose()
    amplitude = np.deg2rad(WAVE_AMPLITUDE_DEG)
    omega = 2 * np.pi * WAVE_FREQ_HZ
    t0 = time.monotonic()
    try:
        while not stop_flag.is_set():
            offset = float(amplitude * np.sin(omega * (time.monotonic() - t0)))
            with motion_lock:
                reachy_mini.set_target(head=neutral, antennas=[offset, -offset])
            time.sleep(WAVE_TICK_S)
    except Exception:
        logger.exception("antenna wave thread crashed")


def _build_tools() -> list[dict]:
    """Build the `tools` array for session.update.

    Returns an empty list when the emotions library can't be loaded — the
    realtime session then runs without function calling, instead of failing.
    """
    moves = _get_emotions()
    if moves is None:
        return []
    try:
        names = list(moves.list_moves())
    except Exception:
        logger.exception("listing emotion clips failed")
        return []
    if not names:
        return []
    return [
        {
            "type": "function",
            "name": "play_emotion",
            "description": (
                "Play a short physical animation on the robot's body and antennas to "
                "react expressively during the conversation. Use sparingly — only when "
                "an emotion would visibly add to what you're saying. The clip plays "
                "asynchronously, so you can keep speaking while it runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": names,
                        "description": "Identifier of the emotion clip to play.",
                    }
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        }
    ]


def _handle_function_call(
    reachy_mini: ReachyMini,
    conn,
    event,
    motion_lock: threading.Lock,
) -> None:
    """Dispatch a `response.function_call_arguments.done` event.

    Parses the arguments, kicks off the requested emotion in a background
    thread (so the model can keep speaking), and sends the function output
    back so the model can continue.
    """
    call_id = getattr(event, "call_id", None)
    name_field = getattr(event, "name", None)
    raw_args = getattr(event, "arguments", "") or ""
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        logger.exception("malformed tool arguments: %r", raw_args)
        args = {}

    output: dict = {"status": "error", "reason": "unhandled_tool"}
    if name_field == "play_emotion":
        emotion = args.get("name")
        if not emotion:
            output = {"status": "error", "reason": "missing_name"}
        else:
            logger.info("play_emotion(%r) requested by model", emotion)
            threading.Thread(
                target=_execute_emotion,
                args=(reachy_mini, emotion, motion_lock),
                name=f"emotion-{emotion}",
                daemon=True,
            ).start()
            output = {"status": "started", "emotion": emotion}

    if call_id:
        try:
            conn.conversation.item.create(item={
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(output),
            })
        except Exception:
            logger.exception("sending function_call_output failed")


def _execute_emotion(reachy_mini: ReachyMini, name: str, motion_lock: threading.Lock) -> None:
    """Play one emotion clip with exclusive body control."""
    moves = _get_emotions()
    if moves is None:
        logger.warning("emotions library unavailable; skipping %r", name)
        return
    try:
        move = moves.get(name)
    except Exception:
        logger.exception("emotion %r not found", name)
        return
    try:
        with motion_lock:
            reachy_mini.play_move(move, initial_goto_duration=EMOTION_GOTO_DURATION_S)
    except Exception:
        logger.exception("playing emotion %r failed", name)


def _get_emotions() -> RecordedMoves | None:
    """Return the singleton emotions library, lazy-loading on first call.

    Returns None if loading fails (e.g. no network on first install). Subsequent
    calls keep returning None — restart the app after fixing the issue.
    """
    global _emotions
    with _emotions_lock:
        if _emotions is None:
            try:
                logger.info("loading emotions library %s", EMOTIONS_LIBRARY)
                t0 = time.monotonic()
                _emotions = RecordedMoves(EMOTIONS_LIBRARY)
                logger.info(
                    "loaded %d emotion clips in %.1fs",
                    len(list(_emotions.list_moves())), time.monotonic() - t0,
                )
            except Exception:
                logger.exception("loading emotions library failed; tool calls disabled")
                _emotions = None
        return _emotions


def warm_emotions() -> None:
    """Pre-load the emotions library. Safe to call from a background thread at
    app startup so the first wake-word doesn't pay the download cost."""
    _get_emotions()


def _push_realtime_audio(reachy_mini: ReachyMini, pcm_bytes: bytes, output_rate: int) -> None:
    """Decode PCM16 mono @ 24 kHz, resample to `output_rate`, push to speaker."""
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    floats = pcm.astype(np.float32) / 32767.0
    if REALTIME_RATE != output_rate:
        g = gcd(REALTIME_RATE, output_rate)
        floats = resample_poly(floats, up=output_rate // g, down=REALTIME_RATE // g)
        floats = floats.astype(np.float32)
    reachy_mini.media.push_audio_sample(floats.reshape(-1, 1))


def _load_instructions() -> str:
    """Concatenate all `prompts/*.md` (sorted) into the realtime `instructions`.

    Files whose names start with `_` or `.` are skipped — handy for disabling
    a fragment without deleting it. Loaded fresh each turn so edits take
    effect on the next wake-word with no restart needed.
    """
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
