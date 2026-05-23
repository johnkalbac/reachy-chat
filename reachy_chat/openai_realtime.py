"""OpenAI Realtime API backend.

Implements the OpenAI-specific half of the provider dispatch driven by
[reachy_chat.realtime][]. Owns the WebSocket session against
`gpt-realtime`, the 16 kHz -> 24 kHz mic resample, base64 PCM framing,
and translation of OpenAI Realtime events into the shared lifecycle
(SPEAKING / LISTENING transitions, antenna waves, tool dispatch).

Public entry points:
- `run_openai_turn` — one multi-turn wake-word session.
- `run_openai_announcement` — one-shot timer announcement.

Both are called from `reachy_chat.realtime` after that module has built
the shared `instructions` string and `tools` list. All robot-control
helpers (antenna waves, audio output, tool registry) are imported back
from `reachy_chat.realtime` so this file stays focused on wire protocol.

Auth: reads `OPENAI_API_KEY` from the environment. On the robot this
must be set in the daemon's systemd unit, not just an interactive shell.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from math import gcd
from typing import Any

import numpy as np
from openai import OpenAI
from scipy.signal import resample_poly

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from reachy_chat.config import (
    ANNOUNCE_MAX_S,
    FOLLOWUP_WINDOW_S,
    MAX_SESSION_S,
    MAX_TURN_S,
    REALTIME_MODEL,
    REALTIME_VOICE,
    RESET_TO_NEUTRAL_DURATION_S,
)
from reachy_chat.realtime import (
    _dispatch_tool_call,
    _push_realtime_audio,
    _to_mono_int16,
    _wave_antennas,
    _wave_antennas_listening,
)

logger = logging.getLogger(__name__)

OPENAI_REALTIME_RATE = 24_000  # OpenAI Realtime requires PCM16 mono @ 24 kHz both directions.
SDK_INPUT_RATE = 16_000        # the Reachy media SDK delivers mic samples at 16 kHz.


# --- Public entry points (called from realtime.py) -----------------------

def run_openai_turn(
    reachy_mini: ReachyMini,
    stop_event: threading.Event,
    output_rate: int,
    instructions: str,
    tools: list[dict],
) -> None:
    """Open one multi-turn realtime session.

    A session contains one or more model responses. After each response the
    robot enters a `FOLLOWUP_WINDOW_S` listening window with a slow antenna
    waggle; if the user starts speaking, that becomes the next turn. The
    session ends when the window elapses with no speech, the server errors,
    `stop_event` is set, or `MAX_SESSION_S` is reached.
    """
    client = OpenAI()
    session_deadline = time.monotonic() + MAX_SESSION_S
    turn_deadline = time.monotonic() + MAX_TURN_S
    followup_deadline: float | None = None  # set while in the listening window.

    producer_done = threading.Event()
    motion_lock = threading.Lock()

    wave_stop = threading.Event()
    wave_thread: threading.Thread | None = None
    listening_wave_stop = threading.Event()
    listening_wave_thread: threading.Thread | None = None

    need_continuation = False

    ctx: dict[str, Any] = {
        "client": client,
        "motion_lock": motion_lock,
        "output_rate": output_rate,
    }

    def _reset_antennas_to_neutral() -> None:
        # Short timeout: a recorded move can hold the lock for many seconds,
        # in which case we'd rather skip the cosmetic reset than block.
        if not motion_lock.acquire(timeout=2.0):
            logger.warning("could not acquire motion_lock for neutral reset; skipping")
            return
        try:
            reachy_mini.goto_target(
                create_head_pose(),
                antennas=[0.0, 0.0],
                duration=RESET_TO_NEUTRAL_DURATION_S,
            )
        except Exception:
            logger.exception("returning antennas to neutral failed")
        finally:
            motion_lock.release()

    def _enter_listening() -> None:
        nonlocal wave_thread, listening_wave_thread, followup_deadline
        if wave_thread is not None:
            wave_stop.set()
            wave_thread.join(timeout=0.5)
            wave_thread = None
            wave_stop.clear()
        _reset_antennas_to_neutral()
        listening_wave_stop.clear()
        listening_wave_thread = threading.Thread(
            target=_wave_antennas_listening,
            args=(reachy_mini, listening_wave_stop, motion_lock),
            name="realtime-listening-wave",
            daemon=True,
        )
        listening_wave_thread.start()
        followup_deadline = time.monotonic() + FOLLOWUP_WINDOW_S
        logger.info("entering follow-up listening window (%.0fs)", FOLLOWUP_WINDOW_S)

    def _exit_listening() -> None:
        nonlocal listening_wave_thread, followup_deadline, turn_deadline
        if listening_wave_thread is not None:
            listening_wave_stop.set()
            listening_wave_thread.join(timeout=0.5)
            listening_wave_thread = None
        _reset_antennas_to_neutral()
        followup_deadline = None
        turn_deadline = time.monotonic() + MAX_TURN_S

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
                target=_pump_mic,
                args=(reachy_mini, conn, stop_event, producer_done),
                name="realtime-mic-producer",
                daemon=True,
            )
            producer.start()

            last_audio_at: float | None = None
            for event in conn:
                if stop_event.is_set():
                    logger.info("stop_event set; ending realtime session")
                    break
                if time.monotonic() > session_deadline:
                    logger.warning("realtime session exceeded %.0fs cap; ending", MAX_SESSION_S)
                    break
                if followup_deadline is not None and time.monotonic() > followup_deadline:
                    logger.info("follow-up window elapsed with no speech; ending session")
                    break
                if followup_deadline is None and time.monotonic() > turn_deadline:
                    logger.warning("model response exceeded %.0fs cap; ending", MAX_TURN_S)
                    break

                if event.type == "response.output_audio.delta":
                    if wave_thread is None:
                        wave_stop.clear()
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
                        logger.info("response.done received; entering listening window")
                        _enter_listening()
                elif event.type == "error":
                    logger.error("realtime error event: %s", getattr(event, "error", event))
                    break
                elif event.type == "input_audio_buffer.speech_started":
                    logger.info("user speech started")
                    if followup_deadline is not None:
                        _exit_listening()
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
        listening_wave_stop.set()
        if wave_thread is not None:
            wave_thread.join(timeout=0.5)
        if listening_wave_thread is not None:
            listening_wave_thread.join(timeout=0.5)
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


def run_openai_announcement(reachy_mini: ReachyMini, output_rate: int, message: str) -> None:
    """One-shot OpenAI session: speak `message` and close. No mic input."""
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


# --- Internal: mic producer + tool-call wiring ---------------------------

def _pump_mic(
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
            pcm24k = _resample_int16(mono16k, SDK_INPUT_RATE, OPENAI_REALTIME_RATE)
            payload = base64.b64encode(pcm24k.tobytes()).decode("utf-8")
            conn.input_audio_buffer.append(audio=payload)
    except Exception:
        logger.exception("mic producer thread crashed")


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
    output = _dispatch_tool_call(reachy_mini, name, raw_args, ctx)

    if call_id:
        try:
            conn.conversation.item.create(item={
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(output),
            })
        except Exception:
            logger.exception("sending function_call_output failed")


def _resample_int16(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples
    g = gcd(src_rate, dst_rate)
    floats = samples.astype(np.float32) / 32767.0
    floats = resample_poly(floats, up=dst_rate // g, down=src_rate // g)
    return np.clip(floats * 32767.0, -32768, 32767).astype(np.int16)
