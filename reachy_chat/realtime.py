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
import logging
import threading
import time
from math import gcd
from pathlib import Path

import numpy as np
from openai import OpenAI
from scipy.signal import resample_poly

from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)

REALTIME_MODEL = "gpt-realtime"
REALTIME_RATE = 24_000  # OpenAI Realtime default: PCM16 mono @ 24 kHz, both directions.
REALTIME_VOICE = "coral"
MAX_TURN_S = 30.0  # hard cap; an unresponsive session shouldn't pin the device.
SDK_INPUT_RATE = 16_000  # matches main.SDK_SAMPLE_RATE; mic is 16 kHz.

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_DEFAULT_INSTRUCTIONS = (
    "You are a helpful assistant living inside a small desktop robot named Reachy. "
    "Keep replies short and conversational — usually one or two sentences."
)


def realtime_turn(reachy_mini: ReachyMini, stop_event: threading.Event, output_rate: int) -> None:
    client = OpenAI()
    deadline = time.monotonic() + MAX_TURN_S
    producer_done = threading.Event()
    instructions = _load_instructions()

    try:
        with client.realtime.connect(model=REALTIME_MODEL) as conn:
            conn.session.update(
                session={
                    "type": "realtime",
                    "model": REALTIME_MODEL,
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {"turn_detection": {"type": "server_vad"}},
                        "output": {"voice": REALTIME_VOICE},
                    },
                    "instructions": instructions,
                }
            )
            logger.info(
                "realtime session opened (model=%s, voice=%s, instructions=%d chars)",
                REALTIME_MODEL, REALTIME_VOICE, len(instructions),
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
                    pcm_bytes = base64.b64decode(event.delta)
                    _push_realtime_audio(reachy_mini, pcm_bytes, output_rate)
                    last_audio_at = time.monotonic()
                elif event.type == "response.done":
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
