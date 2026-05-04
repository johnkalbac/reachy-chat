"""Reachy Chat: wake word -> OpenAI Realtime API conversation turn.

openWakeWord ships pre-trained models for `alexa`, `hey_jarvis`, `hey_mycroft`,
`hey_rhasspy`, `weather`, `timer`. None of them are "Reachy", so we use
`hey_jarvis` as a stand-in until we train a custom model.

After the wake word fires, control hands off to `realtime_turn()` which
opens a WebSocket session against `gpt-realtime`, streams the user's
request up, and plays the assistant's audio reply back through the speaker.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
import wave
from math import gcd
from pathlib import Path

import numpy as np
from openwakeword.model import Model as WakeWordModel
from scipy.signal import resample_poly

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose

from reachy_chat import timers
from reachy_chat.realtime import (
    apply_output_volume,
    play_ready_chime,
    realtime_turn,
    warm_libraries,
)

logger = logging.getLogger(__name__)

WAKE_WORD = "hey_jarvis"
WAKE_WORD_THRESHOLD = 0.5
SDK_SAMPLE_RATE = 16_000
FRAME_MS = 80
FRAME_SAMPLES = SDK_SAMPLE_RATE * FRAME_MS // 1000  # 1280
GREETING = "hello"


class ReachyChat(ReachyMiniApp):
    custom_app_url: str | None = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        # Force ONNX backend; tflite-runtime has no wheel for Python 3.12 aarch64.
        model = WakeWordModel(wakeword_models=[WAKE_WORD], inference_framework="onnx")

        # Warm the emotions + dances libraries in the background so the first
        # wake-word doesn't pay the (one-time) HuggingFace download.
        threading.Thread(target=warm_libraries, name="warm-libraries", daemon=True).start()

        logger.info("=" * 50)
        logger.info("REACHY CHAT STARTING")
        logger.info("  python: %s", sys.version.split()[0])
        logger.info("  wake word: %s (threshold %.2f)", WAKE_WORD, WAKE_WORD_THRESHOLD)
        logger.info("  frame: %d samples (%d ms)", FRAME_SAMPLES, FRAME_MS)
        logger.info("=" * 50)

        timer_service: timers.TimerService | None = None
        try:
            reachy_mini.media.start_recording()
            reachy_mini.media.start_playing()

            input_rate = reachy_mini.media.get_input_audio_samplerate()
            output_rate = reachy_mini.media.get_output_audio_samplerate()
            if input_rate != SDK_SAMPLE_RATE:
                raise RuntimeError(
                    f"SDK input rate {input_rate} != 16 kHz; resampling not implemented"
                )

            timer_service = timers.TimerService(reachy_mini, output_rate)
            timer_service.start()
            timers.set_service(timer_service)

            play_ready_chime(reachy_mini, output_rate)

            buffer = np.empty(0, dtype=np.int16)

            while not stop_event.is_set():
                sample = reachy_mini.media.get_audio_sample()
                if sample is None or len(sample) == 0:
                    time.sleep(0.005)
                    continue

                buffer = np.concatenate([buffer, _to_mono_int16(sample)])

                while len(buffer) >= FRAME_SAMPLES:
                    frame = buffer[:FRAME_SAMPLES]
                    buffer = buffer[FRAME_SAMPLES:]
                    scores = model.predict(frame)
                    score = scores.get(WAKE_WORD, 0.0)
                    if score >= WAKE_WORD_THRESHOLD:
                        logger.info(
                            "wake word %r detected (score=%.3f); opening realtime turn",
                            WAKE_WORD, score,
                        )
                        threading.Thread(
                            target=_do_nod, args=(reachy_mini,),
                            name="wake-word-nod", daemon=True,
                        ).start()
                        realtime_turn(reachy_mini, stop_event, output_rate)
                        # The model keeps a rolling ~1.5s feature buffer; without
                        # flushing it past the wake-word audio, the next predict()
                        # retriggers on the same features. model.reset() only
                        # clears the prediction buffer, not the feature buffer —
                        # so feed silence to roll the window forward.
                        silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
                        for _ in range(25):  # 25 * 80 ms = 2.0 s
                            model.predict(silence)
                        buffer = np.empty(0, dtype=np.int16)
                        # Discard any mic audio left in the SDK queue between the
                        # producer thread exiting and us resuming wake-word reads.
                        drain_until = time.time() + 0.3
                        while time.time() < drain_until:
                            reachy_mini.media.get_audio_sample()
        finally:
            if timer_service is not None:
                timers.set_service(None)
                timer_service.stop()
            reachy_mini.media.stop_recording()
            reachy_mini.media.stop_playing()


def _to_mono_int16(sample: np.ndarray) -> np.ndarray:
    """SDK gives float32 (n, 2) at 16 kHz; openWakeWord wants int16 mono."""
    if sample.ndim == 2:
        sample = sample.mean(axis=1)
    return (np.clip(sample, -1.0, 1.0) * 32767.0).astype(np.int16)


def _do_nod(reachy_mini: ReachyMini) -> None:
    """A gentle 'I heard you' nod: pitch down ~12 deg, then back up."""
    try:
        down = create_head_pose(pitch=np.deg2rad(12.0))
        neutral = create_head_pose()
        reachy_mini.goto_target(down, antennas=[0.0, 0.0], duration=0.2)
        reachy_mini.goto_target(neutral, antennas=[0.0, 0.0], duration=0.25)
    except Exception:
        logger.exception("nod failed")


def _speak(reachy_mini: ReachyMini, text: str, output_rate: int) -> None:
    """Render `text` via espeak-ng, resample to `output_rate`, push to speaker.

    Unused on the happy path now that wake-word hands off to realtime_turn().
    Kept around for offline diagnostics when OPENAI_API_KEY isn't available.
    """
    wav_path = Path("/tmp/reachy_chat_tts.wav")
    subprocess.run(
        ["espeak-ng", "-w", str(wav_path), text],
        check=True,
        capture_output=True,
    )

    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise RuntimeError(f"Unexpected TTS sample width: {sample_width} bytes")

    pcm = np.frombuffer(frames, dtype=np.int16)
    if n_channels == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1).astype(np.int16)
    floats = pcm.astype(np.float32) / 32767.0

    if sample_rate != output_rate:
        g = gcd(sample_rate, output_rate)
        floats = resample_poly(floats, up=output_rate // g, down=sample_rate // g)
        floats = floats.astype(np.float32)

    reachy_mini.media.push_audio_sample(apply_output_volume(floats).reshape(-1, 1))
    # push_audio_sample is non-blocking; sleep for the playback duration plus a margin.
    time.sleep(len(floats) / output_rate + 0.1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = ReachyChat()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
