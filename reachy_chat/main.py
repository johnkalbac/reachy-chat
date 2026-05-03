"""Reachy Chat: wake word -> reply 'hello'. First milestone, no cloud yet.

openWakeWord ships pre-trained models for `alexa`, `hey_jarvis`, `hey_mycroft`,
`hey_rhasspy`, `weather`, `timer`. None of them are "Reachy", so we use
`hey_jarvis` as a stand-in until we train a custom model.
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

logger = logging.getLogger(__name__)

WAKE_WORD = "hey_jarvis"
WAKE_WORD_THRESHOLD = 0.5
SDK_SAMPLE_RATE = 16_000
FRAME_MS = 80
FRAME_SAMPLES = SDK_SAMPLE_RATE * FRAME_MS // 1000  # 1280
GREETING = "hello"


class ReachyChatApp(ReachyMiniApp):
    custom_app_url: str | None = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        model = WakeWordModel(wakeword_models=[WAKE_WORD])

        logger.info("=" * 50)
        logger.info("REACHY CHAT STARTING")
        logger.info("  python: %s", sys.version.split()[0])
        logger.info("  wake word: %s (threshold %.2f)", WAKE_WORD, WAKE_WORD_THRESHOLD)
        logger.info("  frame: %d samples (%d ms)", FRAME_SAMPLES, FRAME_MS)
        logger.info("=" * 50)

        try:
            reachy_mini.media.start_recording()
            reachy_mini.media.start_playing()

            input_rate = reachy_mini.media.get_input_audio_samplerate()
            output_rate = reachy_mini.media.get_output_audio_samplerate()
            if input_rate != SDK_SAMPLE_RATE:
                raise RuntimeError(
                    f"SDK input rate {input_rate} != 16 kHz; resampling not implemented"
                )

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
                            "wake word %r detected (score=%.3f); speaking %r",
                            WAKE_WORD, score, GREETING,
                        )
                        _speak(reachy_mini, GREETING, output_rate)
                        # Drop captured audio to avoid retriggering on the greeting itself.
                        buffer = np.empty(0, dtype=np.int16)
                        drain_until = time.time() + 0.5
                        while time.time() < drain_until:
                            reachy_mini.media.get_audio_sample()
        finally:
            reachy_mini.media.stop_recording()
            reachy_mini.media.stop_playing()


def _to_mono_int16(sample: np.ndarray) -> np.ndarray:
    """SDK gives float32 (n, 2) at 16 kHz; openWakeWord wants int16 mono."""
    if sample.ndim == 2:
        sample = sample.mean(axis=1)
    return (np.clip(sample, -1.0, 1.0) * 32767.0).astype(np.int16)


def _speak(reachy_mini: ReachyMini, text: str, output_rate: int) -> None:
    """Render `text` via espeak-ng, resample to `output_rate`, push to speaker."""
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

    reachy_mini.media.push_audio_sample(floats.reshape(-1, 1))
    # push_audio_sample is non-blocking; sleep for the playback duration plus a margin.
    time.sleep(len(floats) / output_rate + 0.1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = ReachyChatApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
