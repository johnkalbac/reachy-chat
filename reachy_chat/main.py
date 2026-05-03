"""Reachy Chat: wake on 'Reachy', reply 'hello'. First milestone — no cloud yet."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import wave
from math import gcd
from pathlib import Path

import numpy as np
import pvporcupine
from scipy.signal import resample_poly

from reachy_mini import ReachyMini, ReachyMiniApp

logger = logging.getLogger(__name__)

PORCUPINE_SAMPLE_RATE = 16_000
DEFAULT_WAKE_WORD_PATH = Path.home() / ".config" / "reachy_chat" / "wake_word.ppn"
GREETING = "hello"


class ReachyChatApp(ReachyMiniApp):
    custom_app_url: str | None = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
        if not access_key:
            raise RuntimeError(
                "PICOVOICE_ACCESS_KEY is not set. Get a free key at "
                "https://console.picovoice.ai/ and export it before starting the app."
            )

        keyword_path = Path(os.environ.get("WAKE_WORD_PATH", str(DEFAULT_WAKE_WORD_PATH)))
        if not keyword_path.is_file():
            raise FileNotFoundError(
                f"Wake-word file not found at {keyword_path}. Generate a 'Reachy' "
                ".ppn for raspberry-pi at https://console.picovoice.ai/ and place it "
                "there, or override with WAKE_WORD_PATH."
            )

        porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[str(keyword_path)],
        )

        logger.info("=" * 50)
        logger.info("REACHY CHAT STARTING")
        logger.info("  python: %s", sys.version.split()[0])
        logger.info(
            "  porcupine: frame_length=%d sample_rate=%d",
            porcupine.frame_length,
            porcupine.sample_rate,
        )
        logger.info("  wake_word: %s", keyword_path)
        logger.info("=" * 50)

        if porcupine.sample_rate != PORCUPINE_SAMPLE_RATE:
            raise RuntimeError(
                f"Porcupine sample rate {porcupine.sample_rate} != expected 16 kHz"
            )

        try:
            reachy_mini.media.start_recording()
            reachy_mini.media.start_playing()

            input_rate = reachy_mini.media.get_input_audio_samplerate()
            output_rate = reachy_mini.media.get_output_audio_samplerate()
            if input_rate != PORCUPINE_SAMPLE_RATE:
                raise RuntimeError(
                    f"SDK input rate {input_rate} != 16 kHz; resampling not implemented"
                )

            buffer = np.empty(0, dtype=np.int16)
            frame_len = porcupine.frame_length

            while not stop_event.is_set():
                sample = reachy_mini.media.get_audio_sample()
                if sample is None or len(sample) == 0:
                    time.sleep(0.005)
                    continue

                buffer = np.concatenate([buffer, _to_mono_int16(sample)])

                while len(buffer) >= frame_len:
                    frame = buffer[:frame_len]
                    buffer = buffer[frame_len:]
                    if porcupine.process(frame) >= 0:
                        logger.info("wake word detected; speaking %r", GREETING)
                        _speak(reachy_mini, GREETING, output_rate)
                        # Drop captured audio to avoid retriggering on the greeting.
                        buffer = np.empty(0, dtype=np.int16)
                        drain_until = time.time() + 0.5
                        while time.time() < drain_until:
                            reachy_mini.media.get_audio_sample()
        finally:
            try:
                reachy_mini.media.stop_recording()
                reachy_mini.media.stop_playing()
            finally:
                porcupine.delete()


def _to_mono_int16(sample: np.ndarray) -> np.ndarray:
    """SDK gives float32 (n, 2) at 16 kHz; Porcupine wants int16 mono at 16 kHz."""
    if sample.ndim == 2:
        sample = sample.mean(axis=1)
    return (np.clip(sample, -1.0, 1.0) * 32767.0).astype(np.int16)


def _speak(reachy_mini: ReachyMini, text: str, output_rate: int) -> None:
    """Render `text` to PCM via espeak-ng, resample to `output_rate`, push to speaker."""
    wav_path = Path("/tmp/reachy_chat_tts.wav")
    # pyttsx3's espeak driver has known issues with repeated runAndWait calls in the
    # same process; calling espeak-ng directly is one syscall and bulletproof.
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
