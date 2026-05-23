"""Gemini Live API backend for realtime turns.

Selected by setting `REACHY_CHAT_PROVIDER=gemini` in the daemon's environment.
Auth via `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). The Live API is async-only,
so each turn spins up an asyncio loop in the calling (wake-word) thread —
the public entry points (`run_gemini_turn`, `run_gemini_announcement`) stay
synchronous so `realtime.py` can call them in place of the OpenAI versions.

Mirrors the OpenAI implementation: same antenna wave, same motion lock, same
tool registry from realtime.py, same audio output path. The model uses
Gemini's built-in `google_search` grounding in place of the OpenAI-backed
`web_search` function tool when in this provider mode — so a Gemini-only
deployment doesn't need an OpenAI key.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from reachy_chat.realtime import (
    ANNOUNCE_MAX_S,
    FOLLOWUP_WINDOW_S,
    MAX_SESSION_S,
    MAX_TURN_S,
    RESET_TO_NEUTRAL_DURATION_S,
    _dispatch_tool_call,
    _push_realtime_audio,
    _to_mono_int16,
    _wave_antennas,
    _wave_antennas_listening,
)

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-live-preview"
GEMINI_INPUT_RATE = 16_000   # Matches the SDK's mic rate; no resample needed.
GEMINI_OUTPUT_RATE = 24_000  # PCM16 mono @ 24 kHz from the server.
GEMINI_VOICE = "Aoede"       # Prebuilt voices: Puck, Charon, Kore, Fenrir, Aoede.

# Tools whose backing dependency is OpenAI-specific. We swap in Gemini-native
# equivalents (google_search) and drop the function declaration so the model
# doesn't try to call our handler.
_OPENAI_ONLY_TOOLS = {"web_search"}


# --- Public entry points (called from realtime.py) -----------------------

def run_gemini_turn(
    reachy_mini: ReachyMini,
    stop_event: threading.Event,
    output_rate: int,
    instructions: str,
    oai_tools: list[dict],
) -> None:
    """Synchronous wrapper around the async live-session loop."""
    if not _ensure_genai_available():
        return
    asyncio.run(_run_gemini_turn_async(
        reachy_mini, stop_event, output_rate, instructions, oai_tools,
    ))


def run_gemini_announcement(reachy_mini: ReachyMini, output_rate: int, message: str) -> None:
    if not _ensure_genai_available():
        return
    asyncio.run(_run_gemini_announcement_async(reachy_mini, output_rate, message))


def _ensure_genai_available() -> bool:
    try:
        import google.genai  # noqa: F401
    except ImportError:
        logger.error(
            "google-genai is not installed; install it into the apps venv "
            "or set REACHY_CHAT_PROVIDER=openai"
        )
        return False
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        logger.error("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set; cannot open Gemini Live session")
        return False
    return True


# --- Wake-word turn ------------------------------------------------------

async def _run_gemini_turn_async(
    reachy_mini: ReachyMini,
    stop_event: threading.Event,
    output_rate: int,
    instructions: str,
    oai_tools: list[dict],
) -> None:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    # Live (BidiGenerateContent) is only exposed under v1beta; the default
    # client may fall through to v1 for some preview models.
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
    config = _build_gemini_config(instructions, oai_tools)

    session_deadline = time.monotonic() + MAX_SESSION_S
    turn_deadline = time.monotonic() + MAX_TURN_S
    followup_deadline: float | None = None

    motion_lock = threading.Lock()
    wave_stop = threading.Event()
    wave_thread: threading.Thread | None = None
    listening_wave_stop = threading.Event()
    listening_wave_thread: threading.Thread | None = None

    ctx: dict[str, Any] = {
        "client": client,
        "motion_lock": motion_lock,
        "output_rate": output_rate,
    }

    def _reset_antennas_to_neutral() -> None:
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
            name="gemini-listening-wave",
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

    producer: asyncio.Task | None = None
    try:
        async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
            logger.info(
                "gemini live session opened (model=%s, voice=%s, instructions=%d chars, tools=%d)",
                GEMINI_MODEL, GEMINI_VOICE, len(instructions),
                len(config.get("tools", [])),
            )

            producer_done = asyncio.Event()
            producer = asyncio.create_task(
                _pump_mic_to_gemini(reachy_mini, session, stop_event, producer_done),
                name="gemini-mic-producer",
            )

            # Manual iteration with a periodic timeout so the deadline checks
            # below fire even during the silent follow-up window when no
            # messages arrive.
            messages = session.receive().__aiter__()
            last_audio_at: float | None = None
            while True:
                if stop_event.is_set():
                    logger.info("stop_event set; ending gemini session")
                    break
                if time.monotonic() > session_deadline:
                    logger.warning("gemini session exceeded %.0fs cap; ending", MAX_SESSION_S)
                    break
                if followup_deadline is not None and time.monotonic() > followup_deadline:
                    logger.info("follow-up window elapsed with no speech; ending session")
                    break
                if followup_deadline is None and time.monotonic() > turn_deadline:
                    logger.warning("gemini response exceeded %.0fs cap; ending", MAX_TURN_S)
                    break

                try:
                    message = await asyncio.wait_for(messages.__anext__(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                except StopAsyncIteration:
                    break

                audio_bytes = _extract_audio_bytes(message)
                tool_call = getattr(message, "tool_call", None)
                has_function_calls = tool_call is not None and (
                    getattr(tool_call, "function_calls", None) or []
                )

                # Gemini doesn't emit a "user speech started" event. The first
                # message after we entered the listening window is our cue:
                # the model is responding, so the user must have spoken.
                if followup_deadline is not None and (audio_bytes or has_function_calls):
                    _exit_listening()

                if audio_bytes:
                    if wave_thread is None:
                        wave_stop.clear()
                        wave_thread = threading.Thread(
                            target=_wave_antennas,
                            args=(reachy_mini, wave_stop, motion_lock),
                            name="gemini-antenna-wave",
                            daemon=True,
                        )
                        wave_thread.start()
                    _push_realtime_audio(reachy_mini, audio_bytes, output_rate)
                    last_audio_at = time.monotonic()

                if has_function_calls:
                    await _send_gemini_tool_responses(
                        reachy_mini, session, tool_call.function_calls, ctx,
                    )

                server_content = getattr(message, "server_content", None)
                if server_content is not None and getattr(server_content, "turn_complete", False):
                    logger.info("gemini turn_complete; entering listening window")
                    _enter_listening()

            producer_done.set()

            if last_audio_at is not None:
                tail_wait = max(0.0, 0.4 - (time.monotonic() - last_audio_at))
                if tail_wait:
                    await asyncio.sleep(tail_wait)
    except Exception:
        logger.exception("gemini live turn failed")
    finally:
        if producer is not None:
            producer.cancel()
            try:
                await producer
            except (asyncio.CancelledError, Exception):
                pass
        wave_stop.set()
        listening_wave_stop.set()
        if wave_thread is not None:
            wave_thread.join(timeout=0.5)
        if listening_wave_thread is not None:
            listening_wave_thread.join(timeout=0.5)
        if motion_lock.acquire(timeout=10.0):
            try:
                reachy_mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=0.2)
            except Exception:
                logger.exception("returning antennas to neutral failed")
            finally:
                motion_lock.release()
        else:
            logger.warning("could not acquire motion_lock to reset antennas; skipping")


async def _pump_mic_to_gemini(
    reachy_mini: ReachyMini,
    session,
    stop_event: threading.Event,
    producer_done: asyncio.Event,
) -> None:
    """Read mic samples (16 kHz mono float32) and stream them as PCM16 to Gemini."""
    from google.genai import types

    try:
        while not producer_done.is_set() and not stop_event.is_set():
            sample = reachy_mini.media.get_audio_sample()
            if sample is None or len(sample) == 0:
                await asyncio.sleep(0.005)
                continue
            mono16k = _to_mono_int16(sample)
            blob = types.Blob(data=mono16k.tobytes(), mime_type="audio/pcm;rate=16000")
            await session.send_realtime_input(audio=blob)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("gemini mic producer crashed")


async def _send_gemini_tool_responses(
    reachy_mini: ReachyMini,
    session,
    function_calls,
    ctx: dict[str, Any],
) -> None:
    from google.genai import types

    responses = []
    for fc in function_calls:
        name = getattr(fc, "name", "") or ""
        args = getattr(fc, "args", None) or {}
        call_id = getattr(fc, "id", None)
        output = _dispatch_tool_call(reachy_mini, name, args, ctx)
        responses.append(types.FunctionResponse(id=call_id, name=name, response=output))
    if not responses:
        return
    try:
        await session.send_tool_response(function_responses=responses)
    except Exception:
        logger.exception("sending gemini function_responses failed")


# --- Announcement (timer fired) -----------------------------------------

async def _run_gemini_announcement_async(
    reachy_mini: ReachyMini,
    output_rate: int,
    message: str,
) -> None:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

    instructions = f"Say exactly the following sentence and nothing else: {message}"
    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": {"parts": [{"text": instructions}]},
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": GEMINI_VOICE}},
        },
    }
    deadline = time.monotonic() + ANNOUNCE_MAX_S
    try:
        async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
            await session.send_client_content(
                turns=[types.Content(role="user", parts=[types.Part(text=message)])],
                turn_complete=True,
            )
            logger.info("gemini announcement opened: %r", message)

            last_audio_at: float | None = None
            async for msg in session.receive():
                if time.monotonic() > deadline:
                    logger.warning("gemini announcement exceeded %.0fs cap", ANNOUNCE_MAX_S)
                    break
                audio_bytes = _extract_audio_bytes(msg)
                if audio_bytes:
                    _push_realtime_audio(reachy_mini, audio_bytes, output_rate)
                    last_audio_at = time.monotonic()
                server_content = getattr(msg, "server_content", None)
                if server_content is not None and getattr(server_content, "turn_complete", False):
                    break

            if last_audio_at is not None:
                tail_wait = max(0.0, 0.4 - (time.monotonic() - last_audio_at))
                if tail_wait:
                    await asyncio.sleep(tail_wait)
    except Exception:
        logger.exception("gemini announcement failed")


# --- Helpers -------------------------------------------------------------

def _extract_audio_bytes(message) -> bytes:
    """Pull PCM16 bytes from a Live message, accommodating SDK shape drift."""
    data = getattr(message, "data", None)
    if data:
        return data
    server_content = getattr(message, "server_content", None)
    if server_content is None:
        return b""
    model_turn = getattr(server_content, "model_turn", None)
    if model_turn is None:
        return b""
    parts = getattr(model_turn, "parts", None) or []
    chunks: list[bytes] = []
    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        if inline_data is None:
            continue
        blob = getattr(inline_data, "data", None)
        if blob:
            chunks.append(blob)
    return b"".join(chunks)


def _build_gemini_config(instructions: str, oai_tools: list[dict]) -> dict:
    """Translate our OpenAI-shaped tool list into a Gemini Live config."""
    function_declarations: list[dict] = []
    needs_google_search = False
    for tool in oai_tools:
        if tool.get("type") != "function":
            continue
        name = tool.get("name")
        if name in _OPENAI_ONLY_TOOLS:
            # Substitute a Gemini-native equivalent so the model retains the capability.
            if name == "web_search":
                needs_google_search = True
            continue
        function_declarations.append({
            "name": name,
            "description": tool.get("description", ""),
            "parameters": _strip_unsupported_schema(tool.get("parameters", {})),
        })

    tools: list[dict] = []
    if function_declarations:
        tools.append({"function_declarations": function_declarations})
    if needs_google_search:
        tools.append({"google_search": {}})

    config: dict = {
        "response_modalities": ["AUDIO"],
        "system_instruction": {"parts": [{"text": instructions}]},
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": GEMINI_VOICE}},
        },
    }
    if tools:
        config["tools"] = tools
    return config


def _strip_unsupported_schema(schema):
    """Remove JSON-schema fields the Gemini parameter parser rejects (e.g. `additionalProperties`)."""
    if isinstance(schema, dict):
        return {
            k: _strip_unsupported_schema(v)
            for k, v in schema.items()
            if k not in ("additionalProperties",)
        }
    if isinstance(schema, list):
        return [_strip_unsupported_schema(v) for v in schema]
    return schema
