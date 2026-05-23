# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Reachy Mini Python app: package with a `[project.entry-points."reachy_mini_apps"]` entry in [pyproject.toml](pyproject.toml). Current behavior: wake-word ("hey jarvis") → multi-turn voice conversation (audio in, audio out over WebSocket). After each model response the robot enters an 8 s follow-up listening window (signalled by a slow antenna waggle); if the user speaks during that window it becomes the next turn in the same session. The session closes when the window elapses with no speech, or when `MAX_SESSION_S` (3 min) is reached — control then returns to wake-word listening.

The realtime backend is selectable via `provider.name` in [config.toml](config.toml): `"openai"` (default — OpenAI Realtime API, needs `OPENAI_API_KEY`) or `"gemini"` (Gemini Live API, needs `GEMINI_API_KEY`). Dispatch lives in [reachy_chat/realtime.py](reachy_chat/realtime.py) (`realtime_turn()`, `announce_via_realtime()`); the OpenAI backend is [reachy_chat/openai_realtime.py](reachy_chat/openai_realtime.py) and the Gemini backend is [reachy_chat/gemini_realtime.py](reachy_chat/gemini_realtime.py). Both providers share the tool registry, motion lock, antenna wave, audio path, and prompt loading (all in `realtime.py`) — only the wire protocol differs per backend. When provider=gemini the `web_search` function tool is replaced by Gemini's built-in `google_search` grounding so a Gemini-only deployment doesn't need an OpenAI key.

## Where the code actually runs

This is a **device-side** app. The Windows clone at `c:\git\github\reachy-chat` is a mirror; primary edits happen on the robot via VS Code Remote-SSH to `pollen@reachy-mini.local` (password `root`), opening `/home/pollen/reachy-chat/`. Commit/push from whichever side made the edit.

On the robot, `reachy-mini-daemon` (systemd, dashboard at `http://reachy-mini.local:8000`) discovers the app via the entry point, launches it as `python -u -m reachy_chat.main`, hands `run()` a connected `ReachyMini` instance and a `stop_event`, and sends `SIGINT` to stop. Only one Reachy Mini app runs at a time.

The daemon's apps venv is `/venvs/apps_venv/` — always invoke its `pip` / `python` explicitly; the system Python is the wrong target.

## Architecture

Entry point: `reachy_chat.main:ReachyChat` (subclass of `reachy_mini.ReachyMiniApp`). `run(reachy_mini, stop_event)` opens SDK audio streams, then loops:

1. Read mic samples (`reachy_mini.media.get_audio_sample()` → float32 stereo at 16 kHz).
2. Convert to int16 mono and accumulate into 80 ms / 1280-sample frames.
3. Feed each frame to openWakeWord (`Model.predict`, ONNX backend).
4. On `score >= WAKE_WORD_THRESHOLD`: hand off to `reachy_chat.realtime.realtime_turn()` for one multi-turn Realtime session (model response → 8 s follow-up listening window → repeat until the user goes silent), then flush the openWakeWord feature buffer and resume.

User-facing tunables (`WAKE_WORD`, `WAKE_WORD_THRESHOLD`, `REACHY_CHAT_PROVIDER`, `REALTIME_MODEL`, `REALTIME_VOICE`, `WEB_SEARCH_MODEL`, `WEB_SEARCH_TIMEOUT_S`, `GEMINI_MODEL`, `GEMINI_VOICE`, `ANNOUNCE_MAX_S`, `FOLLOWUP_WINDOW_S`, `MAX_SESSION_S`, `MAX_TURN_S`, `RESET_TO_NEUTRAL_DURATION_S`) live in [config.toml](config.toml) at the repo root, grouped into `[wake_word]`, `[provider]`, `[openai]`, `[gemini]`, `[timing]` sections. [reachy_chat/config.py](reachy_chat/config.py) parses the file once at import via `tomllib` and exposes the values as module-level constants — every consumer (`main.py`, `realtime.py`, `openai_realtime.py`, `gemini_realtime.py`) imports from there. Values are read once at process start, so edits require an app restart. If `config.toml` is missing or fails to parse, `_DEFAULTS` in `config.py` kicks in (keep that dict in sync with the file). Lower-level shape constants that aren't worth surfacing — `FRAME_MS`, `LISTENING_WAVE_*`, `WAVE_*`, DOA limits, `PROVIDER_AUDIO_RATE`, `OPENAI_REALTIME_RATE`, `SDK_INPUT_RATE` — stay at the top of `main.py` / `realtime.py` / `openai_realtime.py` / `gemini_realtime.py`. `_speak()` and `GREETING` in main.py are kept around for offline diagnostics — not on the happy path.

### Realtime session lifecycle

`realtime_turn()` in `realtime.py` builds the `instructions` string and the `tools` list, then dispatches to whichever backend matches `provider.name`. The OpenAI backend lives in [reachy_chat/openai_realtime.py](reachy_chat/openai_realtime.py) (`run_openai_turn`); the Gemini backend lives in [reachy_chat/gemini_realtime.py](reachy_chat/gemini_realtime.py) (`run_gemini_turn`). Both receive the same `(reachy_mini, stop_event, output_rate, instructions, tools)` signature.

`run_openai_turn()` opens one synchronous WebSocket session via `OpenAI().realtime.connect(model="gpt-realtime")`, configures it with `session.update` (PCM16 mono @ 24 kHz default, `server_vad` turn detection, `output_modalities: ["audio"]`, voice + instructions), and runs two threads:

- **Producer** (`_pump_mic` in `openai_realtime.py`): reads `get_audio_sample()`, converts to int16 mono, resamples 16 kHz → 24 kHz with `resample_poly(up=3, down=2)`, base64-encodes, sends as `input_audio_buffer.append`. Runs for the entire session — mic stays open during the follow-up listening window so server VAD can pick up the next utterance.
- **Main**: iterates `for event in conn:` as a small state machine — `SPEAKING` while model audio is arriving, `LISTENING` for `FOLLOWUP_WINDOW_S` after each `response.done` with no pending tool continuation. On `response.output_audio.delta` it base64-decodes, resamples 24 kHz → SDK output rate, pushes to speaker. On `response.done` (with no pending tool continuation) it stops the speaking antenna wave, sweeps antennas to neutral (`RESET_TO_NEUTRAL_DURATION_S`), starts the slow `_wave_antennas_listening` cue, and sets `followup_deadline = now + FOLLOWUP_WINDOW_S`. On `input_audio_buffer.speech_started` it inverts the transition — stops the listening wave, resets to neutral, clears the deadline, and waits for the model's next response. The session ends on: silence past the follow-up deadline, `MAX_SESSION_S` total elapsed, `MAX_TURN_S` per-response cap, `error` event, or `stop_event`.

Server VAD means we never send `input_audio_buffer.commit` or `response.create` ourselves — the server detects the user's turn end and starts the response, *except* after a tool call: when `response.function_call_arguments.done` arrives we send the function output back via `conversation.item.create`, mark `need_continuation=True`, and on the next `response.done` we explicitly call `conn.response.create()` to let the model continue. Without this, the assistant's audio reply gets truncated whenever it calls a tool. Tool-call continuations stay in `SPEAKING` — they do **not** trigger the follow-up listening window.

The Gemini backend ([reachy_chat/gemini_realtime.py](reachy_chat/gemini_realtime.py)) mirrors the same state machine on the async receive loop. Gemini emits no equivalent of `speech_started`, so the `LISTENING → SPEAKING` transition is inferred from the next non-empty message (audio chunk or tool call) after `turn_complete`. The receive loop uses `asyncio.wait_for(messages.__anext__(), timeout=0.5)` so deadline checks fire even during the silent listening window.

### Tool calls

The realtime model has a fixed set of function tools, dispatched through a registry in [reachy_chat/realtime.py](reachy_chat/realtime.py):

```python
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
```

Each tool has a `_<name>_schema()` builder (its JSON-schema definition for `session.update`) and a `_tool_<name>(reachy_mini, args, ctx)` handler. `_build_tools()` calls every builder; builders return `None` to opt out (e.g. emotions library failed to load) — adding or removing a tool is one place.

**Sync vs async execution.** Most handlers kick off work in a daemon thread and return `{"status": "started", ...}` immediately so the model can keep speaking. The exception is `web_search`, which **synchronously** calls the OpenAI Responses API (with `tools=[{"type": "web_search"}]`, same `OPENAI_API_KEY`) and returns the answer in the function-call output. The realtime event loop blocks for up to `WEB_SEARCH_TIMEOUT_S` (15 s) during the call — that's fine because the model's response is paused waiting for the tool result anyway. The handler tries `web_search` first and falls back to `web_search_preview` on a tool-name mismatch since OpenAI has shipped both.

**Continuation after tool calls.** Server VAD normally means we never send `input_audio_buffer.commit` or `response.create` ourselves — the server detects the user's turn end and starts the response. *Except* after a tool call: when `response.function_call_arguments.done` arrives we send the function output back via `conversation.item.create`, mark `need_continuation=True`, and on the next `response.done` we explicitly call `conn.response.create()` to let the model continue. Without this, the assistant's audio reply gets truncated whenever it calls a tool.

**Recorded-move library cache.** `_get_recorded_moves(library_id)` lazy-loads and caches both the emotions and dances datasets in `_recorded_libraries: dict[str, RecordedMoves]`. `warm_libraries()` (called from a daemon thread at startup) primes both. `_execute_recorded_move(reachy_mini, library_id, name, motion_lock)` is the shared player.

**Motion lock.** The antenna wave + the recorded-move player share `motion_lock`. Wave acquires it for each `set_target` (microseconds); a `play_move` holds it for the duration of the clip. Without this they'd fight over motion targets.

**Volume scaling chokepoint.** `apply_output_volume(samples)` applies `_volume_pct` / `_muted` (state in `realtime.py`, guarded by `_volume_lock`). Every `push_audio_sample` call routes through it: `_push_realtime_audio`, `play_ready_chime`, `_speak`, and the timer announcement. `set_volume` / `mute` / `unmute` tools mutate the state and return the new state to the model.

**Realtime session lock.** A device-wide `_realtime_session_lock` (in `realtime.py`) ensures only one realtime session is active at a time. `realtime_turn` and `announce_via_realtime` both acquire it. A timer firing while a wake-word turn is in flight waits up to 30 s for the lock before announcing — wake-word turns have higher implicit priority.

**Timer service.** [reachy_chat/timers.py](reachy_chat/timers.py) defines `TimerService` (worker thread + `heapq` of `(deadline, id, label)`), instantiated in `ReachyChat.run()` after the audio streams are up, registered as the module singleton via `timers.set_service(...)`, and stopped in the `finally` clause. The `set_timer` tool handler calls `timers.get_service().add_timer(seconds, label)`. When a timer fires, the worker calls `play_ready_chime` then `announce_via_realtime` with the message `f"Timer {label} is done."` — a one-shot realtime session that triggers a response with no mic input via `conn.response.create()`.

**Graceful degradation.** Any builder that returns `None` simply omits its tool from the session (no emotions library → no `play_emotion`; timer service not running → `set_timer` returns an error to the model). The conversation always works with the tools that *are* available.

### System prompt composition

`instructions` is loaded from `prompts/*.md` at the repo root (path resolved as `Path(__file__).resolve().parent.parent / "prompts"` from `realtime.py`). All `.md` files are sorted lexicographically and concatenated with `\n\n` separators; files starting with `_` or `.` are skipped so a fragment can be disabled without deletion. Reloaded on every `realtime_turn()` call — edits take effect on the next wake-word. If the dir is missing or empty, falls back to `_DEFAULT_INSTRUCTIONS` in [reachy_chat/realtime.py](reachy_chat/realtime.py).

The path is module-relative (one directory up from `realtime.py`), which assumes the editable install layout (`pip install -e .`). A non-editable wheel install wouldn't bundle the `prompts/` directory and would always fall back to the default.

## Non-obvious gotchas — do not "clean up" these

- **Post-detection feature-buffer flush** in [reachy_chat/main.py](reachy_chat/main.py): after a wake-word fires (and the realtime turn returns), the code feeds 25 frames of silence into the wake-word model and drains ~0.3 s of mic audio. openWakeWord keeps a rolling ~1.5 s feature buffer; `model.reset()` only clears the prediction buffer, not the feature buffer, so without this flush the next `predict()` retriggers on the same audio. Don't replace it with `model.reset()`. The flush is *also* needed because the wake-word loop is paused for the full duration of the realtime turn — the feature buffer doesn't roll on its own.
- **`OPENAI_API_KEY` (or `GEMINI_API_KEY`, when `provider.name = "gemini"`) must be set in the systemd unit, not just an interactive shell.** The daemon doesn't inherit your login env. Use `sudo systemctl edit reachy-mini-daemon` and add `Environment=OPENAI_API_KEY=...` (and/or `Environment=GEMINI_API_KEY=...`) under `[Service]`, then restart. Verify with `systemctl show reachy-mini-daemon -p Environment`. The provider toggle itself is *not* an env var — it lives in [config.toml](config.toml).
- **OpenAI Realtime requires 24 kHz PCM16 mono** in both directions; the SDK gives us 16 kHz. `realtime_turn` resamples 16 kHz → 24 kHz on the way up (`resample_poly(up=3, down=2)`) and 24 kHz → SDK output rate on the way down. Don't strip the resampling thinking the SDK can negotiate it — the API can't.
- **`openwakeword` is intentionally absent from `pyproject.toml` `dependencies`**. Its setup pins `tflite-runtime`, which has no wheel for Python 3.12 on aarch64 (the daemon's apps venv). Install it with `pip install --no-deps openwakeword` and force the ONNX backend (`inference_framework="onnx"` — already set in `run()`). Don't add it to `dependencies`; don't drop the `--no-deps` from the README.
- **espeak-ng is a system dependency**, not a Python one — installed once via `sudo apt install espeak-ng`. The app shells out to it. Currently only used by the unused-on-happy-path `_speak()` helper.
- The SDK input rate must equal `SDK_SAMPLE_RATE` (16 kHz); the code raises if not. Input resampling against the SDK is not implemented — only output resampling (and the dedicated 16 kHz → 24 kHz path inside `realtime_turn`).

## Common commands (run on the robot as `pollen`)

```bash
# Install / reinstall after pyproject.toml changes
/venvs/apps_venv/bin/pip install -e .

# First-time openwakeword install (no-deps is required)
/venvs/apps_venv/bin/pip install --no-deps openwakeword

# Pre-download the wake-word ONNX models
/venvs/apps_venv/bin/python -c "import openwakeword.utils; openwakeword.utils.download_models()"

# Pre-fetch the emotions library (HuggingFace dataset; cached after first call)
/venvs/apps_venv/bin/reachy-chat-prefetch

# Validate the app metadata so the daemon can discover it
/venvs/apps_venv/bin/reachy-mini-app-assistant check .

# Run directly (bypasses the daemon's app manager; daemon must still be up for hardware)
/venvs/apps_venv/bin/python -m reachy_chat.main

# Start / stop via the daemon REST API
curl -X POST http://reachy-mini.local:8000/api/apps/start-app/reachy-chat
curl -X POST http://reachy-mini.local:8000/api/apps/stop-current-app

# Tail logs while testing (filters out uvicorn access lines)
sudo journalctl -u reachy-mini-daemon -f | grep -v "uvicorn\|GET \|POST "
```

Inner loop for code edits: edit a `.py` in the Remote-SSH pane → click Stop then Start on `reachy-chat` in the dashboard → watch journalctl. Editable install means no reinstall needed for code changes; only re-run `pip install -e .` when entry points or deps in `pyproject.toml` change.

## Tests / lint

There is no test suite and no lint config in this repo. Don't fabricate `pytest` / `ruff` / `mypy` invocations.
