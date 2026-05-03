# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Reachy Mini Python app: package with a `[project.entry-points."reachy_mini_apps"]` entry in [pyproject.toml](pyproject.toml). Current behavior: wake-word ("hey jarvis") → one-shot OpenAI Realtime API conversation (audio in, audio out over WebSocket) → return to wake-word listening. Single-turn — no conversation memory across detections yet.

## Where the code actually runs

This is a **device-side** app. The Windows clone at `c:\git\github\reachy-chat` is a mirror; primary edits happen on the robot via VS Code Remote-SSH to `pollen@reachy-mini.local` (password `root`), opening `/home/pollen/reachy-chat/`. Commit/push from whichever side made the edit.

On the robot, `reachy-mini-daemon` (systemd, dashboard at `http://reachy-mini.local:8000`) discovers the app via the entry point, launches it as `python -u -m reachy_chat.main`, hands `run()` a connected `ReachyMini` instance and a `stop_event`, and sends `SIGINT` to stop. Only one Reachy Mini app runs at a time.

The daemon's apps venv is `/venvs/apps_venv/` — always invoke its `pip` / `python` explicitly; the system Python is the wrong target.

## Architecture

Entry point: `reachy_chat.main:ReachyChat` (subclass of `reachy_mini.ReachyMiniApp`). `run(reachy_mini, stop_event)` opens SDK audio streams, then loops:

1. Read mic samples (`reachy_mini.media.get_audio_sample()` → float32 stereo at 16 kHz).
2. Convert to int16 mono and accumulate into 80 ms / 1280-sample frames.
3. Feed each frame to openWakeWord (`Model.predict`, ONNX backend).
4. On `score >= WAKE_WORD_THRESHOLD`: hand off to `reachy_chat.realtime.realtime_turn()` for one OpenAI Realtime conversation, then flush the openWakeWord feature buffer and resume.

Wake-word tunables (`WAKE_WORD`, `WAKE_WORD_THRESHOLD`, `FRAME_MS`) sit at the top of [reachy_chat/main.py](reachy_chat/main.py); realtime tunables (`REALTIME_MODEL`, `REALTIME_VOICE`, `REALTIME_INSTRUCTIONS`, `MAX_TURN_S`) at the top of [reachy_chat/realtime.py](reachy_chat/realtime.py). `_speak()` and `GREETING` in main.py are kept around for offline diagnostics — not on the happy path.

### Realtime session lifecycle

`realtime_turn()` opens one synchronous WebSocket session via `OpenAI().realtime.connect(model="gpt-realtime")`, configures it with `session.update` (PCM16 mono @ 24 kHz default, `server_vad` turn detection, `output_modalities: ["audio"]`, voice + instructions), and runs two threads:

- **Producer** (`_pump_mic_to_realtime`): reads `get_audio_sample()`, converts to int16 mono, resamples 16 kHz → 24 kHz with `resample_poly(up=3, down=2)`, base64-encodes, sends as `input_audio_buffer.append`.
- **Main**: iterates `for event in conn:`. On `response.output_audio.delta` it base64-decodes, resamples 24 kHz → SDK output rate, pushes to speaker. On `response.done` (or `error`, or `MAX_TURN_S` cap, or `stop_event`) it sets a done flag, joins the producer, and lets the context manager close the WebSocket.

Server VAD means we never send `input_audio_buffer.commit` or `response.create` ourselves — the server detects the user's turn end and starts the response.

### System prompt composition

`instructions` is loaded from `prompts/*.md` at the repo root (path resolved as `Path(__file__).resolve().parent.parent / "prompts"` from `realtime.py`). All `.md` files are sorted lexicographically and concatenated with `\n\n` separators; files starting with `_` or `.` are skipped so a fragment can be disabled without deletion. Reloaded on every `realtime_turn()` call — edits take effect on the next wake-word. If the dir is missing or empty, falls back to `_DEFAULT_INSTRUCTIONS` in [reachy_chat/realtime.py](reachy_chat/realtime.py).

The path is module-relative (one directory up from `realtime.py`), which assumes the editable install layout (`pip install -e .`). A non-editable wheel install wouldn't bundle the `prompts/` directory and would always fall back to the default.

## Non-obvious gotchas — do not "clean up" these

- **Post-detection feature-buffer flush** in [reachy_chat/main.py](reachy_chat/main.py): after a wake-word fires (and the realtime turn returns), the code feeds 25 frames of silence into the wake-word model and drains ~0.3 s of mic audio. openWakeWord keeps a rolling ~1.5 s feature buffer; `model.reset()` only clears the prediction buffer, not the feature buffer, so without this flush the next `predict()` retriggers on the same audio. Don't replace it with `model.reset()`. The flush is *also* needed because the wake-word loop is paused for the full duration of the realtime turn — the feature buffer doesn't roll on its own.
- **`OPENAI_API_KEY` must be set in the systemd unit, not just an interactive shell.** The daemon doesn't inherit your login env. Use `sudo systemctl edit reachy-mini-daemon` and add `Environment=OPENAI_API_KEY=...` under `[Service]`, then restart. Verify with `systemctl show reachy-mini-daemon -p Environment`.
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
