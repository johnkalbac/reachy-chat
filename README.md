---
title: reachy-chat
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# reachy-chat

An AI-enabled chat assistant for the Reachy Mini Wireless. Wake on a
keyword, then have a one-shot voice conversation with OpenAI's Realtime
API: the user's request streams up over WebSocket, the assistant's audio
reply streams back to the speaker, and we return to wake-word listening.

The wake word right now is **"hey jarvis"**, not "Reachy". Reason:
[openWakeWord](https://github.com/dscripka/openWakeWord) is fully
license-free but only ships a fixed set of pre-trained models. Training a
custom "Reachy" model takes a Colab notebook from the openWakeWord docs —
deferred to a follow-up.

## Resources

- [SDK Documentation](https://huggingface.co/docs/reachy_mini)
- [Simulation Setup](https://huggingface.co/docs/reachy_mini/platforms/simulation/get_started)
- [Python SDK Reference](https://huggingface.co/docs/reachy_mini/SDK/python-sdk)
- [Building & Publishing Apps](https://huggingface.co/docs/reachy_mini/SDK/apps)
- [Wireless Development Workflow](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/development_workflow)
- [openWakeWord](https://github.com/dscripka/openWakeWord)
- [Examples (upstream)](https://github.com/pollen-robotics/reachy_mini/tree/main/examples)

## Architecture

The app is a Reachy Mini Python app — a package with a
`[project.entry-points."reachy_mini_apps"]` entry in `pyproject.toml`. The
daemon (`reachy-mini-daemon`, systemd service on the robot, dashboard at
`http://reachy-mini.local:8000`) discovers it, launches it as a subprocess
(`python -u -m reachy_chat.main`), hands it a connected `ReachyMini`
instance and a `stop_event`, and sends `SIGINT` to stop. Only one Reachy
Mini app runs at a time.

`reachy_chat.main:ReachyChatApp.run()` opens the SDK audio streams, feeds
16 kHz mono int16 PCM frames (80 ms / 1280 samples each) into openWakeWord,
and on detection renders the reply through `espeak-ng` and pushes the
resulting samples back to the speaker.

## Required device-side setup (one-time)

These steps run on the robot, not on your dev machine.

### 1. SSH in
```bash
ssh pollen@reachy-mini.local   # password: root
```

### 2. Install espeak-ng (the TTS backend)
```bash
sudo apt update && sudo apt install -y espeak-ng
```

### 3. Clone this repo and install in editable mode
```bash
cd ~ && git clone git@github.com:<your-user>/reachy-chat.git
cd reachy-chat

# openWakeWord pins tflite-runtime, which has no wheel for Python 3.12 on
# aarch64 (the daemon's apps venv). Install it without deps and provide
# what it actually needs at runtime; we use the ONNX backend.
/venvs/apps_venv/bin/pip install --no-deps openwakeword
/venvs/apps_venv/bin/pip install -e .
```

### 4. Pre-download the openWakeWord models
```bash
/venvs/apps_venv/bin/python -c "import openwakeword.utils; openwakeword.utils.download_models()"
```

### 4b. Pre-fetch the recorded-move libraries

The realtime model can call `play_emotion(...)` and `play_dance(...)` to
trigger animation clips from two HuggingFace datasets:

- [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library) — ~80 short emotional reactions.
- [`pollen-robotics/reachy-mini-dances-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library) — longer choreographed routines.

The datasets are downloaded on first use; pre-fetching avoids a
multi-second delay on the first wake-word and the first dance.

```bash
/venvs/apps_venv/bin/reachy-chat-prefetch
```

### 5. Validate the app metadata
```bash
/venvs/apps_venv/bin/reachy-mini-app-assistant check .
```

### 6. Provide the OpenAI API key to the daemon

The app reads `OPENAI_API_KEY` from its process environment. The daemon
runs under systemd and does not inherit your interactive shell — the key
has to be set in the unit itself. Use a drop-in:

```bash
sudo systemctl edit reachy-mini-daemon
# In the editor that opens, add:
# [Service]
# Environment=OPENAI_API_KEY=sk-...
sudo systemctl restart reachy-mini-daemon
```

Verify the daemon sees it:
```bash
sudo systemctl show reachy-mini-daemon -p Environment | grep OPENAI_API_KEY
```

## Running

Pick one:
- **Dashboard** — open `http://reachy-mini.local:8000/`, find *reachy-chat*
  under installed apps, click **Start**.
- **REST API** —
  ```bash
  curl -X POST http://reachy-mini.local:8000/api/apps/start-app/reachy-chat
  curl -X POST http://reachy-mini.local:8000/api/apps/stop-current-app
  ```
- **Direct Python** (fast iteration; bypasses the daemon's app manager but
  still uses the running daemon for hardware) —
  ```bash
  /venvs/apps_venv/bin/python -m reachy_chat.main
  ```

Tail logs while testing:
```bash
sudo journalctl -u reachy-mini-daemon -f | grep -v "uvicorn\|GET \|POST "
```

## Development workflow

Edit channel: **VS Code Remote-SSH** to `pollen@reachy-mini.local`, opening
`/home/pollen/reachy-chat/`. Source of truth is this GitHub repo; commits
and pushes happen from whichever side made the edits.

Inner loop:
1. Edit a `.py` in VS Code (Remote-SSH pane).
2. In the dashboard, click **Stop** on `reachy-chat`, then **Start**.
   (Editable install means no reinstall, but Python still has to re-import.)
3. Watch `journalctl` for the new run.

When metadata changes (entry points, deps in `pyproject.toml`):
```bash
/venvs/apps_venv/bin/pip install -e .
```

Commit and push from the Remote-SSH terminal as usual. Pull on the Windows
clone (`c:\git\github\reachy-chat`) when you want a local mirror.

## Tuning

Wake-word constants at the top of [`reachy_chat/main.py`](reachy_chat/main.py):

| Constant | Default | What it controls |
|---|---|---|
| `WAKE_WORD` | `"hey_jarvis"` | Which openWakeWord model to listen for. Other built-ins: `alexa`, `hey_mycroft`, `hey_rhasspy`, `weather`, `timer`. |
| `WAKE_WORD_THRESHOLD` | `0.5` | Detection score cutoff in [0, 1]. Raise if you get false triggers; lower if it misses. |
| `GREETING` | `"hello"` | Fallback espeak-ng phrase. Unused on the happy path; kept for offline diagnostics. |

Realtime constants at the top of [`reachy_chat/realtime.py`](reachy_chat/realtime.py):

| Constant | Default | What it controls |
|---|---|---|
| `REALTIME_MODEL` | `"gpt-realtime"` | OpenAI Realtime model name. |
| `REALTIME_VOICE` | `"ballad"` | Output voice — also `marin`, `cedar`, etc. |
| `MAX_TURN_S` | `30.0` | Hard cap on a single conversational turn. |
| `WAVE_AMPLITUDE_DEG` | `15.0` | Antenna sweep amplitude during the assistant's reply. |
| `WAVE_FREQ_HZ` | `0.8` | Antenna sweep frequency. |
| `RECORDED_MOVE_GOTO_DURATION_S` | `0.5` | Smoothing into the first pose of an emotion or dance clip. |
| `WEB_SEARCH_MODEL` | `"gpt-5-mini"` | Model used for the Responses API call backing `web_search`. |
| `WEB_SEARCH_TIMEOUT_S` | `15.0` | Max time to wait for a search response before erroring back to the model. |
| `ANNOUNCE_MAX_S` | `15.0` | Cap on the timer-fired announcement realtime session. |

### Realtime tools

The realtime model is given a fixed set of function tools at session
start. Tools whose backing resource is unavailable (no emotions library,
etc.) are omitted automatically, so a partial environment still works.

| Tool | What it does |
|---|---|
| `play_emotion(name)` | Plays one short emotion clip from the emotions library. Async — model keeps speaking. |
| `play_dance(name)` | Plays one longer dance clip from the dances library. Async. |
| `set_volume(level)` | Sets output volume 0–100. Affects the assistant's voice, the chime, and timer announcements. |
| `mute()` / `unmute()` | Silences / restores all output without losing the volume level. |
| `who_called_me()` | Reads the SDK's direction-of-arrival, turns the head toward the speaker, returns the angle. |
| `web_search(query)` | Bridges to OpenAI's hosted `web_search` tool via the Responses API (uses the same `OPENAI_API_KEY`). The only synchronous tool — the realtime loop blocks up to `WEB_SEARCH_TIMEOUT_S` for the result. |
| `set_timer(seconds, label)` | Registers a countdown. When it fires, plays the chime and opens a brief realtime session to announce the label. |

Implementation notes:

- All tool handlers live in `TOOL_HANDLERS` in `realtime.py`. Each tool is
  a `_<name>_schema()` builder + `_tool_<name>(reachy_mini, args, ctx)`
  handler — adding a tool is one place.
- A device-wide `_realtime_session_lock` ensures only one realtime
  session runs at a time. A timer firing while the user is mid-turn waits
  for the turn to end (up to 30 s) before announcing.
- The antenna wave acquires a `motion_lock` per `set_target` tick, while
  emotion / dance clips hold it for the whole `play_move` — so the wave
  doesn't fight body animations.
- `web_search` errors (timeout, no network, etc.) are returned to the
  model as a structured error so it can apologise rather than crash.

### System prompt

The `instructions` sent on `session.update` are composed from markdown
files in [`prompts/`](prompts/) at the repo root. Every `*.md` file is
read in lexicographic order and concatenated with blank lines between
fragments — so split your prompt into themed pieces (`role.md`,
`personality.md`, `boundaries.md`, …) or leave it as one file, your call.

- Sort with numeric prefixes (`00-`, `10-`, `20-`) when order matters.
- Disable a fragment without deleting it by prefixing the filename with
  `_` (e.g. `_holiday-mode.md`). Hidden dotfiles are also skipped.
- Files are reloaded on every wake-word, so edit a `.md`, say "hey
  jarvis" again, and the new prompt is in effect — no app restart.
- If `prompts/` is missing or empty, a short built-in default kicks in
  (`_DEFAULT_INSTRUCTIONS` in `realtime.py`).
