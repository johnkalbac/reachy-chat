---
title: reachy-chat
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - reachy_mini_python_app
---

# reachy-chat

An AI-enabled chat assistant for the Reachy Mini Wireless. First milestone
(this revision): wake on a keyword and reply with a local text-to-speech
"hello". No cloud calls yet — that comes next.

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
cd ~ && git clone https://github.com/<your-user>/reachy-chat.git
cd reachy-chat
/venvs/apps_venv/bin/pip install -e .
```

### 4. Pre-download the openWakeWord models
```bash
/venvs/apps_venv/bin/python -c "import openwakeword.utils; openwakeword.utils.download_models()"
```

### 5. Validate the app metadata
```bash
/venvs/apps_venv/bin/reachy-mini-app-assistant check .
```

## Running

Pick one:
- **Dashboard** — open `http://reachy-mini.local:8000/`, find *reachy-chat*
  under installed apps, click **Start**.
- **REST API** —
  ```bash
  curl -X POST http://reachy-mini.local:8000/api/apps/start-app/reachy_chat
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

Constants at the top of [`reachy_chat/main.py`](reachy_chat/main.py):

| Constant | Default | What it controls |
|---|---|---|
| `WAKE_WORD` | `"hey_jarvis"` | Which openWakeWord model to listen for. Other built-ins: `alexa`, `hey_mycroft`, `hey_rhasspy`, `weather`, `timer`. |
| `WAKE_WORD_THRESHOLD` | `0.5` | Detection score cutoff in [0, 1]. Raise if you get false triggers; lower if it misses. |
| `GREETING` | `"hello"` | What espeak-ng speaks on detection. |
