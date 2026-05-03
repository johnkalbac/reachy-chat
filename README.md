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
(this revision): wake on the keyword "Reachy" and reply with a local
text-to-speech "hello". No cloud calls yet — that comes next.

## Resources

- [SDK Documentation](https://huggingface.co/docs/reachy_mini)
- [Simulation Setup](https://huggingface.co/docs/reachy_mini/platforms/simulation/get_started)
- [Python SDK Reference](https://huggingface.co/docs/reachy_mini/SDK/python-sdk)
- [Building & Publishing Apps](https://huggingface.co/docs/reachy_mini/SDK/apps)
- [Wireless Development Workflow](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/development_workflow)
- [Examples (upstream)](https://github.com/pollen-robotics/reachy_mini/tree/main/examples)

## Architecture

The app is a Reachy Mini Python app — a package with a
`[project.entry-points."reachy_mini_apps"]` entry in `pyproject.toml`. The
daemon (`reachy-mini-daemon`, systemd service on the robot, dashboard at
`http://reachy-mini.local:8000`) discovers it, launches it as a subprocess
(`python -u -m reachy_chat.main`), hands it a connected `ReachyMini` instance
and a `stop_event`, and sends `SIGINT` to stop. Only one Reachy Mini app
runs at a time.

`reachy_chat.main:ReachyChatApp.run()` opens the SDK audio streams, feeds
16 kHz mono PCM frames into Porcupine, and on detection renders the reply
through `espeak-ng` and pushes the resulting samples back to the speaker.

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

### 3. Get a Picovoice access key and a "Reachy" wake-word file
1. Sign up (free, personal use): https://console.picovoice.ai/
2. Copy the **AccessKey** from the console.
3. In the console, open **Porcupine → Train Wake Word**, type `Reachy`,
   pick language **English**, pick platform **Raspberry Pi**, and download
   the resulting `.ppn` file.
4. Copy it to the robot:
   ```bash
   mkdir -p ~/.config/reachy_chat
   scp ./Reachy_en_raspberry-pi_*.ppn pollen@reachy-mini.local:~/.config/reachy_chat/wake_word.ppn
   ```
5. Persist the access key for the daemon to inherit. The daemon runs as a
   systemd unit, so add it to a drop-in:
   ```bash
   sudo systemctl edit reachy-mini-daemon
   ```
   Add:
   ```ini
   [Service]
   Environment="PICOVOICE_ACCESS_KEY=YOUR_KEY_HERE"
   ```
   Then:
   ```bash
   sudo systemctl restart reachy-mini-daemon
   ```

### 4. Clone this repo and install in editable mode
```bash
cd ~ && git clone https://github.com/<your-user>/reachy-chat.git
cd reachy-chat
/venvs/apps_venv/bin/pip install -e .
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

## Configuration

Read at startup, both via env vars inherited from the daemon:

| Variable | Default | Notes |
|---|---|---|
| `PICOVOICE_ACCESS_KEY` | *(required)* | From the Picovoice console. |
| `WAKE_WORD_PATH` | `~/.config/reachy_chat/wake_word.ppn` | Custom Porcupine `.ppn`. |
