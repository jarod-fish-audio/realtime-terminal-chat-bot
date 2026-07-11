# Terminal Voice Agent (Fish Audio + OpenAI)

<div align="center">
  <video src="assets/demo.mp4" autoplay loop muted playsinline controls width="800"></video>
</div>

Talk to an LLM in realtime from your terminal:

```
mic (push-to-talk) -> Fish ASR -> OpenAI (streaming) -> Fish realtime WebSocket TTS -> speakers
```

The LLM's tokens are streamed straight into Fish's `wss://api.fish.audio/v1/tts/live`
socket as they're generated, so speech starts playing before the reply is finished.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (see `pyproject.toml`):

```
git clone https://github.com/jarod-fish-audio/realtime-terminal-chat-bot.git
cd realtime-terminal-chat-bot
uv sync
uv run voice_agent.py
```

On first launch a setup wizard walks you through everything:

- **Fish Audio API key** — from [fish.audio](https://fish.audio) → API keys
- **OpenAI API key**
- **Microphone** — pick from a list, or Enter for the system default. The wizard
  offers a 2-second mic test so silent virtual devices (VB-Cable etc.) are caught
  up front instead of making the ASR hallucinate mid-conversation.
- **Speakers** — Enter for the system default.
- **Fish voice** — optional voice model `reference_id`; Enter uses Fish's default voice.
- **Models** — OpenAI chat model (default `gpt-4.1-nano`) and Fish TTS model
  (default `s2.1-pro`; also `s2-pro` / `s1`).

Answers — API keys included — are saved to a gitignored `config.json`.
Rerun the wizard anytime:

```
uv run voice_agent.py --setup
```

You can also skip the wizard by copying `config.example.json` to `config.json`
and filling it in by hand. `FISH_API_KEY` / `OPENAI_API_KEY` environment
variables override the file if set.

## Run

```
uv run voice_agent.py
```

- Press **Enter** to start talking, **Enter** again to stop.
- Your transcript and the agent's reply are printed while the reply plays.
- Conversation history is kept for the whole session. **Ctrl+C** to quit.

## Troubleshooting

- Each turn's mic audio is saved to `last_recording.wav` — play it back to check
  what the ASR actually received.
- Recordings quieter than an RMS of ~100 (int16) are skipped as silence instead
  of being sent to ASR.
- If your saved mic/speaker disappears (unplugged USB device etc.), the agent
  falls back to the system default and tells you — rerun `--setup` to repick.
- List every audio device: `uv run python -m sounddevice`.
