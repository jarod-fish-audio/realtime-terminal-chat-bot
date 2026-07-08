"""First-run setup wizard and config loading for the voice agent.

All settings, API keys included, live in `config.json` (gitignored; see
`config.example.json` for the shape). The wizard runs automatically when
the config is missing or incomplete, or on demand with `--setup`.
`FISH_API_KEY` / `OPENAI_API_KEY` environment variables override the file.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
LEGACY_ENV_PATH = ROOT / ".env"  # pre-config.json versions kept keys here

DEFAULT_LLM_MODEL = "gpt-4.1-nano"
DEFAULT_TTS_MODEL = "s2.1-pro"
TTS_MODELS = ("s2.1-pro", "s2-pro", "s1")

MIC_TEST_SECONDS = 2
MIC_TEST_SAMPLE_RATE = 16_000
SILENCE_RMS = 100  # int16 RMS below this counts as "no speech"


@dataclass
class Settings:
    fish_api_key: str
    openai_api_key: str
    fish_voice_id: str
    llm_model: str
    tts_model: str
    mic_device: dict | None  # {"index": int, "name": str} or None = system default
    output_device: dict | None


# --- config IO --------------------------------------------------------------


def _read_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            print(f"(couldn't parse {CONFIG_PATH.name}, starting fresh)")
    return {}


def _write_config(cfg: Settings):
    CONFIG_PATH.write_text(json.dumps(cfg.__dict__, indent=2) + "\n")


def _read_legacy_env() -> dict[str, str]:
    """Parse an old-style .env so its keys can seed the config."""
    env: dict[str, str] = {}
    if LEGACY_ENV_PATH.exists():
        for line in LEGACY_ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _key_defaults(cfg: dict) -> tuple[str, str, str]:
    """Best known (fish key, openai key, voice id): env vars, then
    config.json, then a legacy .env."""
    legacy = _read_legacy_env()
    fish = (
        os.environ.get("FISH_API_KEY")
        or cfg.get("fish_api_key")
        or legacy.get("FISH_API_KEY", "")
    )
    openai = (
        os.environ.get("OPENAI_API_KEY")
        or cfg.get("openai_api_key")
        or legacy.get("OPENAI_API_KEY")
        or legacy.get("OPEN_AI_KEY", "")
    )
    voice = cfg.get("fish_voice_id") or legacy.get("FISH_VOICE_ID", "")
    return fish, openai, voice


# --- prompts ---------------------------------------------------------------


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{label}{suffix}: ").strip()
    return answer or default


def _prompt_secret(label: str, existing: str) -> str:
    """Ask for an API key; Enter keeps the existing one (shown masked)."""
    if existing:
        masked = existing[:4] + "..." + existing[-4:] if len(existing) > 8 else "***"
        answer = input(f"{label} [keep {masked}]: ").strip()
        return answer or existing
    while True:
        answer = input(f"{label}: ").strip()
        if answer:
            return answer
        print("  (required - paste the key and press Enter)")


def _list_devices(kind: str) -> list[tuple[int, str]]:
    """Devices of the default host API with channels in the given direction.

    Filtering to one host API avoids Windows listing every device 3-4 times
    (MME/WASAPI/DirectSound); the default API (MME on Windows) also resamples
    to whatever rate we ask for, which WASAPI won't.
    """
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    try:
        default_hostapi = sd.query_devices(kind=kind)["hostapi"]
    except (sd.PortAudioError, TypeError):
        default_hostapi = 0
    return [
        (i, d["name"])
        for i, d in enumerate(sd.query_devices())
        if d[key] > 0 and d["hostapi"] == default_hostapi
    ]


def _choose_device(kind: str, label: str) -> dict | None:
    devices = _list_devices(kind)
    if not devices:
        print(f"(no {label} devices found - using system default)")
        return None
    print(f"\n{label.capitalize()} devices:")
    for n, (_, name) in enumerate(devices, start=1):
        print(f"  [{n}] {name}")
    while True:
        answer = input(f"Pick a {label} number (Enter = system default): ").strip()
        if not answer:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(devices):
            index, name = devices[int(answer) - 1]
            return {"index": index, "name": name}
        print(f"  (enter a number between 1 and {len(devices)}, or Enter to skip)")


def _mic_test(mic: dict | None):
    """Record a couple of seconds and report the level, so silent virtual
    devices are caught here instead of mid-conversation."""
    while True:
        answer = input("\nTest the mic now? [Y/n] ").strip().lower()
        if answer in ("n", "no"):
            return
        print(f"Recording {MIC_TEST_SECONDS}s - say something...")
        try:
            frames = sd.rec(
                MIC_TEST_SECONDS * MIC_TEST_SAMPLE_RATE,
                samplerate=MIC_TEST_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                device=mic["index"] if mic else None,
            )
            sd.wait()
        except sd.PortAudioError as e:
            print(f"Recording failed: {e}")
            return
        rms = float(np.sqrt(np.mean(frames.astype(np.float64) ** 2)))
        if rms < SILENCE_RMS:
            print(
                f"Only silence recorded (rms={rms:.0f}). This mic may be a "
                "virtual device or muted - consider picking a different one "
                "or checking its gain, then test again."
            )
        else:
            print(f"Mic level looks good (rms={rms:.0f}).")
            return


# --- wizard ----------------------------------------------------------------


def run_wizard() -> Settings:
    cfg = _read_config()
    fish_default, openai_default, voice_default = _key_defaults(cfg)
    print("\n=== Voice agent setup ===")
    print("Answers are saved to config.json (gitignored).")
    print("Rerun anytime with: uv run voice_agent.py --setup\n")

    fish_key = _prompt_secret(
        "Fish Audio API key (https://fish.audio -> API keys)", fish_default
    )
    openai_key = _prompt_secret("OpenAI API key", openai_default)

    mic = _choose_device("input", "microphone")
    _mic_test(mic)
    speaker = _choose_device("output", "speaker")

    voice_id = _prompt(
        "\nFish voice model reference_id (Enter = Fish's default voice)",
        voice_default,
    )
    llm_model = _prompt("OpenAI chat model", cfg.get("llm_model", DEFAULT_LLM_MODEL))
    tts_model = _prompt(
        f"Fish TTS model ({'/'.join(TTS_MODELS)})",
        cfg.get("tts_model", DEFAULT_TTS_MODEL),
    )
    if tts_model not in TTS_MODELS:
        print(f"  (unknown model {tts_model!r} - keeping it, but expect errors if it's a typo)")

    settings = Settings(
        fish_api_key=fish_key,
        openai_api_key=openai_key,
        fish_voice_id=voice_id,
        llm_model=llm_model,
        tts_model=tts_model,
        mic_device=mic,
        output_device=speaker,
    )
    _write_config(settings)
    print(f"\nSaved {CONFIG_PATH.name}. You're set.\n")
    return settings


def ensure_setup(force: bool = False) -> Settings:
    """Load settings, running the wizard first if forced or incomplete."""
    cfg = _read_config()
    fish_key, openai_key, voice_id = _key_defaults(cfg)

    if force or not CONFIG_PATH.exists() or not fish_key or not openai_key:
        return run_wizard()

    return Settings(
        fish_api_key=fish_key,
        openai_api_key=openai_key,
        fish_voice_id=voice_id,
        llm_model=cfg.get("llm_model", DEFAULT_LLM_MODEL),
        tts_model=cfg.get("tts_model", DEFAULT_TTS_MODEL),
        mic_device=cfg.get("mic_device"),
        output_device=cfg.get("output_device"),
    )


def resolve_device(saved: dict | None, kind: str) -> int | None:
    """Turn a saved {"index", "name"} into a current device index.

    Device indices shift when hardware is plugged/unplugged, so trust the
    index only if the name still matches; otherwise search by name.
    """
    if not saved:
        return None  # system default
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = sd.query_devices()
    index, name = saved.get("index"), saved.get("name", "")
    if (
        isinstance(index, int)
        and 0 <= index < len(devices)
        and devices[index]["name"] == name
        and devices[index][key] > 0
    ):
        return index
    for i, d in enumerate(devices):
        if d["name"] == name and d[key] > 0:
            return i
    print(
        f"(saved {kind} device {name!r} not found - using system default; "
        "rerun with --setup to pick again)"
    )
    return None
