"""Realtime terminal voice agent.

Push-to-talk mic -> Fish Audio ASR -> OpenAI (streaming) -> Fish Audio
realtime WebSocket TTS -> speakers, in a conversation loop.

Usage: python voice_agent.py [--setup]
  First run walks you through setup (API keys, mic, speakers).
  [Enter] to start talking, [Enter] again to stop, Ctrl+C to quit.
"""

import asyncio
import io
import queue
import sys
import threading
import time
import wave

import httpx
import msgpack
import numpy as np
import sounddevice as sd
import websockets
from openai import AsyncOpenAI
from websockets.exceptions import ConnectionClosed

from setup_wizard import SILENCE_RMS, Settings, ensure_setup, resolve_device

FISH_ASR_URL = "https://api.fish.audio/v1/asr"
FISH_TTS_WS_URL = "wss://api.fish.audio/v1/tts/live"

MIC_SAMPLE_RATE = 16_000
TTS_SAMPLE_RATE = 44_100

DEBUG_WAV = "last_recording.wav"  # each turn's mic audio, for troubleshooting

SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Your replies are spoken aloud, so keep "
    "them short and conversational - a sentence or three of plain spoken text, "
    "no markdown, no lists."
)


def record_push_to_talk(device: int | None) -> bytes | None:
    """Record mic audio between two Enter presses; return WAV bytes."""
    input("\n[Enter] to talk (Ctrl+C to quit) ")
    frames: list[np.ndarray] = []

    def callback(indata, _frames, _time, _status):
        frames.append(indata.copy())

    with sd.InputStream(
        device=device,
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=callback,
    ):
        input("Recording... [Enter] to stop ")

    if not frames:
        return None
    audio = np.concatenate(frames).flatten()

    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < SILENCE_RMS:
        print(f"(only silence recorded, rms={rms:.0f} - check mic/gain)")
        return None

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(MIC_SAMPLE_RATE)
        w.writeframes(audio.tobytes())
    wav = buf.getvalue()
    with open(DEBUG_WAV, "wb") as f:
        f.write(wav)
    return wav


_http = httpx.Client(timeout=120)  # reused connection saves ~300ms TLS setup per turn


def transcribe(wav_bytes: bytes, fish_api_key: str) -> str:
    r = _http.post(
        FISH_ASR_URL,
        headers={"Authorization": f"Bearer {fish_api_key}"},
        files={"audio": ("speech.wav", wav_bytes, "audio/wav")},
    )
    r.raise_for_status()
    return r.json()["text"].strip()


class SpeakerPlayer:
    """Plays raw 16-bit mono PCM chunks from a queue on a background thread."""

    def __init__(self, sample_rate: int, device: int | None = None):
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._carry = b""  # PCM chunks may split mid-sample; hold odd bytes over
        self._thread = threading.Thread(
            target=self._run, args=(sample_rate, device), daemon=True
        )
        self._thread.start()

    def _run(self, sample_rate: int, device: int | None):
        with sd.RawOutputStream(
            samplerate=sample_rate, channels=1, dtype="int16", device=device
        ) as out:
            while True:
                chunk = self._queue.get()
                if chunk is None:
                    return
                out.write(chunk)

    def feed(self, data: bytes):
        data = self._carry + data
        cut = len(data) - (len(data) % 2)
        self._carry, data = data[cut:], data[:cut]
        if data:
            self._queue.put(data)

    def close(self):
        """Signal end of audio and wait for playback to drain."""
        self._queue.put(None)
        self._thread.join()


async def llm_deltas(stream, collected: list[str], timings: dict, t0: float):
    """Stream assistant text deltas, printing and collecting them as we go."""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            if "llm_first_token" not in timings:
                timings["llm_first_token"] = time.perf_counter() - t0
            collected.append(delta)
            print(delta, end="", flush=True)
            yield delta
    print()


async def speak_turn(
    history: list[dict],
    collected: list[str],
    cfg: Settings,
    output_device: int | None,
):
    """Stream one LLM reply into Fish's realtime TTS and play it as it arrives."""
    oai = AsyncOpenAI(api_key=cfg.openai_api_key)
    player = SpeakerPlayer(TTS_SAMPLE_RATE, output_device)
    t0 = time.perf_counter()
    timings: dict = {}

    request: dict = {
        "text": "",
        "format": "pcm",
        "sample_rate": TTS_SAMPLE_RATE,
        "latency": "low",
    }
    if cfg.fish_voice_id:
        request["reference_id"] = cfg.fish_voice_id

    headers = {"Authorization": f"Bearer {cfg.fish_api_key}", "model": cfg.tts_model}
    try:
        # Kick off the LLM request while the websocket handshake runs
        stream_task = asyncio.create_task(
            oai.chat.completions.create(
                model=cfg.llm_model, messages=history, stream=True
            )
        )
        async with websockets.connect(
            FISH_TTS_WS_URL, additional_headers=headers, max_size=None
        ) as ws:
            timings["ws_open"] = time.perf_counter() - t0
            await ws.send(
                msgpack.packb({"event": "start", "request": request}, use_bin_type=True)
            )

            async def sender():
                try:
                    async for delta in llm_deltas(
                        await stream_task, collected, timings, t0
                    ):
                        await ws.send(
                            msgpack.packb(
                                {"event": "text", "text": delta}, use_bin_type=True
                            )
                        )
                        if "first_text_sent" not in timings:
                            timings["first_text_sent"] = time.perf_counter() - t0
                    await ws.send(msgpack.packb({"event": "stop"}, use_bin_type=True))
                except ConnectionClosed:
                    pass  # server finished before the text stream drained

            send_task = asyncio.create_task(sender())
            try:
                async for raw in ws:
                    msg = msgpack.unpackb(raw, raw=False)
                    if msg["event"] == "audio":
                        if "first_audio" not in timings:
                            timings["first_audio"] = time.perf_counter() - t0
                        player.feed(msg["audio"])
                    elif msg["event"] == "finish":
                        if msg["reason"] == "error":
                            print("\n[TTS error - reply text above is still valid]")
                        break
            finally:
                send_task.cancel()
                try:
                    await send_task
                except (asyncio.CancelledError, ConnectionClosed):
                    pass
    finally:
        player.close()
        await oai.close()
    return timings


def main():
    cfg = ensure_setup(force="--setup" in sys.argv[1:])
    mic = resolve_device(cfg.mic_device, "input")
    speaker = resolve_device(cfg.output_device, "output")

    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    mic_name = sd.query_devices(mic, kind="input")["name"]
    print(f"Fish voice agent ready. Mic: {mic_name}")
    print("Push-to-talk: [Enter] starts, [Enter] stops. Reconfigure: --setup")

    while True:
        wav = record_push_to_talk(mic)
        if wav is None:
            continue

        t_asr = time.perf_counter()
        text = transcribe(wav, cfg.fish_api_key)
        asr_time = time.perf_counter() - t_asr
        if not text:
            print("(heard nothing, try again)")
            continue
        print(f"You: {text}")

        history.append({"role": "user", "content": text})
        collected: list[str] = []
        print("Agent: ", end="", flush=True)
        timings = asyncio.run(speak_turn(history, collected, cfg, speaker))
        history.append({"role": "assistant", "content": "".join(collected)})

        parts = [f"ASR: {asr_time:.2f}s"]
        if "llm_first_token" in timings:
            parts.append(f"LLM first token: {timings['llm_first_token']:.2f}s")
        if "first_audio" in timings and "first_text_sent" in timings:
            tts = timings["first_audio"] - timings["first_text_sent"]
            parts.append(f"TTS first audio: {tts:.2f}s after first text sent")
        print(f"[{' | '.join(parts)}]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
