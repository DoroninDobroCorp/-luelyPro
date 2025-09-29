from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
import websockets
from websockets.server import WebSocketServerProtocol

from live_recognizer import (
    LiveVoiceVerifier,
    VoiceProfile,
    SAMPLE_RATE,
    FRAME_SIZE,
    setup_logging,
)

# -------- Static HTTP Server --------
class _StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve from project root
        super().__init__(*args, directory=str(Path(__file__).parent.resolve()), **kwargs)


def start_http_server(host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), _StaticHandler)
    t = threading.Thread(target=httpd.serve_forever, name="http-server", daemon=True)
    t.start()
    logger.info(f"HTTP server on http://{host}:{port}")
    return httpd


# -------- WebSocket + Audio Pipeline --------
class AudioBridge:
    def __init__(self, ws: WebSocketServerProtocol, loop: asyncio.AbstractEventLoop):
        self.ws = ws
        self.loop = loop
        self.frame_queue: "queue.Queue[np.ndarray]" = __import__("queue").Queue(maxsize=64)
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.verifier: Optional[LiveVoiceVerifier] = None
        self.profile: Optional[VoiceProfile] = None

    def _tts_sink(self, wav_bytes: bytes, sample_rate: int) -> None:
        # Отправка бинарного WAV в браузер из потока сегмент-воркера
        fut = asyncio.run_coroutine_threadsafe(self.ws.send(wav_bytes), self.loop)
        try:
            fut.result(timeout=2.0)
        except Exception as e:
            logger.debug(f"WS send failed: {e}")

    def start_pipeline(self) -> None:
        # Конфиг по умолчанию аналогичен main.py оптимизированному пути
        asr_model = os.getenv("ASR_MODEL", "tiny")
        llm_enable = bool(os.getenv("GEMINI_API_KEY"))
        theses_path = Path("theses.txt") if theses_path_exists() else None

        self.verifier = LiveVoiceVerifier(
            threshold=0.75,
            vad_backend=os.getenv("VAD_BACKEND", "webrtc"),
            vad_aggressiveness=int(os.getenv("VAD_AGGR", "2")),
            min_consec_speech_frames=int(os.getenv("MIN_CONSEC", "3")),
            flatness_reject_threshold=float(os.getenv("FLATNESS_TH", "0.65")),
            asr_enable=True,
            asr_model_size=asr_model,
            asr_language=os.getenv("ASR_LANG", "ru"),
            asr_device=None,
            asr_compute_type=None,
            llm_enable=llm_enable,
            theses_path=theses_path,
            thesis_match_threshold=0.5,
            thesis_semantic_threshold=0.50,
            thesis_semantic_model=None,
            thesis_semantic_enable=False,
            thesis_gemini_enable=False,
            thesis_gemini_min_conf=0.50,
            thesis_autogen_enable=True,
            thesis_autogen_batch=3,
        )
        # Направляем TTS наружу, в браузер
        self.verifier.set_audio_sink(self._tts_sink)

        # Профиль
        prof = Path(os.getenv("PROFILE", "profiles/my_profile.npz"))
        prof.parent.mkdir(parents=True, exist_ok=True)
        self.profile = VoiceProfile.load(prof)
        if self.profile is None:
            # Фоллбэк: пустой профиль (все сегменты будут считаться "чужими")
            logger.warning(f"Профиль не найден: {prof}. Будем считать все голоса 'чужими'.")
            self.profile = VoiceProfile(embedding=np.zeros((192,), dtype=np.float32))

        # Таймер автостопа
        run_seconds_env = os.getenv("ASSISTANT_RUN_SECONDS") or os.getenv("RUN_SECONDS")
        try:
            run_seconds = float(run_seconds_env) if run_seconds_env else 0.0
        except Exception:
            run_seconds = 0.0

        # Стартуем пайплайн в отдельном потоке
        def _worker():
            assert self.verifier is not None and self.profile is not None
            self.verifier.live_verify_stream(
                profile=self.profile,
                frame_queue=self.frame_queue,
                min_segment_ms=int(os.getenv("MIN_SEGMENT_MS", "400")),
                max_silence_ms=int(os.getenv("MAX_SILENCE_MS", "300")),
                pre_roll_ms=int(os.getenv("PRE_ROLL_MS", "100")),
                run_seconds=run_seconds,
                stop_event=self.stop_event,
            )

        self.worker = threading.Thread(target=_worker, name="ws-pipeline", daemon=True)
        self.worker.start()

    def stop_pipeline(self) -> None:
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=2.0)
            self.worker = None


def theses_path_exists() -> bool:
    p = Path("theses.txt")
    return p.exists() and p.is_file() and p.stat().st_size > 0


async def ws_handler(websocket: WebSocketServerProtocol):
    loop = asyncio.get_running_loop()
    bridge = AudioBridge(websocket, loop)
    bridge.start_pipeline()
    logger.info("WebSocket client connected")

    try:
        async for message in websocket:
            if isinstance(message, (bytes, bytearray)):
                # Ожидаем сырые PCM16 LE mono @16k
                try:
                    arr = np.frombuffer(message, dtype=np.int16)
                    if arr.size == 0:
                        continue
                    # в float32 [-1,1]
                    samples = (arr.astype(np.float32) / 32768.0).reshape(-1)
                    try:
                        bridge.frame_queue.put_nowait(samples)
                    except __import__("queue").Full:
                        # Сбрасываем старые кадры при переполнении
                        _ = bridge.frame_queue.get_nowait()
                        bridge.frame_queue.put_nowait(samples)
                except Exception as e:
                    logger.debug(f"Failed to parse binary audio frame: {e}")
            else:
                # Текстовые управляющие сообщения (необязательно)
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                except Exception:
                    pass
    except websockets.ConnectionClosed:
        logger.info("WebSocket client disconnected")
    finally:
        bridge.stop_pipeline()


async def main_async() -> None:
    setup_logging()
    host = os.getenv("HOST", "127.0.0.1")
    ws_port = int(os.getenv("WS_PORT", "8765"))
    http_port = int(os.getenv("HTTP_PORT", "8000"))

    # HTTP static server for index.html / index.js / index.css
    httpd = start_http_server(host, http_port)

    stop_after_env = os.getenv("ASSISTANT_RUN_SECONDS") or os.getenv("RUN_SECONDS")
    stop_after = float(stop_after_env) if stop_after_env else 0.0

    async with websockets.serve(ws_handler, host, ws_port, max_size=2**22):  # ~4MB
        logger.info(f"WebSocket server on ws://{host}:{ws_port}")
        if stop_after > 0:
            logger.info(f"Auto-shutdown in {stop_after:.1f} seconds")
            await asyncio.sleep(stop_after)
        else:
            await asyncio.Future()  # run forever

    # Shutdown HTTP
    try:
        httpd.shutdown()
    except Exception:
        pass


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Interrupted")


if __name__ == "__main__":
    main()
