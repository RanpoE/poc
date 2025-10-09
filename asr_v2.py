import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from typing import Dict, Optional, Set
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from starlette.websockets import WebSocketState

from intent_detector import IntentAnalyzer
from whisper_live_client import WhisperLiveBridge


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------
# In-memory room state
# --------------------
rooms: Dict[str, Set[WebSocket]] = defaultdict(set)
roles: Dict[WebSocket, str] = {}  # ws -> "customer" | "agent"
room_of: Dict[WebSocket, str] = {}  # ws -> room_id


try:
    intent_analyzer: Optional[IntentAnalyzer] = IntentAnalyzer()
except Exception as exc:
    intent_analyzer = None
    print(f"Intent analyzer disabled: {exc}")


# async def broadcast(room_id: str, sender: Optional[WebSocket], data: dict):
#     """Send JSON to all signaling sockets in a room except sender."""
#     if room_id not in rooms:
#         return
#     dead = []
#     for ws in list(rooms[room_id]):
#         if sender is not None and ws is sender:
#             continue
#     try:
#         await ws.send_json(data)
#     except Exception:
#         dead.append(ws)
#     for ws in dead:
#         rooms[room_id].discard(ws)
#         roles.pop(ws, None)
#         room_of.pop(ws, None)


async def broadcast(room_id: str, sender: Optional[WebSocket], data: dict):
    if room_id not in rooms:
        return
    dead = []
    for ws in list(rooms[room_id]):
        if sender is not None and ws is sender:
            continue
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        rooms[room_id].discard(ws)
        roles.pop(ws, None)
        room_of.pop(ws, None)


async def analyze_and_broadcast_intent(room_id: str, role: str, text: str) -> None:
    if not intent_analyzer:
        return
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, intent_analyzer.analyze, text)
    except Exception as exc:
        print(f"Intent analysis failed: {exc}")
        return

    if not isinstance(result, dict):
        return

    payload = {
        "type": "intent",
        "role": role,
        "text": text,
        "intent": result.get("intent"),
        "sentiment": result.get("sentiment"),
        "confidence": result.get("confidence"),
    }

    try:
        await broadcast(room_id, None, payload)
    except Exception as exc:
        print(f"Intent broadcast failed: {exc}")


@app.get("/")
def ok():
    return {"ok": True}


# --------------------
# Signaling WS: /ws
# --------------------
# First message must be: {"type":"join", "room":"<id>", "role":"customer|agent"}
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        hello = await ws.receive_json()
        if hello.get("type") != "join":
            await ws.close(code=1002)
            return
        room_id = str(hello.get("room", "")).strip()
        role = hello.get("role")
        if not room_id or role not in ("customer", "agent"):
            await ws.close(code=1002)
            return

        rooms[room_id].add(ws)
        roles[ws] = role
        room_of[ws] = room_id

        # notify others
        await broadcast(room_id, ws, {"type": "peer-joined", "role": role})

        # relay signaling + caption messages
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except Exception:
                continue
            t = data.get("type")
            if t in {"offer", "answer", "ice"}:
                await broadcast(room_id, ws, data)
            elif t == "caption":
                data.setdefault("role", roles.get(ws, "unknown"))
                await broadcast(room_id, ws, data)
            else:
                # ignore unknown types
                pass
    except WebSocketDisconnect:
        pass
    finally:
        r = room_of.pop(ws, None)
        if r and ws in rooms[r]:
            rooms[r].discard(ws)
            await broadcast(
                r, ws, {"type": "peer-left", "role": roles.get(ws, "unknown")}
            )
        roles.pop(ws, None)


ASR_BACKEND = os.getenv("ASR_BACKEND", "whisper").strip().lower()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    if minimum is not None and value < minimum:
        return minimum
    return value


SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
# Chunk audio more aggressively (defaults to 500ms) so captions land sooner; configurable via ASR_ACCUM_MS.
ACCUM_MS = _env_int("ASR_ACCUM_MS", 500, minimum=100)
ACCUM_BYTES_TARGET = SAMPLE_RATE * BYTES_PER_SAMPLE * ACCUM_MS // 1000
MIN_FLUSH_MS = _env_int("ASR_MIN_FLUSH_MS", max(ACCUM_MS // 2, 150), minimum=50)
IDLE_FLUSH_MS = _env_int("ASR_IDLE_FLUSH_MS", ACCUM_MS, minimum=MIN_FLUSH_MS)
MIN_FLUSH_BYTES = min(
    ACCUM_BYTES_TARGET, SAMPLE_RATE * BYTES_PER_SAMPLE * MIN_FLUSH_MS // 1000
)
IDLE_FLUSH_SECONDS = IDLE_FLUSH_MS / 1000.0


WHISPER_LIVE_HOST = os.getenv("WHISPER_LIVE_HOST", "localhost").strip() or "localhost"
WHISPER_LIVE_PORT = int(os.getenv("WHISPER_LIVE_PORT", "9090") or "9090")
WHISPER_LIVE_USE_WSS = _env_bool("WHISPER_LIVE_USE_WSS", False)
WHISPER_LIVE_MODEL = os.getenv("WHISPER_LIVE_MODEL", "small").strip() or "small"
WHISPER_LIVE_USE_VAD = _env_bool("WHISPER_LIVE_USE_VAD", False)
WHISPER_LIVE_TRANSLATE = _env_bool("WHISPER_LIVE_TRANSLATE", False)
WHISPER_LIVE_TARGET_LANGUAGE = os.getenv("WHISPER_LIVE_TARGET_LANGUAGE", "").strip()
WHISPER_LIVE_READY_TIMEOUT = float(
    os.getenv("WHISPER_LIVE_READY_TIMEOUT", "15.0") or "15.0"
)
WHISPER_LIVE_MAX_CLIENTS = int(os.getenv("WHISPER_LIVE_MAX_CLIENTS", "4") or "4")
WHISPER_LIVE_MAX_CONNECTION_TIME = int(
    os.getenv("WHISPER_LIVE_MAX_CONNECTION_TIME", "600") or "600"
)
WHISPER_LIVE_SEND_LAST_SEGMENTS = int(
    os.getenv("WHISPER_LIVE_SEND_LAST_SEGMENTS", "10") or "10"
)
WHISPER_LIVE_NO_SPEECH_THRESH = float(
    os.getenv("WHISPER_LIVE_NO_SPEECH_THRESH", "0.45") or "0.45"
)
WHISPER_LIVE_CLIP_AUDIO = _env_bool("WHISPER_LIVE_CLIP_AUDIO", False)
WHISPER_LIVE_SAME_OUTPUT_THRESHOLD = int(
    os.getenv("WHISPER_LIVE_SAME_OUTPUT_THRESHOLD", "10") or "10"
)

model: Optional[WhisperModel] = None
qwen_transcriber = None

if ASR_BACKEND == "whisper":
    model = WhisperModel("base", device="cpu")
    print("ASR backend: Whisper")
elif ASR_BACKEND == "qwen":
    try:
        from qwen_asr import QwenConfig, QwenTranscriber

        qwen_model_path = os.getenv("QWEN_MODEL_PATH")
        if qwen_model_path:
            cfg = QwenConfig(model_id=qwen_model_path)
        else:
            cfg = QwenConfig()
        dtype = os.getenv("QWEN_TORCH_DTYPE")
        if dtype:
            cfg.torch_dtype = dtype
        device_override = os.getenv("QWEN_DEVICE")
        if device_override:
            cfg.device = device_override
        qwen_transcriber = QwenTranscriber(cfg)
        print("ASR backend: Qwen-3-ASR")
    except Exception as exc:  # pragma: no cover - runtime path
        qwen_transcriber = None
        ASR_BACKEND = "whisper"
        print(f"Qwen backend unavailable, falling back to Whisper: {exc}")
        model = WhisperModel("base", device="cpu")
        print("ASR backend: Whisper")
elif ASR_BACKEND == "whisper_live":
    print("ASR backend: whisper-live remote")
else:
    print("Unknown ASR backend requested; defaulting to Whisper")
    ASR_BACKEND = "whisper"
    model = WhisperModel("base", device="cpu")


@dataclass
class TranscriptionJob:
    audio: np.ndarray
    language: str
    future: asyncio.Future


AUDIO_QUEUE_MAXSIZE = _env_int("ASR_QUEUE_MAXSIZE", 8, minimum=1)
TRANSCRIPTION_WORKERS = _env_int("ASR_WORKERS", 1, minimum=1)
audio_queue: "asyncio.Queue[TranscriptionJob]" = asyncio.Queue(
    maxsize=AUDIO_QUEUE_MAXSIZE
)
worker_tasks: list[asyncio.Task] = []


def _transcribe_blocking(audio: np.ndarray, language: str) -> str:
    if ASR_BACKEND == "whisper_live":
        raise RuntimeError("whisper-live backend requires streaming mode")

    if qwen_transcriber:
        return qwen_transcriber.transcribe(audio, language=language)

    if model is None:
        raise RuntimeError("Whisper model is unavailable")

    segments, _info = model.transcribe(
        audio,
        language=language,
        vad_filter=True,
        beam_size=1,
        condition_on_previous_text=False,
    )
    return "".join(s.text for s in segments).strip()


async def transcription_worker() -> None:
    loop = asyncio.get_running_loop()
    while True:
        job = await audio_queue.get()
        if job.future.cancelled():
            audio_queue.task_done()
            continue
        try:
            text = await loop.run_in_executor(
                None, _transcribe_blocking, job.audio, job.language
            )
            if not job.future.cancelled():
                job.future.set_result(text)
        except Exception as exc:
            if not job.future.cancelled():
                job.future.set_exception(exc)
        finally:
            audio_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    for _ in range(TRANSCRIPTION_WORKERS):
        worker_tasks.append(asyncio.create_task(transcription_worker()))
    try:
        yield
    finally:
        for task in worker_tasks:
            task.cancel()
        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)
        worker_tasks.clear()


app.router.lifespan_context = lifespan


@app.websocket("/asr")
async def asr_ws(ws: WebSocket):
    await ws.accept()
    room_id: Optional[str] = None
    role: Optional[str] = None
    buffer = bytearray()
    language = "en"

    try:
        init_msg = await ws.receive_text()
        try:
            init = json.loads(init_msg)
        except Exception:
            await ws.close(code=1002)
            return
        if init.get("type") != "init":
            await ws.close(code=1002)
            return
        room_id = str(init.get("room", "")).strip()
        role = init.get("role")
        language = init.get("lang", "en")
        if not room_id or role not in ("customer", "agent"):
            await ws.close(code=1002)
            return

        loop = asyncio.get_running_loop()

        if ASR_BACKEND == "whisper_live":
            audio_buffer = bytearray()
            transcript_queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

            async def forward_remote_transcripts() -> None:
                last_sent: Optional[str] = None
                try:
                    while True:
                        text = await transcript_queue.get()
                        if text is None:
                            transcript_queue.task_done()
                            break
                        if text == last_sent:
                            transcript_queue.task_done()
                            continue
                        last_sent = text
                        try:
                            await broadcast(
                                room_id,
                                None,
                                {"type": "caption", "text": text, "role": role},
                            )
                            if role == "customer":
                                asyncio.create_task(
                                    analyze_and_broadcast_intent(room_id, role, text)
                                )
                        finally:
                            transcript_queue.task_done()
                finally:
                    while True:
                        try:
                            transcript_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        else:
                            transcript_queue.task_done()

            forward_task = asyncio.create_task(forward_remote_transcripts())

            def handle_transcript(text: str) -> None:
                cleaned = text.strip()
                if not cleaned:
                    return
                try:
                    asyncio.run_coroutine_threadsafe(
                        transcript_queue.put(cleaned), loop
                    )
                except RuntimeError:
                    pass

            extra_payload: Dict[str, object] = {}
            needs_translation = WHISPER_LIVE_TRANSLATE or bool(
                WHISPER_LIVE_TARGET_LANGUAGE
            )
            if WHISPER_LIVE_TARGET_LANGUAGE:
                extra_payload["target_language"] = WHISPER_LIVE_TARGET_LANGUAGE
                extra_payload.setdefault("enable_translation", True)

            client = WhisperLiveBridge(
                WHISPER_LIVE_HOST,
                WHISPER_LIVE_PORT,
                language=language,
                translate=needs_translation,
                model=WHISPER_LIVE_MODEL,
                use_vad=WHISPER_LIVE_USE_VAD,
                use_wss=WHISPER_LIVE_USE_WSS,
                max_clients=WHISPER_LIVE_MAX_CLIENTS,
                max_connection_time=WHISPER_LIVE_MAX_CONNECTION_TIME,
                send_last_n_segments=WHISPER_LIVE_SEND_LAST_SEGMENTS,
                no_speech_thresh=WHISPER_LIVE_NO_SPEECH_THRESH,
                clip_audio=WHISPER_LIVE_CLIP_AUDIO,
                same_output_threshold=WHISPER_LIVE_SAME_OUTPUT_THRESHOLD,
                extra_init_payload=extra_payload,
                transcription_callback=handle_transcript,
            )

            try:
                await asyncio.to_thread(
                    client.wait_until_ready, WHISPER_LIVE_READY_TIMEOUT
                )
            except Exception as exc:
                try:
                    await ws.send_json({"type": "error", "message": str(exc)})
                except Exception:
                    pass
                await asyncio.to_thread(client.close)
                if not forward_task.done():
                    await transcript_queue.put(None)
                await forward_task
                return

            try:

                async def send_remote_chunk(raw: bytes) -> None:
                    pcm16 = np.frombuffer(raw, dtype=np.int16)
                    if pcm16.size == 0:
                        return
                    audio = pcm16.astype(np.float32) / 32768.0
                    try:
                        await asyncio.to_thread(client.send_audio, audio)
                    except Exception as exc:
                        try:
                            await ws.send_json({"type": "error", "message": str(exc)})
                        except Exception:
                            pass
                        raise

                while True:
                    try:
                        pkt = await asyncio.wait_for(
                            ws.receive_bytes(), timeout=IDLE_FLUSH_SECONDS
                        )
                    except asyncio.TimeoutError:
                        if len(audio_buffer) >= MIN_FLUSH_BYTES:
                            chunk = bytes(audio_buffer)
                            audio_buffer.clear()
                            await send_remote_chunk(chunk)
                        continue

                    audio_buffer.extend(pkt)

                    while len(audio_buffer) >= ACCUM_BYTES_TARGET:
                        chunk = audio_buffer[:ACCUM_BYTES_TARGET]
                        del audio_buffer[:ACCUM_BYTES_TARGET]
                        await send_remote_chunk(chunk)
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                if len(audio_buffer) >= MIN_FLUSH_BYTES:
                    chunk = bytes(audio_buffer)
                    audio_buffer.clear()
                    try:
                        await send_remote_chunk(chunk)
                    except Exception:
                        pass
                await asyncio.to_thread(client.close)
                if not forward_task.done():
                    await transcript_queue.put(None)
                await forward_task
            return

        async def handle_transcription_result(fut: asyncio.Future) -> None:
            try:
                text = await fut
            except Exception as exc:
                try:
                    await ws.send_json({"type": "error", "message": str(exc)})
                except Exception:
                    pass
                return
            if not text:
                return
            try:
                await broadcast(
                    room_id, None, {"type": "caption", "text": text, "role": role}
                )
                if role == "customer":
                    asyncio.create_task(
                        analyze_and_broadcast_intent(room_id, role, text)
                    )
            except Exception as exc:
                try:
                    await ws.send_json({"type": "error", "message": str(exc)})
                except Exception:
                    pass

        async def submit_local_chunk(raw: bytes) -> None:
            pcm16 = np.frombuffer(raw, dtype=np.int16)
            if pcm16.size == 0:
                return
            audio = pcm16.astype(np.float32) / 32768.0
            future: asyncio.Future = loop.create_future()
            job = TranscriptionJob(audio=audio, language=language, future=future)
            await audio_queue.put(job)
            asyncio.create_task(handle_transcription_result(future))

        while True:
            try:
                pkt = await asyncio.wait_for(
                    ws.receive_bytes(), timeout=IDLE_FLUSH_SECONDS
                )
            except asyncio.TimeoutError:
                if len(buffer) >= MIN_FLUSH_BYTES:
                    chunk = bytes(buffer)
                    buffer.clear()
                    await submit_local_chunk(chunk)
                continue

            buffer.extend(pkt)

            # transcribe roughly every chunk window
            while len(buffer) >= ACCUM_BYTES_TARGET:
                chunk = buffer[:ACCUM_BYTES_TARGET]
                del buffer[:ACCUM_BYTES_TARGET]
                await submit_local_chunk(chunk)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if (
            "submit_local_chunk" in locals()
            and len(buffer) >= MIN_FLUSH_BYTES
            and ASR_BACKEND != "whisper_live"
        ):
            chunk = bytes(buffer)
            buffer.clear()
            try:
                await submit_local_chunk(chunk)
            except Exception:
                pass
        if ws.application_state != WebSocketState.DISCONNECTED:
            try:
                await ws.close()
            except RuntimeError:
                # Connection already closed by the server stack; nothing to do.
                pass
