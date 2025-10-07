import asyncio
from collections import defaultdict
from typing import Dict, Set, Optional


from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import numpy as np
import json

from intent_detector import IntentAnalyzer


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


SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
ACCUM_MS = 1000
ACCUM_BYTES_TARGET = SAMPLE_RATE * BYTES_PER_SAMPLE * ACCUM_MS // 1000


# Load Whisper once (first run downloads the model to cache)
# "tiny" is smaller/faster; "base" is a good balance on CPU
# model_size = "large-v2"
# device = "gpu"

model = WhisperModel("base", device="cpu")
# model = WhisperModel(model_size, device="cpu")


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

        while True:
            pkt = await ws.receive_bytes()
            buffer.extend(pkt)

            # transcribe roughly every second
            while len(buffer) >= ACCUM_BYTES_TARGET:
                chunk = buffer[:ACCUM_BYTES_TARGET]
                del buffer[:ACCUM_BYTES_TARGET]

                pcm16 = np.frombuffer(chunk, dtype=np.int16)
                if pcm16.size == 0:
                    continue
                audio = pcm16.astype(np.float32) / 32768.0

                segments, _info = model.transcribe(
                    audio,
                    language=language,
                    vad_filter=True,
                    beam_size=1,
                    condition_on_previous_text=False,
                )
                text = "".join(s.text for s in segments).strip()
                if text and room_id:
                    await broadcast(
                        room_id, None, {"type": "caption", "text": text, "role": role}
                    )
                    if role == "customer":
                        asyncio.create_task(
                            analyze_and_broadcast_intent(room_id, role, text)
                        )
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        await ws.close()
