import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a small model to keep it snappy on CPU. You can switch "tiny" -> "base".
# First run will download the model to ~/.cache
model = WhisperModel("base", device="cpu")  # use "tiny" for even lighter

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # int16
CHUNK_MS = 40  # how much audio each packet contains from the client (typical 20–60ms)
ACCUM_MS = 1000  # transcribe every ~1s of audio
ACCUM_BYTES_TARGET = SAMPLE_RATE * BYTES_PER_SAMPLE * ACCUM_MS // 1000


@app.websocket("/asr")
async def asr_ws(ws: WebSocket):
    await ws.accept()
    buffer = bytearray()
    try:
        # Optional: first message could be JSON like {"lang":"en"}; we’ll just consume bytes directly.
        while True:
            data = await ws.receive_bytes()
            buffer.extend(data)

            # When we have ~1 second, transcribe it.
            while len(buffer) >= ACCUM_BYTES_TARGET:
                chunk = buffer[:ACCUM_BYTES_TARGET]
                del buffer[:ACCUM_BYTES_TARGET]

                # int16 PCM -> float32 [-1,1]
                pcm16 = np.frombuffer(chunk, dtype=np.int16)
                audio = pcm16.astype(np.float32) / 32768.0

                # Transcribe this second; small segments, quick turnaround.
                # For POC, condition_on_previous_text=False avoids unwanted “sticky” context.
                segments, _info = model.transcribe(
                    audio,
                    language="en",
                    vad_filter=True,
                    beam_size=1,
                    condition_on_previous_text=False,
                )
                text = "".join(s.text for s in segments).strip()
                if text:
                    await ws.send_json({"type": "caption", "text": text})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        # send error for debugging
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        await ws.close()


@app.get("/")
def ok():
    return {"ok": True}
