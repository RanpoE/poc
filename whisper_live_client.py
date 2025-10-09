import json
import threading
import time
import uuid
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import websocket


class WhisperLiveBridge:
    """Minimal client to push PCM audio to a whisper-live server over WebSocket."""

    END_OF_AUDIO = "END_OF_AUDIO"

    def __init__(
        self,
        host: str,
        port: int,
        *,
        language: Optional[str] = None,
        translate: bool = False,
        model: str = "small",
        use_vad: bool = False,
        use_wss: bool = False,
        max_clients: int = 4,
        max_connection_time: int = 600,
        send_last_n_segments: int = 10,
        no_speech_thresh: float = 0.45,
        clip_audio: bool = False,
        same_output_threshold: int = 10,
        extra_init_payload: Optional[Dict[str, object]] = None,
        transcription_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        protocol = "wss" if use_wss else "ws"
        self.url = f"{protocol}://{host}:{port}"
        self.language = language or "en"
        self.translate = translate
        self.model = model
        self.use_vad = use_vad
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold
        self.extra_init_payload = extra_init_payload or {}
        self.transcription_callback = transcription_callback

        self.uid = str(uuid.uuid4())
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._closed_event = threading.Event()
        self._seen_segments: set[tuple[float, float, str]] = set()
        self._server_error: Optional[str] = None
        self._last_transcript: Optional[str] = None
        self._mutex = threading.Lock()

        self._start()

    # -----------------
    # Public interface
    # -----------------
    def wait_until_ready(self, timeout: float = 15.0) -> None:
        """Block until the remote server is ready to receive audio."""
        if not self._ready_event.wait(timeout):
            if self._server_error:
                raise RuntimeError(self._server_error)
            raise TimeoutError("whisper-live server did not become ready in time.")

    def send_audio(self, audio: np.ndarray) -> None:
        """Send a mono float32 numpy array to the remote server."""
        with self._mutex:
            if not self._ws or self._closed_event.is_set():
                raise RuntimeError("Connection is closed.")
            if self._server_error:
                raise RuntimeError(self._server_error)
        if not self._ready_event.is_set():
            raise RuntimeError("whisper-live server is not ready yet.")
        payload = np.asarray(audio, dtype=np.float32).tobytes()
        try:
            self._ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(f"Failed to send audio to whisper-live: {exc}") from exc

    def send_end_of_audio(self) -> None:
        """Notify the server that no more audio will be sent."""
        with self._mutex:
            if not self._ws or self._closed_event.is_set():
                return
        try:
            self._ws.send(
                self.END_OF_AUDIO.encode("utf-8"), opcode=websocket.ABNF.OPCODE_TEXT
            )
        except Exception:
            pass

    def close(self) -> None:
        """Terminate the websocket connection."""
        with self._mutex:
            if self._closed_event.is_set():
                return
            self._closed_event.set()
        try:
            self.send_end_of_audio()
        except Exception:
            pass
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5)

    # -----------------
    # Internal helpers
    # -----------------
    def _start(self) -> None:
        self._ws = websocket.WebSocketApp(
            self.url,
            on_open=self._handle_open,
            on_message=self._handle_message,
            on_error=self._handle_error,
            on_close=self._handle_close,
        )
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()

    def _handle_open(
        self, ws: websocket.WebSocketApp
    ) -> None:  # pragma: no cover - network path
        init_payload: Dict[str, object] = {
            "uid": self.uid,
            "language": self.language,
            "task": "translate" if self.translate else "transcribe",
            "model": self.model,
            "use_vad": self.use_vad,
            "max_clients": self.max_clients,
            "max_connection_time": self.max_connection_time,
            "send_last_n_segments": self.send_last_n_segments,
            "no_speech_thresh": self.no_speech_thresh,
            "clip_audio": self.clip_audio,
            "same_output_threshold": self.same_output_threshold,
        }
        init_payload.update(self.extra_init_payload)
        try:
            ws.send(json.dumps(init_payload))
        except Exception as exc:
            self._server_error = f"Failed to send init payload: {exc}"
            self.close()

    def _handle_message(
        self, _ws: websocket.WebSocketApp, message: str
    ) -> None:  # pragma: no cover
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        if data.get("uid") and data["uid"] != self.uid:
            return

        if "status" in data:
            self._handle_status(data)
            return

        msg = data.get("message")
        if msg == "SERVER_READY":
            self._ready_event.set()
            return
        if msg == "DISCONNECT":
            self._server_error = "whisper-live server disconnected."
            self.close()
            return

        segments = data.get("segments")
        if isinstance(segments, Iterable):
            self._handle_segments(segments)

    def _handle_status(self, data: Dict[str, object]) -> None:
        status = data.get("status")
        if status == "WAIT":
            eta = data.get("message")
            message = (
                f"whisper-live server busy (ETA {eta} minutes)"
                if eta is not None
                else "whisper-live server busy"
            )
            self._server_error = message
        elif status == "ERROR":
            self._server_error = str(data.get("message", "whisper-live server error"))
        elif status == "WARNING":
            # Warnings are logged but do not impact readiness.
            print(f"[whisper-live] warning: {data.get('message')}")

    def _handle_segments(self, segments: Iterable[Dict[str, object]]) -> None:
        new_texts: list[str] = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            if not seg.get("completed", False):
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            key = (start, end, text)
            if key in self._seen_segments:
                continue
            self._seen_segments.add(key)
            new_texts.append(text)
        if not new_texts:
            return
        combined = " ".join(new_texts).strip()
        if not combined:
            return
        if combined == self._last_transcript:
            return
        self._last_transcript = combined
        if self.transcription_callback:
            try:
                self.transcription_callback(combined)
            except Exception as exc:
                print(f"[whisper-live] transcription callback error: {exc}")

    def _handle_error(
        self, _ws: websocket.WebSocketApp, error: Exception
    ) -> None:  # pragma: no cover
        self._server_error = f"whisper-live websocket error: {error}"

    def _handle_close(
        self,
        _ws: websocket.WebSocketApp,
        close_status_code: Optional[int],
        close_msg: Optional[str],
    ) -> None:  # pragma: no cover
        self._closed_event.set()
        if close_status_code not in (1000, 1001):
            if not self._server_error:
                self._server_error = (
                    f"whisper-live websocket closed: {close_status_code} {close_msg}"
                )
