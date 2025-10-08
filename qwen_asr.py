"""
Utility helpers for running Qwen-3-ASR locally and integrating with the FastAPI
service. This module mirrors the interface used by `asr_v2.py` so you can swap
between Whisper and Qwen without touching the websocket flow.

To run Qwen-3-ASR locally you need either:
  * A GPU with at least 16 GB VRAM (24 GB recommended for full precision), or
  * A CPU-only environment willing to tolerate high latency (not recommended).

The examples below assume an NVIDIA GPU with CUDA 11.8+ drivers installed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    # `qwen-vl-utils` exposes the ASR pipeline for Qwen-3; we import lazily so
    # environments without the package can still run Whisper.
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
except Exception as exc:  # pragma: no cover - handled by runtime warning
    AutoModelForSpeechSeq2Seq = None  # type: ignore
    AutoProcessor = None  # type: ignore
    pipeline = None  # type: ignore
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
# For small/CPU setups pick the distilled or 1.5B checkpoints:
# MODEL_ID = "Qwen/Qwen2-Audio-1.5B-Instruct"


@dataclass
class QwenConfig:
    model_id: str = MODEL_ID
    device: str = "cuda"
    torch_dtype: str = "bfloat16"  # or "float16"
    chunk_length_s: float = 20.0  # transcription window
    batch_size: int = 1

    def as_kwargs(self) -> dict[str, object]:
        return {
            "model": self.model_id,
            "torch_dtype": self.torch_dtype,
            "device": self.device,
        }


class QwenTranscriber:
    """
    Wrap the Hugging Face pipeline so the rest of the app can call `transcribe`.
    """

    def __init__(self, config: Optional[QwenConfig] = None) -> None:
        if IMPORT_ERROR:
            raise RuntimeError(
                "Qwen dependencies missing. Install with:\n"
                "  pip install 'transformers>=4.38' 'accelerate>=0.27' 'torch>=2.1' "
                "'sentencepiece' 'soundfile'\n"
                f"Original import error: {IMPORT_ERROR}"
            )
        self.config = config or QwenConfig()

        # Load processor/model explicitly to control dtype/device.
        processor = AutoProcessor.from_pretrained(self.config.model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.model_id,
            torch_dtype=self.config.torch_dtype,  # type: ignore[arg-type]
            device_map=self.config.device,
        )

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=self.config.chunk_length_s,
            batch_size=self.config.batch_size,
        )

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """
        Run transcription on a float32 numpy array normalized to [-1, 1].
        """
        # Hugging Face pipelines accept 16 kHz float arrays directly.
        result = self._pipe(
            audio,
            generate_kwargs={
                "language": language,
                # set `task` to "transcribe" (vs "translate")
                "task": "transcribe",
            },
        )
        if isinstance(result, dict):
            return result.get("text", "").strip()
        if isinstance(result, list) and result:
            return str(result[0].get("text", "")).strip()  # type: ignore[call-arg]
        return ""


# -------------
# Requirements
# -------------
REQUIREMENTS = """
Local Qwen-3-ASR prerequisites:

System:
  - Ubuntu 20.04+/Windows 11/macOS 13.5+ with Python 3.9–3.11.
  - NVIDIA GPU with 16 GB+ VRAM for 7B model (8 GB for 1.5B).
  - CUDA 11.8+ toolkit; install via `conda install cudatoolkit=11.8` or NVIDIA installer.

Python packages (add to `requirements.txt`):
  torch>=2.1.0
  torchvision>=0.16.0
  torchaudio>=2.1.0
  transformers>=4.38.0
  accelerate>=0.27.0
  sentencepiece
  soundfile

Optional:
  - bitsandbytes>=0.41.2 for 8-bit loading on GPUs with less VRAM.
  - flash-attn for faster inference (needs CUDA build).

Model download:
  - `huggingface-cli login` (if model gated) then:
    `huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct --local-dir ./models/qwen2-audio`

Usage:
  from qwen_asr import QwenTranscriber
  transcriber = QwenTranscriber()
  text = transcriber.transcribe(audio_array, language="en")

Integrate with FastAPI queue by replacing `_transcribe_blocking` in `asr_v2.py`
with a call to `transcriber.transcribe`.
""".strip()

