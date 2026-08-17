"""ASR module using faster-whisper for speech-to-text."""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ASREngine:
    """Speech-to-Text engine using faster-whisper with INT8 quantization."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.model = None

    def load(self):
        from faster_whisper import WhisperModel

        logger.info(
            "Loading ASR model: %s (compute_type=%s, device=%s, threads=%d)",
            self.model_size, self.compute_type, self.device, self.cpu_threads,
        )
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )
        logger.info("ASR model loaded successfully.")

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            raise RuntimeError("ASR model not loaded. Call load() first.")

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=1,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        transcribed = " ".join(text_parts).strip()
        logger.info("Transcribed: '%s' (detected language: %s, prob: %.2f)",
                     transcribed, info.language, info.language_probability)
        return transcribed
