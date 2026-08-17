"""TTS module using pyttsx3 (espeak-ng backend) for local text-to-speech."""

import os
import io
import tempfile
import wave
import struct
import logging
import subprocess
import shutil

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-Speech engine using espeak-ng directly for reliability."""

    def __init__(self, voice: str = "en-us", rate: int = 150, amplitude: int = 100):
        self.voice = voice
        self.rate = rate
        self.amplitude = amplitude
        self._espeak_path = None

    def load(self):
        self._espeak_path = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self._espeak_path:
            raise RuntimeError(
                "Neither espeak-ng nor espeak found. Install espeak-ng: "
                "apt-get install -y espeak-ng"
            )
        logger.info("TTS engine loaded (espeak path: %s, voice: %s)",
                     self._espeak_path, self.voice)

    def synthesize(self, text: str) -> bytes:
        if not self._espeak_path:
            raise RuntimeError("TTS engine not loaded. Call load() first.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                self._espeak_path,
                "-v", self.voice,
                "-s", str(self.rate),
                "-a", str(self.amplitude),
                "-w", tmp_path,
                text,
            ]
            result = subprocess.run(
                cmd, capture_output=True, timeout=10, check=True
            )
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            return audio_bytes
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
