"""Pipeline orchestrator that sequences ASR -> NLU -> TTS."""

import os
import time
import base64
import logging
from typing import Dict, Any

from app.asr_module import ASREngine
from app.nlu_module import IntentClassifier
from app.tts_module import TTSEngine

logger = logging.getLogger(__name__)


class VoicePipeline:
    """End-to-end voice pipeline: ASR -> NLU -> TTS."""

    def __init__(self):
        self.asr = ASREngine()
        self.tts = TTSEngine()
        self._loaded = False

    def load(self):
        t0 = time.perf_counter()
        self.asr.load()
        t1 = time.perf_counter()

        nlu_model_path = os.environ.get("NLU_MODEL_PATH", "/app/models/nlu/intent_classifier.joblib")
        nlu_vec_path = os.environ.get("NLU_VECTORIZER_PATH", "/app/models/nlu/tfidf_vectorizer.joblib")
        if os.path.exists(nlu_model_path) and os.path.exists(nlu_vec_path):
            self.nlu = IntentClassifier(model_path=nlu_model_path, vectorizer_path=nlu_vec_path)
            logger.info("NLU model loaded from disk.")
        else:
            self.nlu = IntentClassifier()
            logger.info("NLU model trained in-memory.")
        t2 = time.perf_counter()
        self.tts.load()
        t3 = time.perf_counter()
        self._loaded = True
        logger.info(
            "Pipeline loaded in %.1fs (ASR: %.1fs, NLU: %.1fs, TTS: %.1fs)",
            t3 - t0, t1 - t0, t2 - t1, t3 - t2,
        )

    def process(self, audio_path: str) -> Dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        total_start = time.perf_counter()
        latencies = {}

        # ASR
        asr_start = time.perf_counter()
        transcribed_text = self.asr.transcribe(audio_path)
        latencies["asr"] = (time.perf_counter() - asr_start) * 1000

        if not transcribed_text:
            return {
                "transcribed_text": "",
                "intent": "GetWeather",
                "confidence": 0.0,
                "response_audio_b64": "",
                "latencies_ms": {
                    "asr": latencies["asr"],
                    "intent": 0.0,
                    "tts": 0.0,
                    "total": (time.perf_counter() - total_start) * 1000,
                },
            }

        # NLU
        intent_start = time.perf_counter()
        intent, confidence = self.nlu.predict(transcribed_text)
        latencies["intent"] = (time.perf_counter() - intent_start) * 1000

        # Response generation
        response_text = self.nlu.get_response(intent)

        # TTS
        tts_start = time.perf_counter()
        audio_bytes = self.tts.synthesize(response_text)
        latencies["tts"] = (time.perf_counter() - tts_start) * 1000

        response_audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        total_elapsed = (time.perf_counter() - total_start) * 1000

        return {
            "transcribed_text": transcribed_text,
            "intent": intent,
            "confidence": confidence,
            "response_audio_b64": response_audio_b64,
            "latencies_ms": {
                "asr": round(latencies["asr"], 2),
                "intent": round(latencies["intent"], 2),
                "tts": round(latencies["tts"], 2),
                "total": round(total_elapsed, 2),
            },
        }


def warmup_pipeline(pipeline: VoicePipeline, sample_path: str, rounds: int = 2):
    """Run warmup requests to eliminate cold-start latency from benchmarks."""
    logger.info("Warming up pipeline with %d rounds...", rounds)
    for i in range(rounds):
        try:
            pipeline.process(sample_path)
            logger.info("Warmup round %d/%d complete.", i + 1, rounds)
        except Exception as e:
            logger.warning("Warmup round %d failed: %s", i + 1, e)
    logger.info("Pipeline warmup complete.")
