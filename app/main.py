"""FastAPI application exposing the voice pipeline as a REST API."""

import os
import tempfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.pipeline import VoicePipeline, warmup_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

pipeline = VoicePipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load()
    sample_path = os.environ.get("WARMUP_SAMPLE", "/app/samples/sample_01.wav")
    if os.path.exists(sample_path):
        warmup_pipeline(pipeline, sample_path, rounds=2)
    yield


app = FastAPI(
    title="Voice Pipeline Service",
    description="End-to-end local speech-to-intent pipeline with sub-2s latency.",
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_EXTENSIONS = {".wav"}
MAX_FILE_SIZE_MB = 10


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/process-intent")
async def process_intent(audio: UploadFile = File(...)):
    if audio.filename is None:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    ext = os.path.splitext(audio.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Only .wav files are accepted.",
        )

    content = await audio.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.",
        )

    if len(content) < 44:
        raise HTTPException(status_code=400, detail="File is too small or empty.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = pipeline.process(tmp_path)
        return JSONResponse(content=result)

    except Exception as e:
        logger.error("Pipeline processing failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline processing error: {str(e)}",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
