import asyncio
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.aligner import align_and_write_srt
from backend.core.asr_engine import run_asr
from backend.core.chunker import chunk_audio
from backend.core.converter import convert_to_wav, scan_directory
from backend.core.diarization_engine import run_diarization
from backend.database.models import AudioChunk, MediaTask, TaskStatus
from backend.database.session import AsyncSessionLocal, get_db, init_db

# Load HF_TOKEN and any other env vars from backend/.env before anything else
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
WORKSPACE = _PROJECT_ROOT / "workspace"
CONVERTED_DIR = WORKSPACE / "converted"
TRANSCRIPTS_DIR = WORKSPACE / "transcripts"
OUTPUTS_DIR = WORKSPACE / "outputs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    for d in (WORKSPACE, CONVERTED_DIR, TRANSCRIPTS_DIR, OUTPUTS_DIR):
        d.mkdir(exist_ok=True)
    await init_db()

    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available — inference will run on CPU")

    yield


app = FastAPI(title="NVTranscriber API", version="0.2.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SyncRequest(BaseModel):
    directory: str


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------

async def _set_status(db: AsyncSession, task: MediaTask, status: TaskStatus) -> None:
    task.status = status
    await db.commit()
    logger.info(f"Task {task.id} → {status.value}")


async def _process_task(task_id: str) -> None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(MediaTask).where(MediaTask.id == task_id))
        task = result.scalar_one_or_none()
        if not task:
            logger.error(f"Task {task_id} not found in DB")
            return

        output_wav = CONVERTED_DIR / f"{task_id}.wav"
        asr_json = TRANSCRIPTS_DIR / f"{task_id}_asr.json"
        diar_json = TRANSCRIPTS_DIR / f"{task_id}_diarization.json"
        srt_path = OUTPUTS_DIR / f"{task_id}.srt"

        try:
            # ── Phase 1: FFmpeg conversion ──────────────────────────────────
            await _set_status(db, task, TaskStatus.CONVERTING)
            success = await asyncio.to_thread(
                convert_to_wav, Path(task.original_path), output_wav
            )
            if not success:
                await _set_status(db, task, TaskStatus.FAILED)
                return

            task.converted_wav_path = str(output_wav)
            await db.commit()

            # ── Phase 1: VAD chunking ────────────────────────────────────────
            await _set_status(db, task, TaskStatus.CHUNKING)
            await chunk_audio(str(output_wav), task_id, db)

            # Fetch the persisted chunks for ASR
            chunk_result = await db.execute(
                select(AudioChunk)
                .where(AudioChunk.task_id == task_id)
                .order_by(AudioChunk.sequence_number)
            )
            chunks = list(chunk_result.scalars().all())
            if not chunks:
                logger.error(f"No chunks found for task {task_id} after VAD")
                await _set_status(db, task, TaskStatus.FAILED)
                return

            # ── Phase 2: ASR ─────────────────────────────────────────────────
            await _set_status(db, task, TaskStatus.ASR_INFERENCE)
            asr_segments = await asyncio.to_thread(
                run_asr, str(output_wav), chunks, asr_json
            )

            # ── Phase 2: Diarization ─────────────────────────────────────────
            await _set_status(db, task, TaskStatus.DIARIZING)
            diar_segments = await asyncio.to_thread(
                run_diarization, str(output_wav), diar_json
            )

            # ── Phase 2: Alignment + SRT ─────────────────────────────────────
            await _set_status(db, task, TaskStatus.ALIGNING)
            await asyncio.to_thread(
                align_and_write_srt, asr_segments, diar_segments, srt_path
            )

            task.output_srt_path = str(srt_path)
            await _set_status(db, task, TaskStatus.COMPLETED)

        except Exception as exc:
            logger.error(f"Task {task_id} failed: {exc}", exc_info=True)
            try:
                task.status = TaskStatus.FAILED
                await db.commit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/tasks/sync", summary="Scan a directory and register media files as tasks")
async def sync_directory(request: SyncRequest, db: AsyncSession = Depends(get_db)):
    target = Path(request.directory)
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.directory}")

    media_files = scan_directory(request.directory)
    created = []
    for file_path in media_files:
        task = MediaTask(
            id=str(uuid.uuid4()),
            original_path=str(file_path),
            status=TaskStatus.PENDING,
        )
        db.add(task)
        created.append({"id": task.id, "path": str(file_path)})

    await db.commit()
    return {"scanned_directory": request.directory, "created": len(created), "tasks": created}


_RETRIABLE_STATUSES = {
    TaskStatus.PENDING,
    TaskStatus.FAILED,
    TaskStatus.READY_FOR_INFERENCE,  # tasks left from the Phase-1-only build
}


@app.post(
    "/api/tasks/{task_id}/process",
    summary="Run the full pipeline: FFmpeg → VAD → ASR → Diarize → Align → SRT",
)
async def process_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(MediaTask).where(MediaTask.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status not in _RETRIABLE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Task is in state '{task.status.value}' — "
                "only PENDING, FAILED, or READY_FOR_INFERENCE tasks can be (re)processed"
            ),
        )

    background_tasks.add_task(_process_task, task_id)
    return {"task_id": task_id, "status": "processing_started"}


@app.get("/api/tasks", summary="List all tasks and their current statuses")
async def list_tasks(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(MediaTask))
    tasks = result.scalars().all()
    return [
        {
            "id": t.id,
            "original_path": t.original_path,
            "converted_wav_path": t.converted_wav_path,
            "output_srt_path": t.output_srt_path,
            "status": t.status,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in tasks
    ]
