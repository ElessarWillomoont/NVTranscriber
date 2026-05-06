import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.chunker import chunk_audio
from backend.core.converter import convert_to_wav, scan_directory
from backend.database.models import MediaTask, TaskStatus
from backend.database.session import AsyncSessionLocal, get_db, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
WORKSPACE = _PROJECT_ROOT / "workspace"
CONVERTED_DIR = WORKSPACE / "converted"


@asynccontextmanager
async def lifespan(app: FastAPI):
    WORKSPACE.mkdir(exist_ok=True)
    CONVERTED_DIR.mkdir(exist_ok=True)
    await init_db()

    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available — processing will run on CPU")

    yield


app = FastAPI(title="NVTranscriber API", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SyncRequest(BaseModel):
    directory: str


# ---------------------------------------------------------------------------
# Background processor
# ---------------------------------------------------------------------------

async def _process_task(task_id: str) -> None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(MediaTask).where(MediaTask.id == task_id))
        task = result.scalar_one_or_none()
        if not task:
            logger.error(f"Task {task_id} not found in DB")
            return

        output_wav = CONVERTED_DIR / f"{task_id}.wav"

        try:
            task.status = TaskStatus.CONVERTING
            await db.commit()

            success = convert_to_wav(Path(task.original_path), output_wav)
            if not success:
                task.status = TaskStatus.FAILED
                await db.commit()
                return

            task.converted_wav_path = str(output_wav)
            task.status = TaskStatus.CHUNKING
            await db.commit()

            await chunk_audio(str(output_wav), task_id, db)

            task.status = TaskStatus.READY_FOR_INFERENCE
            await db.commit()
            logger.info(f"Task {task_id} → READY_FOR_INFERENCE")

        except Exception as exc:
            logger.error(f"Task {task_id} failed: {exc}", exc_info=True)
            task.status = TaskStatus.FAILED
            await db.commit()


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


@app.post("/api/tasks/{task_id}/process", summary="Begin FFmpeg conversion and VAD chunking")
async def process_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(MediaTask).where(MediaTask.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status not in (TaskStatus.PENDING, TaskStatus.FAILED):
        raise HTTPException(
            status_code=409,
            detail=f"Task is already in state '{task.status}' — only PENDING or FAILED tasks can be reprocessed",
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
            "status": t.status,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in tasks
    ]
