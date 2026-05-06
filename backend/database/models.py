import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    CONVERTING = "CONVERTING"
    CHUNKING = "CHUNKING"
    READY_FOR_INFERENCE = "READY_FOR_INFERENCE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ChunkStatus(str, Enum):
    PENDING = "PENDING"
    DONE = "DONE"


class MediaTask(Base):
    __tablename__ = "media_tasks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    original_path = Column(String, nullable=False)
    converted_wav_path = Column(String, nullable=True)
    status = Column(SAEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    chunks = relationship("AudioChunk", back_populates="task", cascade="all, delete-orphan")


class AudioChunk(Base):
    __tablename__ = "audio_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, ForeignKey("media_tasks.id"), nullable=False)
    sequence_number = Column(Integer, nullable=False)
    start_offset = Column(Float, nullable=False)
    end_offset = Column(Float, nullable=False)
    status = Column(SAEnum(ChunkStatus), default=ChunkStatus.PENDING, nullable=False)

    task = relationship("MediaTask", back_populates="chunks")
