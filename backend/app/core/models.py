from sqlalchemy import Column, Integer, String, Float, Enum, ForeignKey, DateTime
from sqlalchemy.orm import relationship
import enum
from datetime import datetime
from .database import Base

class TaskStatus(str, enum.Enum):
    PENDING = "PENDING"
    PREPROCESSING = "PREPROCESSING"
    CHUNKING = "CHUNKING"
    INFERENCE = "INFERENCE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ChunkStatus(str, enum.Enum):
    PENDING = "PENDING"
    ASR_DONE = "ASR_DONE"
    DIARIZATION_DONE = "DIARIZATION_DONE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    filepath = Column(String, index=True, nullable=False)
    processed_wav_path = Column(String, nullable=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    chunks = relationship("Chunk", back_populates="task", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    chunk_index = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    filepath = Column(String, nullable=False)
    status = Column(Enum(ChunkStatus), default=ChunkStatus.PENDING)
    asr_result = Column(String, nullable=True) # JSON stored as string
    diarization_result = Column(String, nullable=True) # JSON stored as string
    
    task = relationship("Task", back_populates="chunks")
