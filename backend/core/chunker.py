import asyncio
import logging
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import AudioChunk, ChunkStatus

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SIZE = 512          # Silero VAD window size for 16 kHz
CONTEXT_SIZE = 64          # Samples prepended to each window (v5 requirement)
VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION = 0.1  # seconds — discard sub-100 ms blips

MIN_CHUNK_DURATION = 30.0   # 30 seconds
MAX_CHUNK_DURATION = 60.0   # 60 seconds

_MODEL_PATH = Path(__file__).parent / "silero_vad.onnx"
_SILERO_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad"
    "/master/src/silero_vad/data/silero_vad.onnx"
)

_session: ort.InferenceSession | None = None


def _ensure_model() -> None:
    if not _MODEL_PATH.exists():
        logger.info("Downloading Silero VAD ONNX model …")
        urllib.request.urlretrieve(_SILERO_URL, str(_MODEL_PATH))
        logger.info(f"Silero VAD model saved → {_MODEL_PATH}")


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is not None:
        return _session

    _ensure_model()
    # Silero VAD is a tiny model (~1 MB). The per-window CPU→GPU memcpy overhead
    # exceeds the compute savings, so CPU is both faster and avoids the ORT
    # MemcpyTransformer warning. This matches the official silero-vad OnnxWrapper.
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(
        str(_MODEL_PATH),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    logger.info("Silero VAD ONNX running on CPUExecutionProvider")
    _session = sess
    return _session


def _run_vad(wav_path: str) -> tuple[list[dict], float]:
    """Stream-read wav_path and return (speech_segments, total_duration) via Silero VAD v5.

    Uses sf.blocks() to avoid loading the full file into RAM.
    v5 ONNX interface:
      inputs:  input [1, CONTEXT_SIZE + WINDOW_SIZE], state [2, 1, 128], sr int64
      outputs: output [1, 1], stateN [2, 1, 128]
    """
    info = sf.info(wav_path)
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz WAV, got {info.samplerate} Hz — "
            "re-run the FFmpeg conversion step."
        )
    total_duration = info.frames / info.samplerate

    sess = _get_session()
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, CONTEXT_SIZE), dtype=np.float32)
    sr_arr = np.array(SAMPLE_RATE, dtype=np.int64)

    speeches: list[dict] = []
    triggered = False
    speech_start = 0.0
    offset = 0
    max_prob = 0.0

    for block in sf.blocks(wav_path, blocksize=WINDOW_SIZE, dtype="float32", always_2d=False):
        if block.ndim > 1:
            block = block[:, 0]  # stereo → mono
        if len(block) < WINDOW_SIZE:
            block = np.pad(block, (0, WINDOW_SIZE - len(block)))
        chunk = block.reshape(1, -1).astype(np.float32)

        # Prepend context → actual model input is [1, CONTEXT_SIZE + WINDOW_SIZE]
        inp = np.concatenate([context, chunk], axis=1)
        output, state = sess.run(None, {"input": inp, "state": state, "sr": sr_arr})
        prob: float = float(output[0][0])
        max_prob = max(max_prob, prob)

        # Slide context forward: keep the last CONTEXT_SIZE samples of this input
        context = inp[:, -CONTEXT_SIZE:]

        ts = offset / SAMPLE_RATE

        if prob >= VAD_THRESHOLD and not triggered:
            triggered = True
            speech_start = ts
        elif prob < VAD_THRESHOLD and triggered:
            triggered = False
            seg_dur = ts - speech_start
            if seg_dur >= MIN_SPEECH_DURATION:
                speeches.append({"start": speech_start, "end": ts})

        offset += WINDOW_SIZE

    if triggered:
        seg_dur = total_duration - speech_start
        if seg_dur >= MIN_SPEECH_DURATION:
            speeches.append({"start": speech_start, "end": total_duration})

    logger.info(f"VAD: max_prob={max_prob:.3f}, detected {len(speeches)} speech segment(s)")
    return speeches, total_duration


def _aggregate_chunks(speeches: list[dict], total_duration: float) -> list[tuple[float, float]]:
    """
    Merge speech segments into chunks of MIN_CHUNK_DURATION–MAX_CHUNK_DURATION.
    Cuts are placed only at silence gaps (ends of speech segments), never mid-speech.
    If VAD found no speech, force-split the file into MAX_CHUNK_DURATION windows.
    """
    if not speeches:
        if total_duration <= MAX_CHUNK_DURATION:
            return [(0.0, total_duration)]
        chunks: list[tuple[float, float]] = []
        start = 0.0
        while start < total_duration:
            end = min(start + MAX_CHUNK_DURATION, total_duration)
            chunks.append((start, end))
            start = end
        return chunks

    chunks: list[tuple[float, float]] = []
    chunk_start = 0.0

    for i, seg in enumerate(speeches):
        is_last = i == len(speeches) - 1
        wall_clock = seg["end"] - chunk_start

        # Cut at this silence boundary when:
        # • we have met the minimum duration AND there is a next segment, OR
        # • we have exceeded the maximum duration (forced cut)
        should_cut = (
            (wall_clock >= MIN_CHUNK_DURATION and not is_last)
            or (wall_clock >= MAX_CHUNK_DURATION)
        )

        if should_cut:
            chunks.append((chunk_start, seg["end"]))
            chunk_start = seg["end"]

    # Flush tail (trailing silence or remaining content)
    if chunk_start < total_duration:
        chunks.append((chunk_start, total_duration))

    return chunks


async def chunk_audio(wav_path: str, task_id: str, db: AsyncSession) -> list[tuple[float, float]]:
    """VAD-chunk a 16 kHz mono WAV and persist AudioChunk rows for each segment."""
    # Remove stale chunks from any previous run of this task
    await db.execute(
        sa_text("DELETE FROM audio_chunks WHERE task_id = :task_id"),
        {"task_id": task_id},
    )
    await db.commit()

    speeches, total_duration = await asyncio.to_thread(_run_vad, wav_path)
    logger.info(f"Audio duration: {total_duration:.2f}s — VAD found {len(speeches)} segment(s)")

    boundaries = _aggregate_chunks(speeches, total_duration)

    for seq, (start, end) in enumerate(boundaries):
        chunk = AudioChunk(
            task_id=task_id,
            sequence_number=seq,
            start_offset=round(start, 4),
            end_offset=round(end, 4),
            status=ChunkStatus.PENDING,
        )
        db.add(chunk)
        await db.commit()
        logger.info(f"  Chunk {seq:03d}: {start:.2f}s – {end:.2f}s  ({end - start:.1f}s)")

    logger.info(f"Created {len(boundaries)} chunk(s) for task {task_id}")
    return boundaries
