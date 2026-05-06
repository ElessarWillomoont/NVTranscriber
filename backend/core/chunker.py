import logging
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torchaudio
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import AudioChunk, ChunkStatus

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SIZE = 512          # Silero VAD window size for 16 kHz
CONTEXT_SIZE = 64          # Samples prepended to each window (v5 requirement)
VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION = 0.1  # seconds — discard sub-100 ms blips

MIN_CHUNK_DURATION = 300.0  # 5 minutes
MAX_CHUNK_DURATION = 600.0  # 10 minutes

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


def _run_vad(audio: np.ndarray) -> list[dict]:
    """Return list of {start, end} speech segments (seconds) via Silero VAD v5.

    v5 ONNX interface:
      inputs:  input [1, CONTEXT_SIZE + WINDOW_SIZE], state [2, 1, 128], sr int64
      outputs: output [1, 1], stateN [2, 1, 128]

    The model requires the last CONTEXT_SIZE samples from the previous window to
    be prepended to the current window. Without this the model sees incomplete
    input and returns near-zero probabilities for all frames.
    """
    sess = _get_session()
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, CONTEXT_SIZE), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)

    speeches: list[dict] = []
    triggered = False
    speech_start = 0.0
    n = len(audio)
    max_prob = 0.0

    for offset in range(0, n, WINDOW_SIZE):
        chunk = audio[offset: offset + WINDOW_SIZE]
        if len(chunk) < WINDOW_SIZE:
            chunk = np.pad(chunk, (0, WINDOW_SIZE - len(chunk)))
        chunk = chunk.reshape(1, -1).astype(np.float32)

        # Prepend context → actual model input is [1, CONTEXT_SIZE + WINDOW_SIZE]
        inp = np.concatenate([context, chunk], axis=1)
        output, state = sess.run(None, {"input": inp, "state": state, "sr": sr})
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

    if triggered:
        seg_dur = (n / SAMPLE_RATE) - speech_start
        if seg_dur >= MIN_SPEECH_DURATION:
            speeches.append({"start": speech_start, "end": n / SAMPLE_RATE})

    logger.info(f"VAD: max_prob={max_prob:.3f}, detected {len(speeches)} speech segment(s)")
    return speeches


def _aggregate_chunks(speeches: list[dict], total_duration: float) -> list[tuple[float, float]]:
    """
    Merge speech segments into chunks of MIN_CHUNK_DURATION–MAX_CHUNK_DURATION.
    Cuts are placed only at silence gaps (ends of speech segments), never mid-speech.
    """
    if not speeches:
        return [(0.0, total_duration)]

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
    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        logger.warning(f"Resampling from {sr} Hz to {SAMPLE_RATE} Hz")
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    audio_np: np.ndarray = waveform.squeeze().numpy()
    total_duration = len(audio_np) / SAMPLE_RATE
    logger.info(f"Audio duration: {total_duration:.2f}s — running VAD …")

    speeches = _run_vad(audio_np)
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
