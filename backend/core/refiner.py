"""
Phase 2.7 — Speaker-Aware Re-Transcription (Focus Mode).

Simplified pipeline (Phase B / CTC removed):

  Phase 2.6 atomic tokens  (session boundaries + speaker labels)
        ↓
  Session aggregation      (speaker change | silence gap > 2 s)
        ↓
  Surgical in-memory audio slicing  (numpy views — no temp files)
        ↓
  Whisper-large-v3 ASR     (pure single-speaker context per session)
        ↓
  Whisper semantic chunks  →  global timestamps  →  _build_srt
        ↓
  {task_id}_focus.srt

Whisper's native chunk boundaries replace CTC forced alignment.
This preserves English code-switching and natural sentence breaks without
the uroman / Wav2Vec2 dependency chain.
"""
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from backend.core.asr_engine import _build_pipeline
from backend.core.vram_manager import clear_gpu_memory, get_inference_device

logger = logging.getLogger(__name__)

_SAMPLE_RATE   = 16000
_SESSION_GAP_S = 2.0   # silence gap that triggers a new speaker session
_PADDING_S     = 0.2   # safety padding added to both ends of each audio slice
_MIN_DURATION  = 0.3   # sessions shorter than this are skipped (seconds)

_FOCUS_MODEL_ID = "openai/whisper-large-v3"   # deep-transcription model for Focus Mode


# ---------------------------------------------------------------------------
# Step 1 — Session Aggregation
# ---------------------------------------------------------------------------

def aggregate_sessions(
    atomic_tokens: list[dict],
    session_gap_s: float = _SESSION_GAP_S,
) -> list[dict]:
    """
    Group atomic tokens into Speaker Sessions.

    A new session is started when:
    - The speaker tag changes, OR
    - The silence gap between consecutive tokens exceeds session_gap_s.

    Parameters
    ----------
    atomic_tokens : list of {"char", "start", "end", "speaker"}
    session_gap_s : silence threshold (seconds)

    Returns
    -------
    list of {"speaker": str, "start": float, "end": float, "tokens": list[dict]}
    """
    if not atomic_tokens:
        return []

    sessions: list[dict] = []
    state: dict = {
        "speaker": atomic_tokens[0]["speaker"],
        "tokens":  [atomic_tokens[0]],
    }

    def _flush() -> None:
        toks = state["tokens"]
        sessions.append({
            "speaker": state["speaker"],
            "start":   toks[0]["start"],
            "end":     toks[-1]["end"],
            "tokens":  list(toks),
        })
        toks.clear()

    for token in atomic_tokens[1:]:
        gap        = token["start"] - state["tokens"][-1]["end"]
        spk_change = token["speaker"] != state["speaker"]

        if spk_change or gap > session_gap_s:
            _flush()
            state["speaker"] = token["speaker"]

        state["tokens"].append(token)

    _flush()
    return sessions


# ---------------------------------------------------------------------------
# Step 2 — Surgical Audio Slicing (in-memory, zero-copy views)
# ---------------------------------------------------------------------------

def slice_audio_session(
    audio_np: np.ndarray,
    start_s: float,
    end_s: float,
    sample_rate: int = _SAMPLE_RATE,
    padding_s: float = _PADDING_S,
) -> tuple[np.ndarray, float]:
    """
    Extract a padded audio slice as a numpy view (no allocation).

    Returns
    -------
    (slice_view, actual_start_s)
    actual_start_s is the global timestamp of the first sample in slice_view.
    """
    total_s        = len(audio_np) / sample_rate
    actual_start_s = max(0.0, start_s - padding_s)
    actual_end_s   = min(total_s,    end_s   + padding_s)
    start_sample   = int(actual_start_s * sample_rate)
    end_sample     = int(actual_end_s   * sample_rate)
    return audio_np[start_sample:end_sample], actual_start_s


# ---------------------------------------------------------------------------
# Step 3 — Whisper transcription helper
# ---------------------------------------------------------------------------

def _transcribe_slice(audio_np: np.ndarray, pipe) -> list[dict]:
    """
    Transcribe a numpy audio slice via the Whisper pipeline.
    Returns pipeline chunk dicts: [{"text": str, "timestamp": (s, e)}, …].
    """
    out = pipe(
        {"array": audio_np, "sampling_rate": _SAMPLE_RATE},
        return_timestamps=True,
        chunk_length_s=30,
        generate_kwargs={"task": "transcribe"},
    )
    return out.get("chunks", [])


# ---------------------------------------------------------------------------
# Phase 2.7 main pipeline
# ---------------------------------------------------------------------------

def run_focus_mode(
    wav_path: str,
    atomic_tokens_path: Path,
    output_srt_path: Path,
    session_gap_s: float = _SESSION_GAP_S,
    padding_s: float = _PADDING_S,
) -> Path:
    """
    Phase 2.7 Focus Mode — Whisper-native subtitle generation per speaker session.

    Steps
    -----
    1. Parse atomic_tokens.json → aggregate into speaker sessions.
    2. For each session: slice WAV in memory (padded, no temp files).
    3. Whisper ASR on each slice → chunk list with local timestamps.
    4. Offset every chunk's timestamps by actual_start_s → global timeline.
    5. Attach session speaker label to each chunk → subtitle block.
    6. Write focus SRT directly from the block list.

    Whisper's semantic chunk boundaries are used as subtitle break points.
    No CTC / Wav2Vec2 pass is performed; English code-switching is preserved.

    Returns
    -------
    output_srt_path
    """
    from backend.core.aligner import _build_srt

    # ── Load atomic tokens ────────────────────────────────────────────────
    atomic_tokens: list[dict] = json.loads(
        atomic_tokens_path.read_text(encoding="utf-8")
    )
    if not atomic_tokens:
        logger.warning("Focus Mode: atomic_tokens.json is empty — nothing to refine")
        output_srt_path.parent.mkdir(parents=True, exist_ok=True)
        output_srt_path.write_text("", encoding="utf-8")
        return output_srt_path

    # ── Load full audio ───────────────────────────────────────────────────
    audio_np, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]
    if sr != _SAMPLE_RATE:
        raise ValueError(f"Expected {_SAMPLE_RATE} Hz WAV, got {sr} Hz")

    # ── Session aggregation ───────────────────────────────────────────────
    sessions = aggregate_sessions(atomic_tokens, session_gap_s=session_gap_s)
    logger.info(f"Focus Mode: {len(sessions)} speaker session(s) aggregated")

    # ── Whisper re-transcription → subtitle blocks ────────────────────────
    sentences: list[dict] = []

    device   = get_inference_device(min_vram_gb=4.0)
    asr_pipe = _build_pipeline(device, model_id=_FOCUS_MODEL_ID)

    try:
        for i, session in enumerate(sessions):
            start_s  = session["start"]
            end_s    = session["end"]
            duration = end_s - start_s

            if duration < _MIN_DURATION:
                logger.debug(f"  Session {i+1}: skipped (too short: {duration:.2f}s)")
                continue

            logger.info(
                f"  Focus ASR {i+1}/{len(sessions)} "
                f"[{session['speaker']}] {start_s:.2f}s–{end_s:.2f}s ({duration:.1f}s)"
            )

            slice_np, offset_s = slice_audio_session(
                audio_np, start_s, end_s, padding_s=padding_s
            )
            # Padded slice end — used as fallback when Whisper omits ts[1]
            actual_end_s = min(len(audio_np) / _SAMPLE_RATE, end_s + padding_s)

            try:
                chunks = _transcribe_slice(slice_np, asr_pipe)
            except torch.cuda.OutOfMemoryError:
                logger.error(f"VRAM OOM on Focus ASR session {i+1} — re-raising")
                raise

            for chunk in chunks:
                chunk_text = chunk.get("text", "").strip()
                if not chunk_text:
                    continue

                ts          = chunk.get("timestamp") or (0.0, None)
                local_start = ts[0] or 0.0
                # ts[1] is None for the last sub-window; fall back to slice end
                local_end   = ts[1] if ts[1] is not None else (actual_end_s - offset_s)

                sentences.append({
                    "speaker": session["speaker"],
                    "start":   round(local_start + offset_s, 3),
                    "end":     round(local_end   + offset_s, 3),
                    "text":    chunk_text,
                })

    finally:
        del asr_pipe
        clear_gpu_memory()

    # ── Render & write ────────────────────────────────────────────────────
    output_srt_path.parent.mkdir(parents=True, exist_ok=True)
    output_srt_path.write_text(_build_srt(sentences), encoding="utf-8")
    logger.info(
        f"Focus Mode SRT written → {output_srt_path} "
        f"({len(sentences)} subtitle(s))"
    )
    return output_srt_path
