"""
ASR Engine — Whisper-large-v3 via HF Transformers.

nvidia/canary-1b requires the NeMo toolkit which conflicts with the current
HF transformers stack; openai/whisper-large-v3 is used as the primary model.
Swap ASR_MODEL_ID to any AutoModelForSpeechSeq2Seq-compatible checkpoint to
change the model without touching the rest of the pipeline.
"""
import json
import logging
from pathlib import Path

import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from backend.core.vram_manager import clear_gpu_memory, get_inference_device
from backend.database.models import AudioChunk

logger = logging.getLogger(__name__)

ASR_MODEL_ID = "openai/whisper-small"   # fast Phase 1 scout; Focus Mode overrides this
SAMPLE_RATE = 16000

# Internal sub-chunk size fed to Whisper's attention window.
# The HF pipeline splits each AudioChunk into overlapping 30-s sub-windows
# automatically when chunk_length_s is set.
_WHISPER_WINDOW_S = 30


def _build_pipeline(
    device: torch.device,
    model_id: str = ASR_MODEL_ID,
) -> pipeline:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    logger.info(f"Loading ASR model '{model_id}' → {device} ({dtype}) …")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        batch_size=16,   # parallelise 30-s sub-windows; effective on ≥16 GB VRAM
    )
    logger.info("ASR model ready")
    return asr_pipe


def run_asr(
    wav_path: str,
    chunks: list[AudioChunk],
    output_json_path: Path,
) -> list[dict]:
    """
    Transcribe a list of AudioChunks and write results to output_json_path.

    Returns a list of segment dicts:
        {"chunk_id": int, "start": float, "end": float, "text": str}
    where start/end are global timestamps (seconds from the beginning of the
    full recording).
    """
    device = get_inference_device(min_vram_gb=4.0)
    asr_pipe = None

    try:
        asr_pipe = _build_pipeline(device)

        info = sf.info(wav_path)
        if info.samplerate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz WAV, got {info.samplerate} Hz")

        results: list[dict] = []

        for chunk in sorted(chunks, key=lambda c: c.sequence_number):
            start_s = chunk.start_offset
            end_s = chunk.end_offset
            start_sample = int(start_s * SAMPLE_RATE)
            n_frames = int((end_s - start_s) * SAMPLE_RATE)
            slice_np, _ = sf.read(
                wav_path, start=start_sample, frames=n_frames,
                dtype="float32", always_2d=False,
            )
            if slice_np.ndim > 1:
                slice_np = slice_np[:, 0]

            logger.info(
                f"ASR chunk {chunk.sequence_number:03d}: "
                f"{start_s:.2f}s – {end_s:.2f}s  ({end_s - start_s:.1f}s)"
            )

            try:
                out = asr_pipe(
                    {"array": slice_np, "sampling_rate": SAMPLE_RATE},
                    return_timestamps=True,
                    chunk_length_s=_WHISPER_WINDOW_S,
                    generate_kwargs={"task": "transcribe"},
                )
            except torch.cuda.OutOfMemoryError:
                logger.error(
                    f"VRAM OOM on chunk {chunk.sequence_number} — "
                    "re-raising so the task is marked FAILED. "
                    "Reduce the model size or free VRAM before retrying."
                )
                raise

            sub_segments = out.get("chunks", [])
            if sub_segments:
                for seg in sub_segments:
                    ts = seg.get("timestamp") or (0.0, end_s - start_s)
                    seg_start = (ts[0] or 0.0) + start_s
                    # ts[1] can be None for the final segment; clamp to chunk end
                    seg_end = (ts[1] if ts[1] is not None else (end_s - start_s)) + start_s
                    text = seg.get("text", "").strip()
                    if text:
                        results.append({
                            "chunk_id": chunk.id,
                            "start": round(seg_start, 3),
                            "end": round(seg_end, 3),
                            "text": text,
                        })
            else:
                # Whisper returned a single text with no sub-segment timestamps
                text = out.get("text", "").strip()
                if text:
                    results.append({
                        "chunk_id": chunk.id,
                        "start": round(start_s, 3),
                        "end": round(end_s, 3),
                        "text": text,
                    })

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"ASR complete: {len(results)} segment(s) → {output_json_path}")
        return results

    finally:
        del asr_pipe
        clear_gpu_memory()
