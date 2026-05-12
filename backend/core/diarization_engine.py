"""
Diarization Engine — pyannote/speaker-diarization-3.1.

The pipeline is run on the FULL converted WAV (not per-chunk) so pyannote can
build a global speaker embedding space and guarantee that SPEAKER_00 in minute
1 is the same identity as SPEAKER_00 in minute 50.

Requires HF_TOKEN in the environment (or backend/.env) to download the gated
pyannote model weights.
"""
import json
import logging
import os
from pathlib import Path

import torch
from pyannote.audio import Pipeline as PyannotePipeline

from backend.core.vram_manager import clear_gpu_memory, get_inference_device

logger = logging.getLogger(__name__)

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def _require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add 'HF_TOKEN=hf_...' to backend/.env "
            "or export it as an environment variable before starting the server."
        )
    return token


def run_diarization(wav_path: str, output_json_path: Path) -> list[dict]:
    """
    Run speaker diarization on the full WAV file.

    Returns a list of turn dicts:
        {"speaker": "SPEAKER_00", "start": float, "end": float}
    """
    device = get_inference_device(min_vram_gb=4.0)
    diar_pipeline = None

    try:
        token = _require_hf_token()
        logger.info(f"Loading diarization model '{DIARIZATION_MODEL}' → {device} …")

        diar_pipeline = PyannotePipeline.from_pretrained(
            DIARIZATION_MODEL,
            use_auth_token=token,
        )
        diar_pipeline = diar_pipeline.to(device)

        logger.info(f"Running diarization on full file: {wav_path}")
        annotation = diar_pipeline(wav_path)

        segments: list[dict] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            })

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            f"Diarization complete: {len(segments)} turn(s) → {output_json_path}"
        )
        return segments

    except torch.cuda.OutOfMemoryError:
        logger.error(
            "VRAM OOM during diarization. "
            "Retrying on CPU — this will be significantly slower."
        )
        clear_gpu_memory()

        if diar_pipeline is not None:
            diar_pipeline = diar_pipeline.to(torch.device("cpu"))
            annotation = diar_pipeline(wav_path)
            segments = [
                {
                    "speaker": speaker,
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                }
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            ]
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            output_json_path.write_text(
                json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(
                f"Diarization (CPU fallback) complete: "
                f"{len(segments)} turn(s) → {output_json_path}"
            )
            return segments

        raise  # No pipeline to fall back on — propagate

    finally:
        del diar_pipeline
        clear_gpu_memory()
