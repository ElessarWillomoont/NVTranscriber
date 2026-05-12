"""
Aligner — assigns speaker labels to ASR segments via IoU overlap, then
writes a well-formed SubRip (.srt) file.

IoU definition used here:
    IoU(A, B) = |A ∩ B| / |A ∪ B|

For speaker assignment we choose the diarization turn that maximises IoU
against the ASR segment.  When multiple turns from the same speaker are
interleaved with other speakers, we sum their individual IoU scores before
comparing — this correctly handles short speaker-change artefacts inside a
longer ASR segment.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Intersection-over-Union for two closed time intervals [a_start, a_end]."""
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if intersection == 0.0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union


def _assign_speakers(
    asr_segments: list[dict],
    diar_segments: list[dict],
) -> list[dict]:
    """
    For each ASR segment find the speaker with the highest *total* IoU across
    all their diarization turns.  Returns a copy of asr_segments with two
    extra keys: "speaker" and "iou" (the winning score).

    Uses a dual-pointer sliding window (O(n+m)) instead of O(n*m) brute force.
    Both input lists are assumed unsorted; we sort by start time internally and
    restore the original order via index mapping.
    """
    if not diar_segments:
        return [{**seg, "speaker": "UNKNOWN", "iou": 0.0} for seg in asr_segments]

    # Process ASR segments in start-time order so the diar pointer only moves forward
    order = sorted(range(len(asr_segments)), key=lambda i: asr_segments[i]["start"])
    diar_sorted = sorted(diar_segments, key=lambda t: t["start"])

    results: list[dict | None] = [None] * len(asr_segments)
    lo = 0  # first diar turn that might still overlap future ASR segments

    for idx in order:
        seg = asr_segments[idx]
        a_start, a_end = seg["start"], seg["end"]

        # Advance lo past turns that end at or before this segment starts
        while lo < len(diar_sorted) and diar_sorted[lo]["end"] <= a_start:
            lo += 1

        speaker_scores: dict[str, float] = {}
        hi = lo
        while hi < len(diar_sorted) and diar_sorted[hi]["start"] < a_end:
            turn = diar_sorted[hi]
            score = _interval_iou(a_start, a_end, turn["start"], turn["end"])
            if score > 0.0:
                speaker_scores[turn["speaker"]] = (
                    speaker_scores.get(turn["speaker"], 0.0) + score
                )
            hi += 1

        if speaker_scores:
            best_speaker = max(speaker_scores, key=lambda s: speaker_scores[s])
            best_iou = speaker_scores[best_speaker]
        else:
            best_speaker = "UNKNOWN"
            best_iou = 0.0

        results[idx] = {**seg, "speaker": best_speaker, "iou": round(best_iou, 4)}

    return results  # type: ignore[return-value]  # all slots filled by construction


# ---------------------------------------------------------------------------
# SRT formatting
# ---------------------------------------------------------------------------

def _to_srt_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to SRT timestamp HH:MM:SS,mmm."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1.0) * 1000))
    # Round-up milliseconds can push seconds to 1000 ms
    if ms >= 1000:
        ms -= 1000
        s += 1
    if s >= 60:
        s -= 60
        m += 1
    if m >= 60:
        m -= 60
        h += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_srt(aligned: list[dict]) -> str:
    """Render aligned segments as a SubRip string."""
    blocks: list[str] = []
    for idx, seg in enumerate(aligned, start=1):
        t_start = _to_srt_timestamp(seg["start"])
        t_end = _to_srt_timestamp(seg["end"])
        speaker = seg["speaker"]
        text = seg["text"].strip()
        blocks.append(f"{idx}\n{t_start} --> {t_end}\n[{speaker}]: {text}\n")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def align_and_write_srt(
    asr_segments: list[dict],
    diar_segments: list[dict],
    output_srt_path: Path,
) -> Path:
    """
    Align ASR text segments with diarization turns and write an SRT file.

    Parameters
    ----------
    asr_segments:
        Output of run_asr() — list of {"chunk_id", "start", "end", "text"}.
    diar_segments:
        Output of run_diarization() — list of {"speaker", "start", "end"}.
    output_srt_path:
        Destination .srt file (parent dirs are created if missing).

    Returns
    -------
    The resolved output_srt_path.
    """
    output_srt_path.parent.mkdir(parents=True, exist_ok=True)

    if not asr_segments:
        logger.warning("No ASR segments — writing empty SRT file")
        output_srt_path.write_text("", encoding="utf-8")
        return output_srt_path

    if not diar_segments:
        logger.warning(
            "No diarization segments available — "
            "all subtitles will be labelled UNKNOWN"
        )

    aligned = _assign_speakers(asr_segments, diar_segments)
    srt_content = _build_srt(aligned)
    output_srt_path.write_text(srt_content, encoding="utf-8")

    unknown_count = sum(1 for s in aligned if s["speaker"] == "UNKNOWN")
    logger.info(
        f"SRT written: {output_srt_path} — "
        f"{len(aligned)} subtitle(s), {unknown_count} with unknown speaker"
    )
    return output_srt_path
