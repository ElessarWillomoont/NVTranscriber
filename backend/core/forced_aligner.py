"""
Phase 2.6 — Atomic Token Persistence & Advanced Rendering.
           (with multilingual model routing)

Language routing
----------------
  cmn / zh  →  jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
               (pure transformers — no uroman / MMS tokenizer issues)
  other     →  torchaudio.pipelines.MMS_FA  (facebook/mms-300m)

New flow
--------
  Whisper ASR segments
        ↓
  CTC forced alignment  (backend selected by lang)
        ↓
  Micro-diarization  (latest-starting speaker at each char midpoint)
        ↓
  Tag smoothing  (majority-vote window)
        ↓
  Persist atomic tokens  →  workspace/transcripts/{task_id}_atomic_tokens.json
        ↓
  render_subtitles()  →  strict max_chars / max_duration blocks
        ↓
  aligner._build_srt  →  .srt
"""
import json
import logging
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from backend.core.vram_manager import clear_gpu_memory, get_inference_device

logger = logging.getLogger(__name__)

_SAMPLE_RATE   = 16000
_SILENCE_GAP_S = 1.0
_MIN_VRAM_GB   = 2.0

_CHINESE_LANGS = frozenset({"cmn", "zh"})
CN_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"

# ── singletons ────────────────────────────────────────────────────────────────
_cn_model     = None
_cn_processor = None

_mms_bundle   = None
_mms_model    = None
_mms_tokenizer = None
_mms_aligner   = None


# ---------------------------------------------------------------------------
# Model loading — Chinese backend
# ---------------------------------------------------------------------------

def _get_cn_components():
    """Lazy-load the Chinese Wav2Vec2-CTC model (singleton)."""
    global _cn_model, _cn_processor
    if _cn_model is not None:
        return _cn_model, _cn_processor

    from transformers import AutoModelForCTC, Wav2Vec2Processor

    device = get_inference_device(min_vram_gb=_MIN_VRAM_GB)
    logger.info(f"Loading Chinese CTC model '{CN_MODEL_ID}' → {device} …")

    _cn_processor = Wav2Vec2Processor.from_pretrained(CN_MODEL_ID)
    # use_safetensors=True bypasses torch.load(.bin) — required on torch < 2.6
    # to avoid the CVE-2025-32434 security block added in transformers 4.x.
    _cn_model = AutoModelForCTC.from_pretrained(CN_MODEL_ID, use_safetensors=True).to(device)
    _cn_model.eval()

    logger.info("Chinese CTC model ready")
    return _cn_model, _cn_processor


# ---------------------------------------------------------------------------
# Model loading — MMS_FA backend (all non-Chinese languages)
# ---------------------------------------------------------------------------

def _get_mms_fa_components():
    """Lazy-load torchaudio MMS_FA bundle (singleton)."""
    global _mms_bundle, _mms_model, _mms_tokenizer, _mms_aligner
    if _mms_model is not None:
        return _mms_model, _mms_tokenizer, _mms_aligner

    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "torchaudio is required for non-Chinese forced alignment. "
            "Run: pip install torchaudio --index-url https://download.pytorch.org/whl/cu124"
        ) from exc

    device = get_inference_device(min_vram_gb=_MIN_VRAM_GB)
    logger.info(f"Loading MMS_FA model → {device} …")

    _mms_bundle   = torchaudio.pipelines.MMS_FA
    _mms_model    = _mms_bundle.get_model(with_star=False).to(device)
    _mms_model.eval()
    _mms_tokenizer = _mms_bundle.get_tokenizer()
    _mms_aligner   = _mms_bundle.get_aligner()

    logger.info(f"MMS_FA ready (sample_rate={_mms_bundle.sample_rate} Hz)")
    return _mms_model, _mms_tokenizer, _mms_aligner


def unload_fa_model() -> None:
    """Release VRAM for both backends."""
    global _cn_model, _cn_processor
    global _mms_bundle, _mms_model, _mms_tokenizer, _mms_aligner
    _cn_model = _cn_processor = None
    _mms_bundle = _mms_model = _mms_tokenizer = _mms_aligner = None
    clear_gpu_memory()
    logger.info("Forced-alignment models unloaded")


# ---------------------------------------------------------------------------
# Chinese CTC alignment backend
# ---------------------------------------------------------------------------

def _get_timestamps_chinese(
    audio_np: np.ndarray,
    text: str,
    time_offset: float,
) -> list[dict]:
    """
    CTC forced alignment via jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn.

    Uses torchaudio.functional.forced_align + merge_tokens so no extra deps
    (uroman, sentencepiece, etc.) are needed beyond transformers + torchaudio.
    """
    try:
        import torchaudio.functional as F  # noqa — deferred for torchcodec safety
    except ImportError as exc:
        raise ImportError("torchaudio required for Chinese CTC alignment") from exc

    cn_model, cn_processor = _get_cn_components()
    device = next(cn_model.parameters()).device

    # ── Filter text to characters in the model vocabulary ─────────────────
    vocab   = cn_processor.tokenizer.get_vocab()       # {char: id, ...}
    pad_id  = cn_processor.tokenizer.pad_token_id      # CTC blank

    char_tok = [(ch, vocab[ch]) for ch in text if ch in vocab]
    if not char_tok:
        logger.warning("No characters from the text found in CN model vocab")
        return []

    chars, token_ids = zip(*char_tok)

    # ── Model forward → log-probabilities ─────────────────────────────────
    inputs = cn_processor(
        audio_np,
        sampling_rate=_SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    try:
        with torch.inference_mode():
            logits = cn_model(inputs.input_values.to(device)).logits  # [1,T,C]
    except torch.cuda.OutOfMemoryError:
        logger.error("VRAM OOM during Chinese CTC emission — re-raising")
        raise

    # forced_align expects CPU float32 log-probs
    log_probs = torch.log_softmax(logits, dim=-1).cpu().float()
    T = log_probs.shape[1]
    S = len(token_ids)

    # ── Forced alignment ──────────────────────────────────────────────────
    try:
        aligned, scores = F.forced_align(
            log_probs,
            targets=torch.tensor([list(token_ids)], dtype=torch.int32),
            input_lengths=torch.tensor([T],  dtype=torch.int32),
            target_lengths=torch.tensor([S], dtype=torch.int32),
            blank=pad_id,
        )
        token_spans = F.merge_tokens(aligned[0], scores[0], blank=pad_id)
    except Exception as exc:
        logger.error(f"Chinese CTC alignment failed: {exc}")
        return []

    # ── Frame indices → seconds ───────────────────────────────────────────
    samples_per_frame = len(audio_np) / max(T, 1)

    result: list[dict] = []
    for span, ch in zip(token_spans, chars):
        start_s = (span.start * samples_per_frame / _SAMPLE_RATE) + time_offset
        end_s   = (span.end   * samples_per_frame / _SAMPLE_RATE) + time_offset
        result.append({
            "char":  ch,
            "start": round(start_s, 4),
            "end":   round(end_s,   4),
        })

    return result


# ---------------------------------------------------------------------------
# MMS_FA alignment backend  (non-Chinese languages)
# ---------------------------------------------------------------------------

def _tokenize_mms(tokenizer, filtered: str, lang: str) -> list[int]:
    """
    Call the MMS_FA tokenizer with graceful handling across torchaudio API versions.

    torchaudio < 2.1 : tokenizer([lang], [text])  → [[ids]]
    torchaudio ≥ 2.1 : tokenizer([text], language=lang) → [[ids]]
                     OR tokenizer([text]) → [[ids]]   (language inferred)
    """
    # Preferred: new keyword-argument API
    try:
        return tokenizer([filtered], language=lang)[0]
    except TypeError:
        pass

    # Fallback: old positional API
    try:
        return tokenizer([lang], [filtered])[0]
    except Exception as exc:
        raise RuntimeError(f"MMS tokenizer call failed for lang={lang!r}: {exc}") from exc


def _get_timestamps_mms_fa(
    audio_np: np.ndarray,
    text: str,
    lang: str,
    time_offset: float,
) -> list[dict]:
    mms_model, mms_tokenizer, mms_aligner = _get_mms_fa_components()
    device = next(mms_model.parameters()).device

    # ── Vocab filter ──────────────────────────────────────────────────────
    try:
        vocab: set[str] = set(mms_tokenizer.get_labels(lang))
    except Exception:
        vocab = set()

    filtered = "".join(ch for ch in text if not vocab or ch in vocab)
    if not filtered:
        logger.warning(f"All chars filtered out for lang={lang!r}")
        return []

    # ── Emissions ─────────────────────────────────────────────────────────
    waveform = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    try:
        with torch.inference_mode():
            emission, _ = mms_model(waveform)
        emission = emission[0]
    except torch.cuda.OutOfMemoryError:
        logger.error("VRAM OOM during MMS_FA emission — re-raising")
        raise

    # ── Tokenise (API-version-aware) ──────────────────────────────────────
    try:
        token_ids = _tokenize_mms(mms_tokenizer, filtered, lang)
    except Exception as exc:
        logger.error(str(exc))
        return []

    if not token_ids:
        logger.warning("MMS tokenizer returned empty token list")
        return []

    # ── Align ─────────────────────────────────────────────────────────────
    try:
        spans, _ = mms_aligner(emission, token_ids)
    except Exception as exc:
        logger.error(f"MMS_FA aligner failed: {exc}")
        return []

    # ── Frame indices → seconds ───────────────────────────────────────────
    num_frames = emission.shape[0]
    samples_per_frame = audio_np.shape[0] / max(num_frames, 1)

    try:
        labels: tuple[str, ...] = mms_tokenizer.get_labels(lang)
    except Exception:
        labels = ()

    result: list[dict] = []
    for span in spans:
        char    = labels[span.token] if span.token < len(labels) else "?"
        start_s = (span.start * samples_per_frame / _SAMPLE_RATE) + time_offset
        end_s   = (span.end   * samples_per_frame / _SAMPLE_RATE) + time_offset
        result.append({
            "char":  char,
            "start": round(start_s, 4),
            "end":   round(end_s,   4),
        })

    return result


# ---------------------------------------------------------------------------
# Public router — dispatches to the correct backend
# ---------------------------------------------------------------------------

def get_word_timestamps(
    audio_np: np.ndarray,
    text: str,
    lang: str = "cmn",
    time_offset: float = 0.0,
) -> list[dict]:
    """
    Return per-character timestamps for the given audio/text pair.

    lang in {"cmn", "zh"} → Chinese Wav2Vec2 CTC backend
    all other lang codes  → torchaudio MMS_FA backend

    Parameters
    ----------
    audio_np    : 1-D float32 numpy array at 16 kHz
    text        : transcript already produced by the ASR model
    lang        : ISO 639-3 language code
    time_offset : seconds added to all output timestamps (global offset)

    Returns
    -------
    list of {"char": str, "start": float, "end": float}
    Empty list on any failure — caller decides how to fall back.
    """
    if not text.strip():
        return []

    if lang in _CHINESE_LANGS:
        return _get_timestamps_chinese(audio_np, text, time_offset)
    else:
        return _get_timestamps_mms_fa(audio_np, text, lang, time_offset)


# ---------------------------------------------------------------------------
# Micro-diarization — tag each character with its speaker
# ---------------------------------------------------------------------------

def assign_speakers(
    char_timestamps: list[dict],
    pyannote_segments: list[dict],
) -> list[dict]:
    """
    O(n+m) dual-pointer sweep.  Latest-starting overlapping speaker wins
    (interrupter gets the character rather than the long-tail speaker).
    """
    if not pyannote_segments:
        return [{**c, "speaker": "UNKNOWN"} for c in char_timestamps]

    diar  = sorted(pyannote_segments, key=lambda t: t["start"])
    chars = sorted(char_timestamps,   key=lambda c: c["start"])

    tagged: list[dict] = []
    ptr = 0

    for char in chars:
        mid = (char["start"] + char["end"]) / 2.0

        while ptr < len(diar) and diar[ptr]["end"] <= mid:
            ptr += 1

        overlapping: list[dict] = []
        for i in range(ptr, len(diar)):
            t = diar[i]
            if t["start"] > mid:
                break
            if t["start"] <= mid < t["end"]:
                overlapping.append(t)

        speaker = (
            max(overlapping, key=lambda t: t["start"])["speaker"]
            if overlapping else "UNKNOWN"
        )
        tagged.append({**char, "speaker": speaker})

    return tagged


# ---------------------------------------------------------------------------
# Tag smoothing — suppress single-character glitches
# ---------------------------------------------------------------------------

def smooth_speaker_tags(
    tagged_chars: list[dict],
    window: int = 5,
) -> list[dict]:
    """
    Sliding-window majority vote.  Shrinks near the edges automatically.

    Example (window=5):  [A,A,A,B,A,A]  →  [A,A,A,A,A,A]
    """
    n = len(tagged_chars)
    if n < window:
        return list(tagged_chars)

    half = window // 2
    result: list[dict] = []
    for i, char in enumerate(tagged_chars):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        majority = Counter(tagged_chars[j]["speaker"] for j in range(lo, hi)).most_common(1)[0][0]
        result.append({**char, "speaker": majority})

    return result


# ---------------------------------------------------------------------------
# reconstruct_sentences — in-memory version (kept for backward compatibility)
# ---------------------------------------------------------------------------

def reconstruct_sentences(
    tagged_chars: list[dict],
    silence_gap_s: float = _SILENCE_GAP_S,
    max_chars: int = 40,
    max_duration_s: float = 8.0,
) -> list[dict]:
    """
    Group contiguous same-speaker characters into sentences.
    Flush on: speaker change | silence gap | max_chars | max_duration.
    """
    if not tagged_chars:
        return []

    sentences: list[dict] = []
    state: dict = {
        "speaker":  tagged_chars[0]["speaker"],
        "start":    tagged_chars[0]["start"],
        "chars":    [tagged_chars[0]["char"]],
        "prev_end": tagged_chars[0]["end"],
    }

    def _flush() -> None:
        text = "".join(state["chars"]).strip()
        if text:
            sentences.append({
                "speaker": state["speaker"],
                "text":    text,
                "start":   round(state["start"],    3),
                "end":     round(state["prev_end"], 3),
            })
        state["chars"].clear()

    for char in tagged_chars[1:]:
        gap       = char["start"] - state["prev_end"]
        spk_chg   = char["speaker"] != state["speaker"]
        cur_chars = len(state["chars"])
        cur_dur   = state["prev_end"] - state["start"]

        if spk_chg or gap > silence_gap_s or cur_chars >= max_chars or cur_dur >= max_duration_s:
            _flush()
            state["speaker"] = char["speaker"]
            state["start"]   = char["start"]

        state["chars"].append(char["char"])
        state["prev_end"] = char["end"]

    _flush()
    return sentences


# ---------------------------------------------------------------------------
# render_subtitles — reads atomic token JSON, strict formatting constraints
# ---------------------------------------------------------------------------

def render_subtitles(
    atomic_tokens_path: Path | str,
    max_chars: int = 30,
    max_duration_s: float = 6.0,
    silence_gap_s: float = _SILENCE_GAP_S,
) -> list[dict]:
    """
    Produce subtitle blocks from the persisted atomic token JSON.

    Flush conditions (checked BEFORE appending each token):
    - speaker changes
    - silence gap > silence_gap_s
    - accumulated char count >= max_chars
    - token["end"] - line_start_time >= max_duration_s  (projected duration)
    """
    tokens: list[dict] = json.loads(
        Path(atomic_tokens_path).read_text(encoding="utf-8")
    )
    if not tokens:
        return []

    blocks: list[dict] = []
    state: dict = {
        "speaker":  tokens[0]["speaker"],
        "start":    tokens[0]["start"],
        "chars":    [tokens[0]["char"]],
        "prev_end": tokens[0]["end"],
    }

    def _flush() -> None:
        text = "".join(state["chars"]).strip()
        if text:
            blocks.append({
                "speaker": state["speaker"],
                "text":    text,
                "start":   round(state["start"],    3),
                "end":     round(state["prev_end"], 3),
            })
        state["chars"].clear()

    for token in tokens[1:]:
        gap           = token["start"] - state["prev_end"]
        spk_chg       = token["speaker"] != state["speaker"]
        cur_chars     = len(state["chars"])
        proj_dur      = token["end"] - state["start"]   # spec §3.3

        if spk_chg or gap > silence_gap_s or cur_chars >= max_chars or proj_dur >= max_duration_s:
            _flush()
            state["speaker"] = token["speaker"]
            state["start"]   = token["start"]

        state["chars"].append(token["char"])
        state["prev_end"] = token["end"]

    _flush()
    logger.info(
        f"render_subtitles: {len(blocks)} block(s) "
        f"(max_chars={max_chars}, max_dur={max_duration_s}s)"
    )
    return blocks


# ---------------------------------------------------------------------------
# Phase 2.6 pipeline runner
# ---------------------------------------------------------------------------

def run_forced_alignment(
    wav_path: str,
    asr_segments: list[dict],
    diar_segments: list[dict],
    atomic_tokens_path: Path,
    lang: str = "cmn",
) -> Path:
    """
    1. CTC forced alignment per ASR segment  (language-routed)
    2. Micro-diarization
    3. Majority-vote tag smoothing
    4. Persist atomic tokens to atomic_tokens_path

    Returns the path that was written.
    """
    # HOTFIX: Temporarily hide all speechbrain.* entries from sys.modules so
    # that PyTorch/Transformers' internal inspect.getmembers() calls during the
    # CTC forward pass cannot trigger SpeechBrain's broken lazy importers
    # (k2_fsa, nlp, wordemb, …) on Windows.  Modules are fully restored in the
    # finally block whether alignment succeeds or raises.
    _sb_hidden = {k: v for k, v in sys.modules.items() if k.startswith("speechbrain")}
    for k in _sb_hidden:
        del sys.modules[k]

    try:
        audio_np, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]
        if sr != _SAMPLE_RATE:
            raise ValueError(f"Expected {_SAMPLE_RATE} Hz WAV, got {sr} Hz")

        diar_sorted = sorted(diar_segments, key=lambda t: t["start"])
        all_tagged: list[dict] = []

        for seg in sorted(asr_segments, key=lambda s: s["start"]):
            text    = seg.get("text", "").strip()
            start_s = seg["start"]
            end_s   = seg["end"]

            if not text:
                continue

            # ── Guard: skip reversed / zero-length segments ───────────────────
            if end_s <= start_s + 0.05:
                logger.warning(
                    f"Skipping reversed/zero-length segment: "
                    f"{start_s:.3f}s → {end_s:.3f}s  (text: {text[:30]!r})"
                )
                continue

            logger.info(f"FA segment {start_s:.2f}s–{end_s:.2f}s: {len(text)} chars")

            slice_np = audio_np[int(start_s * _SAMPLE_RATE): int(end_s * _SAMPLE_RATE)]
            char_ts  = get_word_timestamps(slice_np, text, lang=lang, time_offset=start_s)

            if char_ts:
                all_tagged.extend(assign_speakers(char_ts, diar_segments))
            else:
                logger.warning(
                    f"CTC returned no chars for {start_s:.2f}s — "
                    "falling back to segment-level speaker lookup"
                )
                mid = (start_s + end_s) / 2.0
                all_tagged.append({
                    "char":    text,
                    "start":   start_s,
                    "end":     end_s,
                    "speaker": _midpoint_speaker(mid, diar_sorted),
                })

        smoothed = smooth_speaker_tags(all_tagged)

        atomic_tokens_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_tokens_path.write_text(
            json.dumps(smoothed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            f"Atomic tokens → {atomic_tokens_path} ({len(smoothed)} token(s))"
        )
        return atomic_tokens_path

    finally:
        # Restore all speechbrain modules regardless of success or failure
        sys.modules.update(_sb_hidden)


def _midpoint_speaker(t: float, diar_sorted: list[dict]) -> str:
    for turn in diar_sorted:
        if turn["start"] > t:
            break
        if turn["start"] <= t < turn["end"]:
            return turn["speaker"]
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Public entry point (with IoU fallback)
# ---------------------------------------------------------------------------

def forced_align_and_write_srt(
    wav_path: str,
    asr_segments: list[dict],
    diar_segments: list[dict],
    output_srt_path: Path,
    atomic_tokens_path: Path,
    lang: str = "cmn",
    max_chars: int = 30,
    max_duration_s: float = 6.0,
) -> Path:
    """
    Phase 2.6 end-to-end entry point.
    Falls back to IoU alignment if CTC fails for any reason.
    """
    from backend.core.aligner import _build_srt, align_and_write_srt

    output_srt_path.parent.mkdir(parents=True, exist_ok=True)

    # HOTFIX: Prevent SpeechBrain's lazy importer from crashing the pipeline on Windows
    # when PyTorch/Transformers inspects modules during the CTC forward pass.
    if "speechbrain" in sys.modules and "speechbrain.integrations.k2_fsa" not in sys.modules:
        sys.modules["speechbrain.integrations.k2_fsa"] = types.ModuleType("speechbrain.integrations.k2_fsa")

    try:
        tokens_path = run_forced_alignment(
            wav_path, asr_segments, diar_segments, atomic_tokens_path, lang
        )
        unload_fa_model()

        blocks = render_subtitles(tokens_path, max_chars=max_chars, max_duration_s=max_duration_s)
        output_srt_path.write_text(_build_srt(blocks), encoding="utf-8")
        logger.info(f"Phase 2.6 SRT written → {output_srt_path}")

    except Exception as exc:
        logger.warning(
            f"Phase 2.6 forced alignment failed "
            f"({type(exc).__name__}: {exc}) — falling back to IoU alignment"
        )
        unload_fa_model()
        align_and_write_srt(asr_segments, diar_segments, output_srt_path)

    return output_srt_path
