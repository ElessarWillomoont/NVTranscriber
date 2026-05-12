"""
Tests for backend/core/refiner.py  (Phase 2.7 Focus Mode).

All GPU-heavy operations (Whisper ASR, CTC aligner) are mocked so the suite
runs entirely on CPU in milliseconds without touching HuggingFace model weights.
"""
import json

import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.core.refiner import aggregate_sessions, slice_audio_session, run_focus_mode
from backend.core.forced_aligner import get_word_timestamps

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tokens(speaker_seq: list[tuple[str, float, float]]) -> list[dict]:
    """Build atomic-token dicts from (speaker, start, end) triples."""
    return [
        {"char": "X", "start": s, "end": e, "speaker": spk}
        for spk, s, e in speaker_seq
    ]


# ---------------------------------------------------------------------------
# Test 1: Session Aggregation
# ---------------------------------------------------------------------------

class TestAggregateSessions:

    def test_speaker_change_triggers_new_session(self):
        """An abrupt speaker change must split into separate sessions."""
        tokens = _make_tokens([
            ("SPEAKER_00", 0.0, 0.3),
            ("SPEAKER_00", 0.3, 0.6),
            ("SPEAKER_00", 0.6, 0.9),
            ("SPEAKER_01", 0.9, 1.2),   # ← speaker change
            ("SPEAKER_01", 1.2, 1.5),
        ])
        sessions = aggregate_sessions(tokens, session_gap_s=2.0)

        assert len(sessions) == 2

        s0 = sessions[0]
        assert s0["speaker"] == "SPEAKER_00"
        assert s0["start"]   == pytest.approx(0.0)
        assert s0["end"]     == pytest.approx(0.9)   # last token's .end

        s1 = sessions[1]
        assert s1["speaker"] == "SPEAKER_01"
        assert s1["start"]   == pytest.approx(0.9)
        assert s1["end"]     == pytest.approx(1.5)

    def test_silence_gap_triggers_new_session(self):
        """A silence gap > session_gap_s must split the same speaker's speech."""
        tokens = _make_tokens([
            ("SPEAKER_00", 0.0, 0.5),
            ("SPEAKER_00", 0.5, 1.0),
            # 2.5 s gap — well above the 2.0 s threshold
            ("SPEAKER_00", 3.5, 4.0),
            ("SPEAKER_00", 4.0, 4.5),
        ])
        sessions = aggregate_sessions(tokens, session_gap_s=2.0)

        assert len(sessions) == 2
        assert sessions[0]["end"]   == pytest.approx(1.0)
        assert sessions[1]["start"] == pytest.approx(3.5)
        assert sessions[1]["end"]   == pytest.approx(4.5)

    def test_combined_change_and_gap(self):
        """Three sessions: speaker change, then same-speaker silence gap."""
        tokens = _make_tokens([
            ("SPEAKER_00", 0.0,  0.3),
            ("SPEAKER_00", 0.3,  0.6),
            ("SPEAKER_00", 0.6,  0.9),
            ("SPEAKER_01", 0.9,  1.2),   # speaker change → new session
            ("SPEAKER_01", 1.2,  1.5),
            ("SPEAKER_00", 4.0,  4.4),   # speaker change + silence (2.5 s gap)
            ("SPEAKER_00", 4.4,  4.8),
        ])
        sessions = aggregate_sessions(tokens, session_gap_s=2.0)

        assert len(sessions) == 3

        assert sessions[0]["speaker"] == "SPEAKER_00"
        assert sessions[0]["start"]   == pytest.approx(0.0)
        assert sessions[0]["end"]     == pytest.approx(0.9)

        assert sessions[1]["speaker"] == "SPEAKER_01"
        assert sessions[1]["start"]   == pytest.approx(0.9)
        assert sessions[1]["end"]     == pytest.approx(1.5)

        assert sessions[2]["speaker"] == "SPEAKER_00"
        assert sessions[2]["start"]   == pytest.approx(4.0)
        assert sessions[2]["end"]     == pytest.approx(4.8)

    def test_gap_exactly_at_threshold_does_not_split(self):
        """
        Gap == session_gap_s must NOT trigger a new session.
        The condition is strictly greater-than, not greater-than-or-equal.
        """
        tokens = _make_tokens([
            ("S", 0.0, 0.5),
            ("S", 2.5, 3.0),   # gap = 2.5 - 0.5 = 2.0, exactly at threshold
        ])
        sessions = aggregate_sessions(tokens, session_gap_s=2.0)

        assert len(sessions) == 1
        assert sessions[0]["start"] == pytest.approx(0.0)
        assert sessions[0]["end"]   == pytest.approx(3.0)

    def test_no_breaks_yields_one_session(self):
        """Continuous single-speaker audio with no gaps → exactly one session."""
        tokens = _make_tokens([("S", i * 0.1, (i + 1) * 0.1) for i in range(10)])
        sessions = aggregate_sessions(tokens, session_gap_s=2.0)

        assert len(sessions) == 1
        assert sessions[0]["start"] == pytest.approx(0.0)
        assert sessions[0]["end"]   == pytest.approx(1.0)

    def test_empty_input(self):
        assert aggregate_sessions([]) == []

    def test_single_token_yields_one_session(self):
        tokens = _make_tokens([("S", 5.0, 5.5)])
        sessions = aggregate_sessions(tokens)
        assert len(sessions) == 1
        assert sessions[0]["speaker"] == "S"


# ---------------------------------------------------------------------------
# Test 2: Audio Slicing & Bounds Padding
# ---------------------------------------------------------------------------

class TestSliceAudioSession:

    def test_normal_slice_with_padding(self):
        """
        Session [2.0 s, 4.0 s] with 0.2 s padding → slice of 2.4 s.
        Expected length = (2.0 + 2 × 0.2) × 16000 = 38 400 samples.
        """
        audio_np = np.zeros(int(10.0 * SAMPLE_RATE), dtype=np.float32)

        sl, offset_s = slice_audio_session(audio_np, 2.0, 4.0, padding_s=0.2)

        expected_len = int((2.0 + 2 * 0.2) * SAMPLE_RATE)   # 38 400
        assert len(sl) == expected_len
        assert offset_s == pytest.approx(1.8)                 # 2.0 - 0.2

    def test_left_edge_clamps_to_zero(self):
        """
        Session starting at 0.0 s: left padding clamps to 0, no negative index.
        actual_start_s = max(0.0, 0.0 - 0.2) = 0.0
        """
        audio_np = np.zeros(int(10.0 * SAMPLE_RATE), dtype=np.float32)

        sl, offset_s = slice_audio_session(audio_np, 0.0, 1.0, padding_s=0.2)

        assert offset_s == pytest.approx(0.0)
        # Only right-padding applies: 1.0 + 0.2 = 1.2 s
        assert len(sl) == int(1.2 * SAMPLE_RATE)

    def test_right_edge_clamps_to_array_length(self):
        """
        Session ending at file end: right padding clamps to len(audio), no overrun.
        actual_end_s = min(10.0, 10.0 + 0.2) = 10.0
        """
        audio_np = np.zeros(int(10.0 * SAMPLE_RATE), dtype=np.float32)

        sl, offset_s = slice_audio_session(audio_np, 9.0, 10.0, padding_s=0.2)

        # Only left-padding applies: 9.0 - 0.2 = 8.8 s
        assert offset_s == pytest.approx(8.8)
        assert len(sl) == int(1.2 * SAMPLE_RATE)

    def test_both_edges_clamped(self):
        """
        Single-sample audio: both sides clamp; slice = full array.
        """
        audio_np = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)

        sl, offset_s = slice_audio_session(audio_np, 0.0, 1.0, padding_s=0.5)

        assert offset_s == pytest.approx(0.0)
        assert len(sl) == len(audio_np)   # right side also clamped to 1.0 s

    def test_slice_is_numpy_view_not_copy(self):
        """
        The returned slice must share memory with the original array (zero-copy).
        Modifying the source should be visible in the slice.
        """
        audio_np = np.ones(int(5.0 * SAMPLE_RATE), dtype=np.float32)
        sl, _ = slice_audio_session(audio_np, 1.0, 2.0, padding_s=0.1)

        assert np.shares_memory(audio_np, sl)


# ---------------------------------------------------------------------------
# Test 3: Global Timestamp Re-mapping via time_offset
# ---------------------------------------------------------------------------

class TestGlobalTimestampOffset:

    @patch("backend.core.forced_aligner._get_mms_fa_components")
    def test_local_to_global_offset_applied(self, mock_get_components):
        """
        Verify time_offset maps local CTC frame timestamps to global file time.

        Setup
        -----
        audio_np   : 1 s at 16 kHz  →  16 000 samples
        num_frames : 10              →  samples_per_frame = 1 600
        span.start : frame 5        →  local = 5 × 1600 / 16000 = 0.5 s
        span.end   : frame 8        →  local = 8 × 1600 / 16000 = 0.8 s
        time_offset : 9.8 s (session [10.0, 11.0] with 0.2 s padding)

        Expected
        --------
        result[0]["start"] = 9.8 + 0.5 = 10.3 s
        result[0]["end"]   = 9.8 + 0.8 = 10.6 s
        result[1]["start"] = 9.8 + 0.8 = 10.6 s
        result[1]["end"]   = 9.8 + 1.0 = 10.8 s
        """
        mock_model = MagicMock()
        # Return emission of shape [1, 10, 30]; code slices [0] → [10, 30]
        mock_model.return_value = (torch.randn(1, 10, 30), None)
        mock_model.parameters.return_value = iter(
            [MagicMock(device=torch.device("cpu"))]
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = [[1, 2]]                        # token IDs
        mock_tokenizer.get_labels.return_value = ("<pad>", "H", "e")  # vocab

        class _Span:
            def __init__(self, token, start, end):
                self.token = token
                self.start = start
                self.end   = end

        mock_aligner = MagicMock()
        mock_aligner.return_value = (
            [_Span(1, 5, 8), _Span(2, 8, 10)],
            0.9,
        )
        mock_get_components.return_value = (mock_model, mock_tokenizer, mock_aligner)

        audio_np    = np.zeros(SAMPLE_RATE, dtype=np.float32)   # exactly 1 s
        time_offset = 9.8   # actual_start_s = 10.0 - 0.2 (padded)

        result = get_word_timestamps(audio_np, "He", lang="eng", time_offset=time_offset)

        assert len(result) == 2

        assert result[0]["char"]  == "H"
        assert result[0]["start"] == pytest.approx(10.3, abs=1e-3)
        assert result[0]["end"]   == pytest.approx(10.6, abs=1e-3)

        assert result[1]["char"]  == "e"
        assert result[1]["start"] == pytest.approx(10.6, abs=1e-3)
        assert result[1]["end"]   == pytest.approx(10.8, abs=1e-3)

    @patch("backend.core.forced_aligner._get_mms_fa_components")
    def test_zero_offset_preserves_local_timestamps(self, mock_get_components):
        """With time_offset=0.0 the output timestamps equal the raw local times."""
        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(1, 10, 30), None)
        mock_model.parameters.return_value = iter(
            [MagicMock(device=torch.device("cpu"))]
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = [[1]]
        mock_tokenizer.get_labels.return_value = ("<pad>", "A")

        class _Span:
            def __init__(self, token, start, end):
                self.token = token; self.start = start; self.end = end

        mock_aligner = MagicMock()
        mock_aligner.return_value = ([_Span(1, 0, 5)], 1.0)
        mock_get_components.return_value = (mock_model, mock_tokenizer, mock_aligner)

        audio_np = np.zeros(SAMPLE_RATE, dtype=np.float32)
        result   = get_word_timestamps(audio_np, "A", lang="eng", time_offset=0.0)

        assert len(result) == 1
        # samples_per_frame = 16000/10 = 1600; span(0,5) → local (0.0s, 0.5s)
        assert result[0]["start"] == pytest.approx(0.0,  abs=1e-3)
        assert result[0]["end"]   == pytest.approx(0.5,  abs=1e-3)


# ---------------------------------------------------------------------------
# Test 4: run_focus_mode end-to-end (GPU ops mocked; no CTC)
# ---------------------------------------------------------------------------

class TestRunFocusMode:

    def _long_tokens(self) -> list[dict]:
        """Two well-separated sessions, each ≥ _MIN_DURATION = 0.3 s."""
        return [
            # SPEAKER_00: 1.0 → 1.5 s
            {"char": "A", "start": 1.0, "end": 1.2, "speaker": "SPEAKER_00"},
            {"char": "B", "start": 1.2, "end": 1.5, "speaker": "SPEAKER_00"},
            # SPEAKER_01: 2.0 → 2.5 s  (speaker change triggers new session)
            {"char": "C", "start": 2.0, "end": 2.3, "speaker": "SPEAKER_01"},
            {"char": "D", "start": 2.3, "end": 2.5, "speaker": "SPEAKER_01"},
        ]

    @patch("backend.core.refiner._build_pipeline")
    @patch("backend.core.refiner.sf.read")
    def test_srt_file_is_created(self, mock_sf_read, mock_build_pipe, tmp_path):
        """run_focus_mode must always write an SRT file on success."""
        audio_np = np.zeros(int(10.0 * SAMPLE_RATE), dtype=np.float32)
        mock_sf_read.return_value = (audio_np, SAMPLE_RATE)

        mock_pipe = MagicMock()
        mock_pipe.return_value = {"chunks": [{"text": "Hello", "timestamp": (0.0, 1.0)}]}
        mock_build_pipe.return_value = mock_pipe

        tokens_path = tmp_path / "tokens.json"
        tokens_path.write_text(json.dumps(self._long_tokens()), encoding="utf-8")

        run_focus_mode("dummy.wav", tokens_path, tmp_path / "focus.srt")

        assert (tmp_path / "focus.srt").exists()

    @patch("backend.core.refiner._build_pipeline")
    @patch("backend.core.refiner.sf.read")
    def test_asr_called_once_per_session(self, mock_sf_read, mock_build_pipe, tmp_path):
        """Whisper must be invoked exactly once per valid speaker session."""
        audio_np = np.zeros(int(10.0 * SAMPLE_RATE), dtype=np.float32)
        mock_sf_read.return_value = (audio_np, SAMPLE_RATE)

        mock_pipe = MagicMock()
        mock_pipe.return_value = {"chunks": [{"text": "X", "timestamp": (0.0, 0.5)}]}
        mock_build_pipe.return_value = mock_pipe

        tokens_path = tmp_path / "tokens.json"
        tokens_path.write_text(json.dumps(self._long_tokens()), encoding="utf-8")

        run_focus_mode("dummy.wav", tokens_path, tmp_path / "focus.srt")

        assert mock_pipe.call_count == 2   # two valid sessions

    @patch("backend.core.refiner._build_pipeline")
    @patch("backend.core.refiner.sf.read")
    def test_whisper_chunks_timestamps_mapped_to_global(
        self, mock_sf_read, mock_build_pipe, tmp_path
    ):
        """
        Verify that Whisper chunk timestamps are offset by actual_start_s
        to produce correct global timestamps in the SRT.

        Setup
        -----
        Session: SPEAKER_00, start=1.0 s, end=1.5 s
        Padding: 0.2 s  →  actual_start_s = max(0, 1.0 - 0.2) = 0.8 s

        Whisper chunk 1: timestamp=(0.1, 0.8)
          → global start = 0.1 + 0.8 = 0.9 s  →  "00:00:00,900"
          → global end   = 0.8 + 0.8 = 1.6 s  →  "00:00:01,600"

        Whisper chunk 2: timestamp=(0.9, 1.5)
          → global start = 0.9 + 0.8 = 1.7 s  →  "00:00:01,700"
          → global end   = 1.5 + 0.8 = 2.3 s  →  "00:00:02,300"
        """
        audio_np = np.zeros(int(10.0 * SAMPLE_RATE), dtype=np.float32)
        mock_sf_read.return_value = (audio_np, SAMPLE_RATE)

        mock_pipe = MagicMock()
        mock_pipe.return_value = {
            "chunks": [
                {"text": "Hello world",  "timestamp": (0.1, 0.8)},
                {"text": "how are you",  "timestamp": (0.9, 1.5)},
            ]
        }
        mock_build_pipe.return_value = mock_pipe

        tokens = [
            {"char": "A", "start": 1.0, "end": 1.2, "speaker": "SPEAKER_00"},
            {"char": "B", "start": 1.2, "end": 1.5, "speaker": "SPEAKER_00"},
        ]
        tokens_path = tmp_path / "tokens.json"
        tokens_path.write_text(json.dumps(tokens), encoding="utf-8")

        output_srt = tmp_path / "focus.srt"
        run_focus_mode("dummy.wav", tokens_path, output_srt)

        srt = output_srt.read_text(encoding="utf-8")
        assert "Hello world" in srt
        assert "how are you" in srt
        assert "00:00:00,900" in srt   # chunk 1 global start
        assert "00:00:01,600" in srt   # chunk 1 global end
        assert "00:00:01,700" in srt   # chunk 2 global start
        assert "00:00:02,300" in srt   # chunk 2 global end

    @patch("backend.core.refiner._build_pipeline")
    @patch("backend.core.refiner.sf.read")
    def test_speaker_label_from_session_in_srt(
        self, mock_sf_read, mock_build_pipe, tmp_path
    ):
        """
        The speaker label must come from the session (atomic tokens), not from
        any model.  The SRT must carry the exact label stored in the token JSON.
        """
        audio_np = np.zeros(int(5.0 * SAMPLE_RATE), dtype=np.float32)
        mock_sf_read.return_value = (audio_np, SAMPLE_RATE)

        mock_pipe = MagicMock()
        mock_pipe.return_value = {"chunks": [{"text": "Test", "timestamp": (0.0, 0.5)}]}
        mock_build_pipe.return_value = mock_pipe

        tokens = [
            {"char": "T", "start": 1.0, "end": 1.2, "speaker": "SPEAKER_99"},
            {"char": "e", "start": 1.2, "end": 1.5, "speaker": "SPEAKER_99"},
        ]
        tokens_path = tmp_path / "tokens.json"
        tokens_path.write_text(json.dumps(tokens), encoding="utf-8")

        output_srt = tmp_path / "focus.srt"
        run_focus_mode("dummy.wav", tokens_path, output_srt)

        srt_content = output_srt.read_text(encoding="utf-8")
        assert "SPEAKER_99" in srt_content, (
            f"Expected SPEAKER_99 in SRT.\nGot:\n{srt_content}"
        )

    @patch("backend.core.refiner._build_pipeline")
    @patch("backend.core.refiner.sf.read")
    def test_empty_atomic_tokens_writes_empty_srt(
        self, mock_sf_read, mock_build_pipe, tmp_path
    ):
        """Graceful handling when atomic_tokens.json is empty."""
        mock_sf_read.return_value = (np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        mock_build_pipe.return_value = MagicMock()

        tokens_path = tmp_path / "tokens.json"
        tokens_path.write_text("[]", encoding="utf-8")

        output_srt = tmp_path / "focus.srt"
        run_focus_mode("dummy.wav", tokens_path, output_srt)

        assert output_srt.exists()
        assert output_srt.read_text(encoding="utf-8") == ""
        mock_build_pipe.assert_not_called()   # Whisper must not load for empty input
