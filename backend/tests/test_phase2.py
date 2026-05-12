import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.core.aligner import _interval_iou, _assign_speakers, _to_srt_timestamp, _build_srt
from backend.core.vram_manager import get_inference_device, check_vram


# ---------------------------------------------------------------------------
# 1. VRAM Manager Tests
# ---------------------------------------------------------------------------

def test_vram_threshold_logic():
    """Test that get_inference_device returns CPU when VRAM is insufficient."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(1 * 1024**3, 8 * 1024**3)):
        # 1GB free, 4GB required
        device = get_inference_device(min_vram_gb=4.0)
        assert device.type == "cpu"

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(6 * 1024**3, 8 * 1024**3)):
        # 6GB free, 4GB required
        device = get_inference_device(min_vram_gb=4.0)
        assert device.type == "cuda"


# ---------------------------------------------------------------------------
# 2. Aligner Tests (The "Black Magic" IoU & Alignment)
# ---------------------------------------------------------------------------

def test_interval_iou_cases():
    """Verify IoU calculation for various overlap scenarios."""
    # Perfect overlap
    assert _interval_iou(0, 10, 0, 10) == 1.0
    # No overlap
    assert _interval_iou(0, 10, 11, 20) == 0.0
    # Partial overlap (50%)
    # Intersection: [5, 10] = 5
    # Union: [0, 15] = 15
    # IoU = 5/15 = 0.3333
    assert round(_interval_iou(0, 10, 5, 15), 4) == 0.3333
    # B inside A
    # Intersection: 2, Union: 10, IoU: 0.2
    assert _interval_iou(0, 10, 4, 6) == 0.2


def test_speaker_assignment():
    """Test that ASR segments are assigned to the speaker with most overlap."""
    asr_segments = [
        {"start": 1.0, "end": 3.0, "text": "Hello world"}
    ]
    diar_segments = [
        {"speaker": "SPEAKER_01", "start": 0.0, "end": 2.1}, # 1.1s overlap
        {"speaker": "SPEAKER_02", "start": 2.1, "end": 4.0}, # 0.9s overlap
    ]
    
    aligned = _assign_speakers(asr_segments, diar_segments)
    assert aligned[0]["speaker"] == "SPEAKER_01"
    assert aligned[0]["text"] == "Hello world"


def test_srt_timestamp_formatting():
    """Verify SRT timestamp conversion (HH:MM:SS,mmm)."""
    assert _to_srt_timestamp(0.0) == "00:00:00,000"
    assert _to_srt_timestamp(61.5) == "00:01:01,500"
    assert _to_srt_timestamp(3661.005) == "01:01:01,005"
    # Test rounding edge case
    assert _to_srt_timestamp(1.9999) == "00:00:02,000"
    
def test_srt_timestamp_rollover():
    """Ensure timestamps roll over correctly from 59.999s to 1m0s."""
    # 59.9996 -> rounds to 60.000ms -> should be 01:00,000
    assert _to_srt_timestamp(59.9996) == "00:01:00,000"
    # 3599.9996 -> rounds to 3600.000ms -> should be 01:00:00,000
    assert _to_srt_timestamp(3599.9996) == "01:00:00,000"


def test_srt_content_generation():
    """Verify final SRT string structure."""
    aligned = [
        {"start": 1.0, "end": 2.5, "speaker": "A", "text": "Hi"},
        {"start": 3.0, "end": 4.0, "speaker": "B", "text": "Bye"}
    ]
    srt = _build_srt(aligned)
    assert "1\n00:00:01,000 --> 00:00:02,500\n[A]: Hi" in srt
    assert "2\n00:00:03,000 --> 00:00:04,000\n[B]: Bye" in srt


# ---------------------------------------------------------------------------
# 3. Engine Interface Tests (Mocked)
# ---------------------------------------------------------------------------

@patch("backend.core.asr_engine.sf.info")
@patch("backend.core.asr_engine._build_pipeline")
@patch("backend.core.asr_engine.sf.read")
def test_run_asr_calls_pipeline(mock_sf_read, mock_build_pipe, mock_sf_info, tmp_path):
    """Ensure run_asr processes chunks and outputs JSON correctly."""
    from backend.core.asr_engine import run_asr

    # Mock sf.info
    mock_sf_info.return_value.frames = 16000
    mock_sf_info.return_value.samplerate = 16000
    
    # soundfile.read returns (np.ndarray, sample_rate)
    mock_sf_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
    
    # Mock ASR pipeline response
    mock_pipe = MagicMock()
    mock_pipe.return_value = {"chunks": [{"timestamp": (0.0, 1.0), "text": "Test text"}]}
    mock_build_pipe.return_value = mock_pipe
    
    # Dummy chunks
    class MockChunk:
        def __init__(self, id, seq, start, end):
            self.id = id
            self.sequence_number = seq
            self.start_offset = start
            self.end_offset = end
            
    chunks = [MockChunk(1, 0, 0.0, 1.0)]
    output_json = tmp_path / "asr.json"
    
    results = run_asr("dummy.wav", chunks, output_json)
    
    assert len(results) == 1
    assert results[0]["text"] == "Test text"
    assert output_json.exists()
    assert "Test text" in output_json.read_text()


@patch("backend.core.diarization_engine.PyannotePipeline.from_pretrained")
@patch("backend.core.diarization_engine._require_hf_token", return_value="fake_token")
def test_run_diarization_output(mock_token, mock_pyannote, tmp_path):
    """Ensure run_diarization handles pyannote output and saves JSON."""
    from backend.core.diarization_engine import run_diarization
    
    # Mock Annotation object
    mock_annotation = MagicMock()
    mock_turn = MagicMock(start=0.5, end=1.5)
    mock_annotation.itertracks.return_value = [(mock_turn, None, "SPEAKER_X")]
    
    mock_pipeline_inst = MagicMock()
    mock_pipeline_inst.to.return_value = mock_pipeline_inst  # Critical: .to() returns self
    mock_pipeline_inst.return_value = mock_annotation
    mock_pyannote.return_value = mock_pipeline_inst
    
    output_json = tmp_path / "diar.json"
    results = run_diarization("dummy.wav", output_json)
    
    assert len(results) == 1
    assert results[0]["speaker"] == "SPEAKER_X"
    assert results[0]["start"] == 0.5
    assert output_json.exists()
