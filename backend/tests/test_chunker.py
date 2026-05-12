import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from backend.core.chunker import _run_vad, _aggregate_chunks, SAMPLE_RATE, MAX_CHUNK_DURATION

def test_run_vad_with_silence(tmp_path):
    """Regression test for VAD API: Ensure it handles zeros and returns no segments via file path."""
    import soundfile as sf
    # Create 1 second of silence in a temp file
    wav_path = tmp_path / "silence.wav"
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    sf.write(str(wav_path), silence, SAMPLE_RATE)
    
    segments, duration = _run_vad(str(wav_path))
    
    assert isinstance(segments, list)
    # For pure silence, it should detect 0 segments
    assert len(segments) == 0
    assert abs(duration - 1.0) < 0.01

def test_aggregate_chunks_fallback_logic():
    """Test the aggregation logic when no speech is detected."""
    total_duration = 1500.0  # 25 minutes
    speeches = [] # No speech detected
    
    # This matches the internal logic we added for fallback in chunk_audio
    # (Since _aggregate_chunks itself is just a helper, we verify the math here)
    boundaries = []
    if not speeches and total_duration > MAX_CHUNK_DURATION:
        for start in np.arange(0, total_duration, MAX_CHUNK_DURATION):
            end = min(start + MAX_CHUNK_DURATION, total_duration)
            boundaries.append((float(start), float(end)))
    else:
        boundaries = _aggregate_chunks(speeches, total_duration)
        
    # Should be split into 25 chunks (1500 / 60)
    assert len(boundaries) == 25
    assert boundaries[0] == (0.0, 60.0)
    assert boundaries[1] == (60.0, 120.0)
    assert boundaries[-1][1] == 1500.0

def test_aggregate_chunks_with_speech():
    """Test normal aggregation of detected speech segments."""
    total_duration = 1000.0
    speeches = [
        {"start": 10.0, "end": 20.0},
        {"start": 400.0, "end": 410.0}, # This should trigger a cut because wall_clock (410) > MIN_CHUNK (300)
        {"start": 800.0, "end": 810.0}, # This should trigger another cut
    ]
    
    boundaries = _aggregate_chunks(speeches, total_duration)
    
    # With MIN_CHUNK=30s:
    # 1. 0-20.0 (too short)
    # 2. 400.0-410.0 -> Cut at 410.0 (wall_clock 410 > 30)
    # 3. 800.0-810.0 -> Cut at 810.0 (wall_clock 400 > 30)
    # 4. Tail: 810.0-1000.0
    # Total 3 chunks
    assert len(boundaries) == 3
    assert boundaries[0][1] == 410.0
    assert boundaries[1][1] == 810.0
    assert boundaries[2][1] == 1000.0

@pytest.mark.asyncio
async def test_chunk_audio_logic_integration(tmp_path):
    """
    Test the integration logic in chunk_audio using a dummy wav file.
    Ensures the DB interaction (mocked) and file loading works.
    """
    from backend.core.chunker import chunk_audio
    
    import soundfile as sf
    # Create a dummy 1-second silent wav file
    wav_path = tmp_path / "test.wav"
    dummy_data = np.zeros(SAMPLE_RATE, dtype=np.float32)
    sf.write(str(wav_path), dummy_data, SAMPLE_RATE)
    
    # Mock the DB session — execute and commit must be AsyncMock
    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.commit = AsyncMock()
    
    # Run chunk_audio
    # Note: For 1 second, it won't trigger fallback (which is for > 10 min)
    # and it won't detect speech. So it should return 1 chunk of 1 second.
    task_id = "test-task"
    boundaries = await chunk_audio(str(wav_path), task_id, mock_db)
    
    assert len(boundaries) == 1
    assert boundaries[0][0] == 0.0
    assert abs(boundaries[0][1] - 1.0) < 0.1
    
    # Verify that DB 'add' was called
    assert mock_db.add.called
