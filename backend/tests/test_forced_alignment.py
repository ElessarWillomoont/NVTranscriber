import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Note: The implementation backend/core/forced_aligner.py will be created by another agent.
# These tests define the expected behavior of that module.

try:
    from backend.core.forced_aligner import get_word_timestamps, assign_speakers, reconstruct_sentences
except ImportError:
    # Fallback for structural testing if the file doesn't exist yet
    def get_word_timestamps(*args, **kwargs): return []
    def assign_speakers(*args, **kwargs): return []
    def reconstruct_sentences(*args, **kwargs): return []

# ---------------------------------------------------------------------------
# 1. Micro-Diarization Mapping Tests
# ---------------------------------------------------------------------------

def test_assign_speakers_midpoint_logic():
    """
    Verify that characters are assigned to speakers based on their midpoint.
    mid_time = (start + end) / 2
    """
    char_timestamps = [
        {"char": "哈", "start": 10.0, "end": 10.4},  # mid = 10.2 -> S1
        {"char": "喽", "start": 10.4, "end": 10.8},  # mid = 10.6 -> S1
        {"char": "大", "start": 10.8, "end": 11.2},  # mid = 11.0 -> S2
        {"char": "家", "start": 11.2, "end": 11.6},  # mid = 11.4 -> S2
    ]
    
    pyannote_segments = [
        {"speaker": "SPEAKER_01", "start": 9.0, "end": 10.7},
        {"speaker": "SPEAKER_02", "start": 10.7, "end": 12.0},
    ]
    
    tagged = assign_speakers(char_timestamps, pyannote_segments)
    
    assert len(tagged) == 4
    assert tagged[0]["speaker"] == "SPEAKER_01"
    assert tagged[1]["speaker"] == "SPEAKER_01"
    assert tagged[2]["speaker"] == "SPEAKER_02"
    assert tagged[3]["speaker"] == "SPEAKER_02"


# ---------------------------------------------------------------------------
# 2. Sentence Reconstruction Tests
# ---------------------------------------------------------------------------

def test_reconstruct_sentences_basic_grouping():
    """
    Verify that contiguous characters with the same speaker are grouped into sentences.
    """
    tagged_chars = [
        {"char": "H", "start": 1.0, "end": 1.1, "speaker": "S1"},
        {"char": "i", "start": 1.1, "end": 1.2, "speaker": "S1"},
        {"char": "!", "start": 1.2, "end": 1.3, "speaker": "S1"},
        {"char": "O", "start": 2.0, "end": 2.1, "speaker": "S2"},
        {"char": "k", "start": 2.1, "end": 2.2, "speaker": "S2"},
    ]
    
    sentences = reconstruct_sentences(tagged_chars)
    
    assert len(sentences) == 2
    assert sentences[0]["text"] == "Hi!"
    assert sentences[0]["speaker"] == "S1"
    assert sentences[0]["start"] == 1.0
    assert sentences[0]["end"] == 1.3
    
    assert sentences[1]["text"] == "Ok"
    assert sentences[1]["speaker"] == "S2"
    assert sentences[1]["start"] == 2.0
    assert sentences[1]["end"] == 2.2


def test_reconstruct_sentences_silence_gap():
    """
    Verify that a silence gap > 1 second triggers a new sentence even if speaker is the same.
    """
    tagged_chars = [
        {"char": "A", "start": 1.0, "end": 1.1, "speaker": "S1"},
        {"char": "B", "start": 1.1, "end": 1.2, "speaker": "S1"},
        # 1.5 second gap
        {"char": "C", "start": 2.7, "end": 2.8, "speaker": "S1"},
    ]
    
    sentences = reconstruct_sentences(tagged_chars)
    
    assert len(sentences) == 2
    assert sentences[0]["text"] == "AB"
    assert sentences[1]["text"] == "C"
    assert sentences[1]["start"] == 2.7


# ---------------------------------------------------------------------------
# 3. Forced Alignment Interface Tests (Mocked)
# ---------------------------------------------------------------------------

@patch("backend.core.forced_aligner._get_mms_fa_components")
def test_get_word_timestamps_structure(mock_get_components):
    """
    Verify the output structure of get_word_timestamps via the MMS_FA backend.
    Uses lang="eng" to trigger the non-Chinese routing path.
    """
    mock_model = MagicMock()
    # model(waveform) -> (emission [1,T,C], _)
    mock_model.return_value = (torch.randn(1, 10, 30), None)
    mock_model.parameters.return_value = iter([MagicMock(device=torch.device("cpu"))])

    mock_tokenizer = MagicMock()
    # Handles both old and new tokenizer API signatures
    mock_tokenizer.return_value = [[1, 2]]
    mock_tokenizer.get_labels.return_value = ("<pad>", "H", "e")

    class MockSpan:
        def __init__(self, token, start, end):
            self.token = token
            self.start = start
            self.end = end

    mock_aligner = MagicMock()
    mock_aligner.return_value = ([MockSpan(1, 0, 5), MockSpan(2, 5, 10)], 0.9)

    mock_get_components.return_value = (mock_model, mock_tokenizer, mock_aligner)

    audio_np = np.zeros(16000, dtype=np.float32)
    # lang="eng" → MMS_FA path (Chinese would call _get_cn_components instead)
    result = get_word_timestamps(audio_np, "He", lang="eng")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["char"] == "H"
    assert result[1]["char"] == "e"
    assert "start" in result[0]
    assert "end" in result[0]
