import torch
import torchaudio
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SmartChunker:
    def __init__(self, target_chunk_duration_s=300):
        self.target_chunk_duration_s = target_chunk_duration_s
        self.sample_rate = 16000
        # Load silero-vad from torch hub
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False,
                                           onnx=False)
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils

    def chunk(self, wav_filepath: str, output_dir: str):
        """
        Chunk a wav file into segments of roughly ~5 minutes based on silero VAD.
        Returns a list of dicts: {"start_time": float, "end_time": float, "filepath": str}
        """
        wav_path = Path(wav_filepath)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        wav = self.read_audio(wav_filepath, sampling_rate=self.sample_rate)
        # get speech timestamps
        speech_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=self.sample_rate)
        
        # We want to group these speech segments into ~5 min chunks.
        chunks = []
        current_chunk_audio = []
        current_start_sample = None
        current_end_sample = None
        chunk_index = 0
        
        target_samples = self.target_chunk_duration_s * self.sample_rate
        
        for segment in speech_timestamps:
            start = segment['start']
            end = segment['end']
            
            if current_start_sample is None:
                current_start_sample = start
                
            current_end_sample = end
            
            length = end - current_start_sample
            
            if length >= target_samples:
                # Time to cut a chunk
                chunk_audio = wav[current_start_sample:current_end_sample]
                chunk_filename = f"{wav_path.stem}_chunk_{chunk_index:03d}.wav"
                chunk_filepath = output_dir_path / chunk_filename
                
                self.save_audio(str(chunk_filepath), chunk_audio, self.sample_rate)
                chunks.append({
                    "chunk_index": chunk_index,
                    "start_time": current_start_sample / self.sample_rate,
                    "end_time": current_end_sample / self.sample_rate,
                    "filepath": str(chunk_filepath)
                })
                
                chunk_index += 1
                current_start_sample = None
                current_end_sample = None

        # handle last remaining chunk
        if current_start_sample is not None and current_end_sample is not None:
            chunk_audio = wav[current_start_sample:current_end_sample]
            chunk_filename = f"{wav_path.stem}_chunk_{chunk_index:03d}.wav"
            chunk_filepath = output_dir_path / chunk_filename
            
            self.save_audio(str(chunk_filepath), chunk_audio, self.sample_rate)
            chunks.append({
                "chunk_index": chunk_index,
                "start_time": current_start_sample / self.sample_rate,
                "end_time": current_end_sample / self.sample_rate,
                "filepath": str(chunk_filepath)
            })
            
        return chunks
