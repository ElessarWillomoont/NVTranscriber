import subprocess
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self):
        pass
        
    def process(self, input_filepath: str, output_dir: str) -> str:
        """
        Convert input media file to 16kHz, mono, PCM s16le .wav format.
        """
        input_path = Path(input_filepath)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{input_path.stem}_processed.wav"
        output_filepath = output_dir_path / output_filename
        
        if output_filepath.exists():
            logger.info(f"File {output_filepath} already exists. Skipping preprocessing.")
            return str(output_filepath)
            
        cmd = [
            "ffmpeg",
            "-y", # Overwrite output files without asking
            "-i", str(input_path),
            "-vn", # Disable video recording
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_filepath)
        ]
        
        try:
            # Run ffmpeg subprocess
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            logger.info(f"Successfully processed {input_path} to {output_filepath}")
            return str(output_filepath)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing {input_path}: {e.stderr}")
            raise RuntimeError(f"FFmpeg failed to process {input_path}") from e
