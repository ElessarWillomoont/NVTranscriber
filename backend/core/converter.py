import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

MEDIA_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mp3", ".wav", ".m4a", ".flac"}


def scan_directory(directory: str) -> list[Path]:
    """Recursively find all media files under directory."""
    root = Path(directory)
    if not root.is_dir():
        logger.warning(f"Directory not found: {directory}")
        return []

    found = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS]
    logger.info(f"Found {len(found)} media file(s) in '{directory}'")
    return found


def convert_to_wav(input_path: Path, output_path: Path) -> bool:
    """Convert/extract audio to 16 kHz mono PCM WAV via ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]

    logger.info(f"[ffmpeg] Converting: {input_path.name} → {output_path.name}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stderr.splitlines():
            stripped = line.strip()
            if stripped:
                logger.debug(f"[ffmpeg] {stripped}")
        logger.info(f"[ffmpeg] Done: {output_path}")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f"[ffmpeg] Conversion failed for '{input_path}': {exc.stderr[-500:]}")
        return False
    except FileNotFoundError:
        logger.error("[ffmpeg] 'ffmpeg' not found in PATH — please install ffmpeg and add it to PATH")
        return False
