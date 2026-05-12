import gc
import logging

import torch

logger = logging.getLogger(__name__)

# Minimum free VRAM required before loading a large model
MIN_VRAM_GB = 4.0


def get_free_vram_gb() -> float:
    """Return free VRAM in GiB, or 0 if CUDA is not available."""
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _ = torch.cuda.mem_get_info()
    return free_bytes / (1024 ** 3)


def get_total_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    _, total_bytes = torch.cuda.mem_get_info()
    return total_bytes / (1024 ** 3)


def check_vram(min_gb: float = MIN_VRAM_GB) -> bool:
    """
    Return True if enough free VRAM is available.
    Logs a warning (not an exception) when the check fails — callers decide
    whether to fall back to CPU or abort.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available — all inference will run on CPU")
        return False
    free = get_free_vram_gb()
    if free < min_gb:
        logger.warning(
            f"Insufficient VRAM: {free:.2f} GB free, {min_gb:.1f} GB required. "
            "Model will be loaded on CPU — expect slower processing."
        )
        return False
    logger.info(
        f"VRAM OK: {free:.2f} GB free / {get_total_vram_gb():.2f} GB total "
        f"(threshold: {min_gb:.1f} GB)"
    )
    return True


def clear_gpu_memory() -> None:
    """
    Aggressively free GPU memory between model loads.
    Call this after deleting a model reference and before loading the next one.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.debug("GPU memory cleared")


def get_inference_device(min_vram_gb: float = MIN_VRAM_GB) -> torch.device:
    """
    Return 'cuda' if enough VRAM is free, else 'cpu'.
    Intended to be called once per model load — not per inference call.
    """
    if check_vram(min_vram_gb):
        return torch.device("cuda")
    return torch.device("cpu")
