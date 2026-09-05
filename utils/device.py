"""
Hardware acceleration and device management utility.
Supports automatic selection of CUDA GPU, Apple Silicon MPS, or CPU fallback.
"""

import torch
from utils.logger import get_logger

logger = get_logger(__name__)


def select_device(device_str: str = "auto") -> torch.device:
    """
    Selects the optimal torch.device based on requested preference and hardware availability.

    Args:
        device_str: 'auto', 'cuda', 'cuda:0', 'mps', or 'cpu'.

    Returns:
        torch.device object.
    """
    requested = str(device_str).strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Auto-selected CUDA GPU: {gpu_name}")
            return dev
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Auto-selected Apple Silicon MPS")
            return torch.device("mps")
        else:
            logger.info("Auto-selected CPU execution")
            return torch.device("cpu")

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                dev = torch.device(requested)
                # Test device initialization
                _ = torch.zeros(1, device=dev)
                logger.info(f"Using requested CUDA device: {requested}")
                return dev
            except Exception as e:
                logger.warning(f"Failed to initialize requested CUDA device '{requested}': {e}. Falling back to cpu.")
                return torch.device("cpu")
        else:
            logger.warning(f"CUDA was requested ('{device_str}') but is not available on this system. Falling back to cpu.")
            return torch.device("cpu")

    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            logger.warning("MPS requested but not available. Falling back to cpu.")
            return torch.device("cpu")

    return torch.device("cpu")


def get_device_info() -> dict:
    """
    Returns diagnostic information about available compute devices.
    """
    cuda_avail = torch.cuda.is_available()
    info = {
        "cuda_available": cuda_avail,
        "device_count": torch.cuda.device_count() if cuda_avail else 0,
        "current_device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
        "torch_version": torch.__version__,
    }
    if cuda_avail:
        props = torch.cuda.get_device_properties(0)
        info["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
    return info
