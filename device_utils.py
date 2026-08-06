"""
device_utils.py

Single source of truth for picking an embedding device across environments:
- Local Windows dev machine with an AMD GPU -> DirectML
- Local/HPC machine with an NVIDIA GPU -> CUDA
- Docker / Kubernetes (Linux, no GPU on free-tier nodes) -> CPU

DirectML has no Linux build, so importing torch_directml inside a Linux
container will raise ImportError or crash on load. We treat that import as
best-effort and fall back cleanly.
"""

import logging

import torch

logger = logging.getLogger("device_utils")


def get_embedding_device() -> str:
    """Return the best available device string/object for HuggingFaceEmbeddings."""

    # 1. NVIDIA GPU (HPC cluster, cloud GPU instance)
    if torch.cuda.is_available():
        logger.info("Using CUDA device for embeddings.")
        return "cuda"

    # 2. AMD GPU via DirectML (Windows-only, local dev)
    try:
        import torch_directml  # noqa: F401 - optional dependency, Windows-only

        device = torch_directml.device()
        logger.info("Using DirectML device for embeddings.")
        return device
    except ImportError:
        pass
    except Exception as e:  # pragma: no cover - defensive, DirectML can fail at runtime too
        logger.warning(f"torch_directml import succeeded but device init failed: {e}")

    # 3. CPU fallback (Docker / Kubernetes / any Linux host without CUDA)
    logger.info("No GPU backend available. Falling back to CPU for embeddings.")
    return "cpu"


def get_embedding_batch_size(device: str) -> int:
    """
    CPU wants a smaller batch size than GPU to avoid long stalls per batch.
    GPU batch size is also kept conservative (not 32) because longer chunks
    (e.g. the 1200-token splitter) multiply per-item memory cost — a batch
    size safe for 400-token chunks can still OOM on 1200-token chunks on
    GPUs with limited VRAM (this bit us on DirectML specifically).
    """
    if device == "cpu":
        return 8
    return 8