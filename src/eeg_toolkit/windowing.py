"""Window and block helpers for EEG sample arrays."""

from __future__ import annotations

import numpy as np


def trim_to_recent_samples(eeg_data: np.ndarray, required_samples: int) -> np.ndarray:
    """Keep the most recent samples to align with planned recording period."""
    if required_samples <= 0:
        raise ValueError("required_samples must be positive.")
    if eeg_data.shape[1] < required_samples:
        raise RuntimeError(
            f"Not enough samples. Expected at least {required_samples}, got {eeg_data.shape[1]}."
        )
    return eeg_data[:, -required_samples:]


def split_into_blocks(eeg_data: np.ndarray, block_size_samples: int) -> list[np.ndarray]:
    """Split EEG data into contiguous equal-length blocks."""
    if block_size_samples <= 0:
        raise ValueError("block_size_samples must be positive.")
    total_samples = eeg_data.shape[1]
    if total_samples % block_size_samples != 0:
        raise RuntimeError("Total samples must be divisible by block size.")

    blocks = []
    num_blocks = total_samples // block_size_samples
    for block_idx in range(num_blocks):
        start = block_idx * block_size_samples
        end = start + block_size_samples
        blocks.append(eeg_data[:, start:end])
    return blocks


def iter_windows(block: np.ndarray, window_size_samples: int):
    """Yield contiguous non-overlapping windows from a single block."""
    if window_size_samples <= 0:
        raise ValueError("window_size_samples must be positive.")
    if block.shape[1] % window_size_samples != 0:
        raise RuntimeError("Block size must be divisible by window size.")

    num_windows = block.shape[1] // window_size_samples
    for window_idx in range(num_windows):
        start = window_idx * window_size_samples
        end = start + window_size_samples
        yield block[:, start:end]

