"""EEG preprocessing helpers."""

from __future__ import annotations

import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes


def apply_bandpass_inplace(
    eeg_data: np.ndarray,
    sampling_rate: int,
    low_hz: float,
    high_hz: float,
    order: int = 2,
) -> None:
    """Apply a zero-phase Butterworth bandpass filter to each EEG channel."""
    for eeg_channel in eeg_data:
        DataFilter.perform_bandpass(
            eeg_channel,
            sampling_rate,
            low_hz,
            high_hz,
            order,
            FilterTypes.BUTTERWORTH_ZERO_PHASE,
            0,
        )


def apply_synthetic_state_transform(
    block: np.ndarray,
    state_type: str,
    sampling_rate: int,
    rng,
) -> np.ndarray:
    """Apply synthetic low/high frequency state transform used in examples."""
    transformed = block.copy()
    if state_type == "low_freq":
        for channel in transformed:
            DataFilter.perform_lowpass(
                channel,
                sampling_rate,
                10.0,
                2,
                FilterTypes.BUTTERWORTH_ZERO_PHASE,
                0,
            )
    elif state_type == "high_freq":
        t = np.arange(transformed.shape[1]) / sampling_rate
        for ch_idx, channel in enumerate(transformed):
            DataFilter.perform_bandpass(
                channel,
                sampling_rate,
                18.0,
                35.0,
                2,
                FilterTypes.BUTTERWORTH_ZERO_PHASE,
                0,
            )
            channel *= 1.4
            channel += 6.0 * np.sin(2.0 * np.pi * 22.0 * t + 0.2 * ch_idx)
            channel += rng.normal(0.0, 0.8, transformed.shape[1])
    else:
        raise ValueError(f"Unsupported synthetic state type: {state_type}")
    return transformed

