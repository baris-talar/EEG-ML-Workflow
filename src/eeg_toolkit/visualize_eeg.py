"""
Educational script: stream synthetic EEG and write text inspection reports.

Lives under eeg_toolkit so editable installs resolve the console entrypoint.
Self-contained: does not import other eeg_toolkit pipeline modules.
"""

from __future__ import annotations

import os
import time
from typing import Sequence

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes

# --- recording / segmentation (seconds) ---
RECORDING_DURATION_SEC = 20
BLOCK_DURATION_SEC = 5
WINDOW_DURATION_SEC = 1

FIRST_N_VALUES = 10
PREVIEW_BLOCK_CHANNELS = 3
PREVIEW_BLOCK_SAMPLES = 10
PREVIEW_WINDOW_CHANNELS = 3
PREVIEW_WINDOW_SAMPLES = 10

ARTIFACTS_DIR = "artifacts"
RAW_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "raw_eeg_preview.txt")
PREPROCESSED_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "preprocessed_eeg_preview.txt")


def _fmt_float(x: float) -> str:
    return f"{x:+.8g}"


def _split_into_blocks(eeg_data: np.ndarray, block_size_samples: int) -> list[np.ndarray]:
    """Split (n_channels, n_samples) into contiguous equal-length blocks."""
    total = eeg_data.shape[1]
    if block_size_samples <= 0 or total % block_size_samples != 0:
        raise ValueError("Invalid block size or non-divisible sample count.")
    n_blocks = total // block_size_samples
    return [
        eeg_data[:, i * block_size_samples : (i + 1) * block_size_samples]
        for i in range(n_blocks)
    ]


def _iter_windows(block: np.ndarray, window_size_samples: int):
    """Yield non-overlapping windows from one block: each (n_channels, window_size)."""
    t = block.shape[1]
    if window_size_samples <= 0 or t % window_size_samples != 0:
        raise ValueError("Invalid window size or non-divisible block length.")
    n_win = t // window_size_samples
    for w in range(n_win):
        a = w * window_size_samples
        b = a + window_size_samples
        yield block[:, a:b]


def _mean_variance_feature_vector(window: np.ndarray) -> np.ndarray:
    """Per-channel mean then per-channel variance."""
    means = np.mean(window, axis=1)
    variances = np.var(window, axis=1)
    return np.concatenate([means, variances])


def _trim_to_aligned_length(
    eeg_data: np.ndarray, block_duration_sec: int, sampling_rate: int
) -> np.ndarray:
    """Keep the tail so duration is a multiple of block_duration_sec."""
    block_samples = block_duration_sec * sampling_rate
    n_samples = eeg_data.shape[1]
    n_full_blocks = n_samples // block_samples
    if n_full_blocks <= 0:
        raise RuntimeError("Recording too short for one full block.")
    used_samples = n_full_blocks * block_samples
    return eeg_data[:, -used_samples:]


def _apply_bandpass_3_45_inplace(eeg_data: np.ndarray, sampling_rate: int) -> None:
    """Apply 3–45 Hz Butterworth zero-phase bandpass to each row (channel)."""
    for channel in eeg_data:
        DataFilter.perform_bandpass(
            channel,
            sampling_rate,
            3.0,
            45.0,
            2,
            FilterTypes.BUTTERWORTH_ZERO_PHASE,
            0,
        )


def _channel_label(channel_names: Sequence[str], idx: int) -> str:
    if idx < len(channel_names):
        return str(channel_names[idx])
    return f"ch{idx}"


def format_report(
    eeg_data: np.ndarray,
    sampling_rate: int,
    channel_names: Sequence[str],
    block_duration_sec: int,
    window_duration_sec: int,
) -> str:
    """Build a human-readable multi-section report for one EEG matrix."""
    lines: list[str] = []
    n_ch, n_samp = eeg_data.shape
    names = [str(channel_names[i]) for i in range(min(n_ch, len(channel_names)))]
    while len(names) < n_ch:
        names.append(f"ch{len(names)}")

    lines.append("===== SUMMARY =====")
    lines.append(f"channels x samples: ({n_ch}, {n_samp})")
    lines.append(f"sampling rate (Hz): {sampling_rate}")
    lines.append(f"channel names: {names}")
    lines.append("")

    lines.append("===== FIRST VALUES PER CHANNEL =====")
    for ci in range(n_ch):
        n_show = min(FIRST_N_VALUES, n_samp)
        vals = eeg_data[ci, :n_show]
        formatted = ", ".join(_fmt_float(float(v)) for v in vals)
        lines.append(f"[{ci}] {_channel_label(names, ci)}: [{formatted}]")
    lines.append("")

    lines.append("===== BASIC STATS PER CHANNEL =====")
    for ci in range(n_ch):
        row = eeg_data[ci, :].astype(float)
        lines.append(
            f"[{ci}] {_channel_label(names, ci)}: "
            f"mean={_fmt_float(float(np.mean(row)))}, "
            f"variance={_fmt_float(float(np.var(row)))}, "
            f"min={_fmt_float(float(np.min(row)))}, "
            f"max={_fmt_float(float(np.max(row)))}"
        )
    lines.append("")

    aligned = _trim_to_aligned_length(eeg_data, block_duration_sec, sampling_rate)
    block_samples = block_duration_sec * sampling_rate
    window_samples = window_duration_sec * sampling_rate
    blocks = _split_into_blocks(aligned, block_samples)
    n_blocks = len(blocks)

    lines.append(f"===== BLOCKS ({block_duration_sec}s) =====")
    lines.append(f"number of blocks: {n_blocks}")
    lines.append(f"shape of each block (channels, samples): ({n_ch}, {block_samples})")
    lines.append(
        f"preview — block 0, first {PREVIEW_BLOCK_CHANNELS} channels × "
        f"first {min(PREVIEW_BLOCK_SAMPLES, block_samples)} samples:"
    )
    b0 = blocks[0]
    for ci in range(min(PREVIEW_BLOCK_CHANNELS, n_ch)):
        n_show = min(PREVIEW_BLOCK_SAMPLES, b0.shape[1])
        vals = b0[ci, :n_show]
        formatted = ", ".join(_fmt_float(float(v)) for v in vals)
        lines.append(f"  [{ci}] {_channel_label(names, ci)}: [{formatted}]")
    lines.append("")

    if block_samples % window_samples != 0:
        lines.append(f"===== WINDOWS ({window_duration_sec}s, from block 0) =====")
        lines.append("(skipped: block length not divisible by window length)")
        lines.append("")
        lines.append("===== FEATURES (first window of block 0) =====")
        lines.append("(skipped)")
        return "\n".join(lines) + "\n"

    wins_in_block = block_samples // window_samples
    lines.append(f"===== WINDOWS ({window_duration_sec}s, from block 0) =====")
    lines.append(f"number of windows in first block: {wins_in_block}")
    lines.append(f"shape of each window (channels, samples): ({n_ch}, {window_samples})")
    w0 = next(_iter_windows(blocks[0], window_samples))
    lines.append(
        f"preview — first window of block 0, first {PREVIEW_WINDOW_CHANNELS} channels × "
        f"first {min(PREVIEW_WINDOW_SAMPLES, w0.shape[1])} samples:"
    )
    for ci in range(min(PREVIEW_WINDOW_CHANNELS, n_ch)):
        n_show = min(PREVIEW_WINDOW_SAMPLES, w0.shape[1])
        vals = w0[ci, :n_show]
        formatted = ", ".join(_fmt_float(float(v)) for v in vals)
        lines.append(f"  [{ci}] {_channel_label(names, ci)}: [{formatted}]")
    lines.append("")

    feat = _mean_variance_feature_vector(w0)
    means = np.mean(w0, axis=1)
    variances = np.var(w0, axis=1)

    lines.append("===== FEATURES (first window of block 0) =====")
    lines.append("mean per channel:")
    for ci in range(n_ch):
        lines.append(f"  [{ci}] {_channel_label(names, ci)}: {_fmt_float(float(means[ci]))}")
    lines.append("variance per channel:")
    for ci in range(n_ch):
        lines.append(f"  [{ci}] {_channel_label(names, ci)}: {_fmt_float(float(variances[ci]))}")
    lines.append("final feature vector [mean..., var...]:")
    feat_str = ", ".join(_fmt_float(float(v)) for v in feat)
    lines.append(f"  [{feat_str}]")
    lines.append(f"  (length {feat.size})")

    return "\n".join(lines) + "\n"


def write_report(path: str, content: str) -> None:
    """Write UTF-8 text; create parent directory if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def run() -> None:
    BoardShim.enable_dev_board_logger()

    board_id = BoardIds.SYNTHETIC_BOARD
    params = BrainFlowInputParams()
    board = BoardShim(board_id, params)

    try:
        board.prepare_session()
        board.start_stream()
        time.sleep(RECORDING_DURATION_SEC)

        data = board.get_board_data()
        eeg_ix = BoardShim.get_eeg_channels(board_id)
        eeg_names = BoardShim.get_eeg_names(board_id)
        fs = int(BoardShim.get_sampling_rate(board_id))
        raw_eeg = data[eeg_ix, :].copy()

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        raw_report = format_report(
            raw_eeg,
            fs,
            eeg_names,
            BLOCK_DURATION_SEC,
            WINDOW_DURATION_SEC,
        )
        write_report(RAW_REPORT_PATH, raw_report)

        preprocessed_eeg = raw_eeg.copy()
        _apply_bandpass_3_45_inplace(preprocessed_eeg, fs)
        pre_report = format_report(
            preprocessed_eeg,
            fs,
            eeg_names,
            BLOCK_DURATION_SEC,
            WINDOW_DURATION_SEC,
        )
        write_report(PREPROCESSED_REPORT_PATH, pre_report)

        print("Reports saved:")
        print(f"  {RAW_REPORT_PATH}")
        print(f"  {PREPROCESSED_REPORT_PATH}")

    finally:
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception:
            pass


def main() -> None:
    run()


if __name__ == "__main__":
    main()
