"""Calibration pipeline for personalized supervised classification."""

from __future__ import annotations

import time

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

from eeg_toolkit.artifacts import save_calibration_dataset
from eeg_toolkit.features import encode_labels, extract_mean_variance_features
from eeg_toolkit.preprocess import (
    apply_bandpass_inplace,
    apply_synthetic_state_transform,
)
from eeg_toolkit.windowing import iter_windows, split_into_blocks


DEFAULT_TOTAL_DURATION_SEC = 20
DEFAULT_BLOCK_DURATION_SEC = 5
DEFAULT_WINDOW_DURATION_SEC = 1
DEFAULT_RANDOM_SEED = 42
DEFAULT_APPLY_PREPROCESSING = True
DATASET_OUTPUT_PATH = "python_package/examples/tests/artifacts/eeg_calibration_dataset.npz"


def _input_with_default(prompt_text, default_value):
    try:
        value = input(prompt_text).strip()
    except EOFError:
        value = ""
    return value if value else str(default_value)


def _input_int(prompt_text, default_value, min_value=1):
    while True:
        raw_value = _input_with_default(prompt_text, default_value)
        try:
            parsed = int(raw_value)
        except ValueError:
            print("Please enter an integer value.")
            continue
        if parsed < min_value:
            print(f"Value must be at least {min_value}.")
            continue
        return parsed


def _input_yes_no(prompt_text, default_yes=True):
    default_text = "Y/n" if default_yes else "y/N"
    while True:
        raw_value = _input_with_default(f"{prompt_text} [{default_text}]: ", "").lower()
        if raw_value == "":
            return default_yes
        if raw_value in ("y", "yes"):
            return True
        if raw_value in ("n", "no"):
            return False
        print("Please answer with y or n.")


def run():
    BoardShim.enable_dev_board_logger()

    print("Calibration workflow for personalized supervised classification.")
    total_duration_sec = _input_int(
        "Total recording duration in seconds [20]: ",
        DEFAULT_TOTAL_DURATION_SEC,
        min_value=4,
    )
    block_duration_sec = _input_int(
        "Block duration in seconds [5]: ",
        DEFAULT_BLOCK_DURATION_SEC,
        min_value=1,
    )
    if total_duration_sec % block_duration_sec != 0:
        raise ValueError(
            "Total duration must be divisible by block duration for equal blocks."
        )

    num_blocks = total_duration_sec // block_duration_sec
    window_duration_sec = DEFAULT_WINDOW_DURATION_SEC
    random_seed = DEFAULT_RANDOM_SEED
    rng = np.random.default_rng(random_seed)
    apply_preprocessing = _input_yes_no(
        "Apply common EEG preprocessing (bandpass 3-45 Hz)?",
        default_yes=DEFAULT_APPLY_PREPROCESSING,
    )

    print()
    print(f"Number of calibration blocks: {num_blocks}")
    block_labels = []
    for block_idx in range(num_blocks):
        default_label = "low_freq" if block_idx % 2 == 0 else "high_freq"
        label = _input_with_default(
            f"Label for block {block_idx + 1} (default {default_label}): ",
            default_label,
        )
        block_labels.append(label)

    unique_labels = list(dict.fromkeys(block_labels))
    label_to_state = {}
    print()
    print("Assign synthetic state type per label (used only for Synthetic Board calibration).")
    for label in unique_labels:
        while True:
            suggested = "low_freq" if "low" in label.lower() else "high_freq"
            state = _input_with_default(
                f"State type for label '{label}' [low_freq/high_freq] (default {suggested}): ",
                suggested,
            ).lower()
            if state in ("low_freq", "high_freq"):
                label_to_state[label] = state
                break
            print("Use either 'low_freq' or 'high_freq'.")

    print()
    print("Planned calibration schedule:")
    for block_idx, label in enumerate(block_labels):
        start_t = block_idx * block_duration_sec
        end_t = start_t + block_duration_sec
        print(
            f"  Block {block_idx + 1}: {start_t:>3}-{end_t:>3}s "
            f"label='{label}' state='{label_to_state[label]}'"
        )

    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD
    board = BoardShim(board_id, params)

    try:
        print()
        print("Preparing data acquisition session...")
        board.prepare_session()
        print("Starting data stream...")
        board.start_stream()

        for block_idx, label in enumerate(block_labels):
            print(f"Current block {block_idx + 1}/{num_blocks}: {label}")
            time.sleep(block_duration_sec)

        data = board.get_board_data()
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        eeg_names = BoardShim.get_eeg_names(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        eeg_data = data[eeg_channels, :]

        if apply_preprocessing:
            print("Applying preprocessing (bandpass 3-45 Hz)...")
            apply_bandpass_inplace(
                eeg_data,
                sampling_rate=sampling_rate,
                low_hz=3.0,
                high_hz=45.0,
                order=2,
            )

        block_size_samples = block_duration_sec * sampling_rate
        required_samples = num_blocks * block_size_samples
        if eeg_data.shape[1] < required_samples:
            raise RuntimeError(
                f"Not enough EEG samples for planned schedule. "
                f"Expected at least {required_samples}, got {eeg_data.shape[1]}."
            )

        # Keep the most recent samples to align with the scheduled recording period.
        eeg_data = eeg_data[:, -required_samples:]

        raw_blocks = split_into_blocks(eeg_data, block_size_samples)

        transformed_blocks = []
        block_state_types = []
        for block_idx, block in enumerate(raw_blocks):
            label = block_labels[block_idx]
            state_type = label_to_state[label]
            transformed_blocks.append(
                apply_synthetic_state_transform(block, state_type, sampling_rate, rng)
            )
            block_state_types.append(state_type)

        transformed_eeg = np.hstack(transformed_blocks)

        window_size_samples = window_duration_sec * sampling_rate
        if block_size_samples % window_size_samples != 0:
            raise RuntimeError("Block duration must be divisible by window duration.")

        feature_vectors = []
        window_labels = []
        for block_idx, block in enumerate(transformed_blocks):
            label = block_labels[block_idx]
            block_windows = list(iter_windows(block, window_size_samples))
            block_features = extract_mean_variance_features(block_windows)
            feature_vectors.extend(block_features.tolist())
            window_labels.extend([label] * len(block_windows))

        X = np.array(feature_vectors, dtype=float)
        y = np.array(window_labels, dtype=str)
        label_names, y_idx = encode_labels(window_labels)

        label_counts = {}
        for label in label_names:
            label_counts[label] = int(np.sum(y == label))

        output_path = _input_with_default(
            f"Dataset output path [{DATASET_OUTPUT_PATH}]: ",
            DATASET_OUTPUT_PATH,
        )
        dataset = {
            "X": X,
            "y": y,
            "y_idx": y_idx,
            "label_names": label_names,
            "block_labels": np.array(block_labels, dtype=str),
            "block_state_types": np.array(block_state_types, dtype=str),
            "sampling_rate": int(sampling_rate),
            "eeg_channel_indices": np.array(eeg_channels, dtype=int),
            "eeg_channel_names": np.array(eeg_names, dtype=str),
            "window_size_samples": int(window_size_samples),
            "window_duration_sec": int(window_duration_sec),
            "block_duration_sec": int(block_duration_sec),
            "total_duration_sec": int(total_duration_sec),
            "apply_preprocessing": int(apply_preprocessing),
            "preprocessing_bandpass_low_hz": 3.0,
            "preprocessing_bandpass_high_hz": 45.0,
            "preprocessing_filter_order": 2,
            "feature_description": "mean_and_variance_per_channel",
            "random_seed": int(random_seed),
            "transformed_eeg": transformed_eeg,
        }
        save_calibration_dataset(output_path, dataset)

        print()
        print("--- Calibration dataset summary ---")
        print("Saved dataset:", output_path)
        print("Feature matrix X shape:", X.shape)
        print("Labels y shape:", y.shape)
        print("Unique labels:", label_names.tolist())
        print("Label counts:", label_counts)
        print("Sampling rate (Hz):", sampling_rate)
        print("EEG channels:", eeg_names)
        print("Window size (samples):", window_size_samples)
        print("Block labels:", block_labels)
        print("Block state types:", block_state_types)

    finally:
        print("Stopping stream and releasing session...")
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception:
            pass


def main():
    run()


if __name__ == "__main__":
    main()

