"""Prediction pipeline for personalized EEG state classification."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

from eeg_toolkit.artifacts import load_model_artifact
from eeg_toolkit.features import extract_mean_variance_features
from eeg_toolkit.modeling import predict_labels
from eeg_toolkit.preprocess import (
    apply_bandpass_inplace,
    apply_synthetic_state_transform,
)
from eeg_toolkit.windowing import iter_windows, split_into_blocks


DEFAULT_MODEL_PATH = "artifacts/eeg_logreg_model.joblib"
DEFAULT_RECORDING_DURATION_SEC = 20
DEFAULT_BLOCK_DURATION_SEC = 5
DEFAULT_RANDOM_SEED = 42


def _input_with_default(prompt_text, default_value):
    try:
        value = input(prompt_text).strip()
    except EOFError:
        value = ""
    return value if value else str(default_value)


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


def _select_validation_labels(label_names):
    lower_names = [name.lower() for name in label_names]
    low_idx = next((i for i, name in enumerate(lower_names) if "low" in name), None)
    high_idx = next((i for i, name in enumerate(lower_names) if "high" in name), None)

    if low_idx is not None and high_idx is not None and low_idx != high_idx:
        return str(label_names[low_idx]), str(label_names[high_idx])
    if len(label_names) >= 2:
        return str(label_names[0]), str(label_names[1])
    if len(label_names) == 1:
        return str(label_names[0]), str(label_names[0])
    raise ValueError("No labels available in model artifact.")


def run():
    BoardShim.enable_dev_board_logger()

    model_path = _input_with_default(
        f"Model path [{DEFAULT_MODEL_PATH}]: ",
        DEFAULT_MODEL_PATH,
    )
    model_artifact = load_model_artifact(model_path)
    model = model_artifact["model"]
    label_names = np.array(model_artifact["label_names"], dtype=str)

    window_size_samples = int(model_artifact["window_size_samples"])
    trained_preprocessing = bool(model_artifact["apply_preprocessing"])
    band_low_hz = float(model_artifact["preprocessing_bandpass_low_hz"])
    band_high_hz = float(model_artifact["preprocessing_bandpass_high_hz"])
    filter_order = int(model_artifact["preprocessing_filter_order"])
    trained_sampling_rate = int(model_artifact["sampling_rate"])

    apply_preprocessing = _input_yes_no(
        "Apply common EEG preprocessing (bandpass 3-45 Hz)?",
        default_yes=trained_preprocessing,
    )
    synthetic_validation_enabled = _input_yes_no(
        "Enable synthetic validation mode (inject low/high synthetic states)?",
        default_yes=True,
    )

    recording_duration_sec = int(
        _input_with_default(
            f"Prediction recording duration in seconds [{DEFAULT_RECORDING_DURATION_SEC}]: ",
            DEFAULT_RECORDING_DURATION_SEC,
        )
    )
    block_duration_sec = DEFAULT_BLOCK_DURATION_SEC
    if recording_duration_sec % block_duration_sec != 0:
        raise ValueError(
            f"Recording duration must be divisible by {block_duration_sec} seconds."
        )
    num_blocks = recording_duration_sec // block_duration_sec
    if num_blocks < 2:
        raise ValueError("Prediction recording must include at least 2 blocks.")

    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD
    board = BoardShim(board_id, params)

    try:
        print("Preparing data acquisition session...")
        board.prepare_session()
        print("Starting unlabeled recording...")
        board.start_stream()
        time.sleep(recording_duration_sec)

        data = board.get_board_data()
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        eeg_names = BoardShim.get_eeg_names(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        eeg_data = data[eeg_channels, :]

        if sampling_rate != trained_sampling_rate:
            print(
                "Warning: sampling rate differs from training artifact. "
                f"Training={trained_sampling_rate}, current={sampling_rate}"
            )

        if apply_preprocessing:
            print(
                f"Applying preprocessing bandpass ({band_low_hz}-{band_high_hz} Hz, "
                f"order {filter_order})..."
            )
            apply_bandpass_inplace(
                eeg_data,
                sampling_rate=sampling_rate,
                low_hz=band_low_hz,
                high_hz=band_high_hz,
                order=filter_order,
            )

        block_size_samples = block_duration_sec * sampling_rate
        required_samples = num_blocks * block_size_samples
        if eeg_data.shape[1] < required_samples:
            raise RuntimeError(
                f"Not enough samples for configured block schedule. "
                f"Need {required_samples}, got {eeg_data.shape[1]}."
            )

        if block_size_samples % window_size_samples != 0:
            raise RuntimeError("Block size must be divisible by window size.")

        # Keep recent samples aligned with the planned block timeline.
        eeg_data = eeg_data[:, -required_samples:]

        random_seed = DEFAULT_RANDOM_SEED
        rng = np.random.default_rng(random_seed)

        raw_blocks = split_into_blocks(eeg_data, block_size_samples)
        transformed_blocks = raw_blocks

        synthetic_block_labels = None
        synthetic_block_state_types = None
        actual_window_labels = None
        validation_accuracy = None
        correct_windows = None

        if synthetic_validation_enabled:
            # Synthetic validation setup:
            # inject known state patterns in prediction recording to verify end-to-end behavior.
            low_label, high_label = _select_validation_labels(label_names)
            low_blocks = num_blocks // 2
            high_blocks = num_blocks - low_blocks
            synthetic_block_labels = [low_label] * low_blocks + [high_label] * high_blocks
            synthetic_block_state_types = ["low_freq"] * low_blocks + ["high_freq"] * high_blocks
            shuffle_order = rng.permutation(num_blocks)
            synthetic_block_labels = [synthetic_block_labels[idx] for idx in shuffle_order]
            synthetic_block_state_types = [
                synthetic_block_state_types[idx] for idx in shuffle_order
            ]

            transformed_blocks = []
            for block_idx, raw_block in enumerate(raw_blocks):
                state_type = synthetic_block_state_types[block_idx]
                transformed_blocks.append(
                    apply_synthetic_state_transform(
                        raw_block, state_type, sampling_rate, rng
                    )
                )

        eeg_data = np.hstack(transformed_blocks)

        features_per_block = []
        for block in transformed_blocks:
            block_windows = list(iter_windows(block, window_size_samples))
            block_features = extract_mean_variance_features(block_windows)
            features_per_block.append(block_features)
        X_pred = np.vstack(features_per_block) if features_per_block else np.empty((0, 0))

        if X_pred.shape[0] == 0:
            raise RuntimeError("No full windows available for prediction.")

        predicted_labels = predict_labels(model, X_pred, label_names)

        windows_per_block = block_size_samples // window_size_samples

        if synthetic_validation_enabled:
            actual_window_labels = []
            for block_label in synthetic_block_labels:
                actual_window_labels.extend([block_label] * windows_per_block)

            if len(actual_window_labels) != len(predicted_labels):
                raise RuntimeError("Mismatch between actual and predicted window counts.")

            correct_windows = sum(
                int(actual == predicted)
                for actual, predicted in zip(actual_window_labels, predicted_labels)
            )
            validation_accuracy = correct_windows / len(predicted_labels)

        print()
        print("--- Prediction summary ---")
        print("Full data shape:", data.shape)
        print("EEG-only shape:", eeg_data.shape)
        print("EEG channel names:", eeg_names)
        print("Sampling rate (Hz):", sampling_rate)
        print("Block size (samples):", block_size_samples)
        print("Number of blocks:", num_blocks)
        print("Random seed:", random_seed)
        print("Feature matrix shape:", X_pred.shape)

        if synthetic_validation_enabled:
            print("Synthetic block labels:", synthetic_block_labels)
            print("Synthetic block state types:", synthetic_block_state_types)
            print("Actual window labels:", actual_window_labels)

            print()
            print("Block-level signal summary (processed EEG):")
            for block_idx, block in enumerate(transformed_blocks):
                mean_abs = float(np.mean(np.abs(block)))
                variance = float(np.var(block))
                block_min = float(np.min(block))
                block_max = float(np.max(block))
                print(
                    f"  Block {block_idx + 1} [{synthetic_block_labels[block_idx]} | "
                    f"{synthetic_block_state_types[block_idx]}]: "
                    f"mean_abs={mean_abs:.4f} variance={variance:.4f} "
                    f"min={block_min:.4f} max={block_max:.4f}"
                )

        print()
        print("Predicted labels per window:")
        for window_idx, predicted in enumerate(predicted_labels):
            if synthetic_validation_enabled:
                actual = actual_window_labels[window_idx]
                print(f"  Window {window_idx + 1}: actual={actual} predicted={predicted}")
            else:
                print(f"  Window {window_idx + 1}: predicted={predicted}")

        unique_labels, counts = np.unique(predicted_labels, return_counts=True)
        print(
            "Prediction counts:",
            {label: int(count) for label, count in zip(unique_labels, counts)},
        )

        if synthetic_validation_enabled:
            print(
                f"Validation accuracy against synthetic reference: "
                f"{correct_windows}/{len(predicted_labels)} = {validation_accuracy:.4f}"
            )

            # Plot 1: Continuous signal across all blocks for selected channels.
            total_samples = eeg_data.shape[1]
            time_axis = np.arange(total_samples) / sampling_rate
            channels_to_plot = min(3, eeg_data.shape[0])
            fig_cont, ax_cont = plt.subplots(figsize=(12, 4))
            for ch_idx in range(channels_to_plot):
                ch_name = eeg_names[ch_idx] if ch_idx < len(eeg_names) else f"ch_{ch_idx}"
                ax_cont.plot(time_axis, eeg_data[ch_idx, :], linewidth=1.0, label=ch_name)
            for boundary_idx in range(1, num_blocks):
                ax_cont.axvline(
                    boundary_idx * block_duration_sec,
                    color="gray",
                    linestyle="--",
                    alpha=0.6,
                )
            ax_cont.set_title("Prediction EEG over time with synthetic state blocks")
            ax_cont.set_xlabel("Time (s)")
            ax_cont.set_ylabel("EEG amplitude")
            ax_cont.legend(loc="upper right")
            ax_cont.grid(alpha=0.3)
            fig_cont.tight_layout()

            # Plot 2: One subplot per block for one selected channel with actual/predicted context.
            selected_channel = 0
            selected_name = (
                eeg_names[selected_channel]
                if selected_channel < len(eeg_names)
                else f"ch_{selected_channel}"
            )
            fig_blocks, axes = plt.subplots(
                num_blocks, 1, figsize=(12, 2.2 * num_blocks), sharex=True
            )
            if num_blocks == 1:
                axes = [axes]
            block_time = np.arange(block_size_samples) / sampling_rate
            for block_idx in range(num_blocks):
                axes[block_idx].plot(
                    block_time,
                    transformed_blocks[block_idx][selected_channel, :],
                    linewidth=1.0,
                )
                block_start = block_idx * windows_per_block
                block_end = block_start + windows_per_block
                block_pred_labels = predicted_labels[block_start:block_end]
                if block_pred_labels:
                    pred_unique, pred_counts = np.unique(block_pred_labels, return_counts=True)
                    majority_pred_label = str(pred_unique[np.argmax(pred_counts)])
                else:
                    majority_pred_label = "N/A"
                axes[block_idx].set_title(
                    f"Block {block_idx + 1}: actual={synthetic_block_labels[block_idx]} "
                    f"({synthetic_block_state_types[block_idx]}), "
                    f"majority_pred={majority_pred_label}"
                )
                axes[block_idx].set_ylabel("Amplitude")
                axes[block_idx].grid(alpha=0.3)
            axes[-1].set_xlabel("Time within block (s)")
            fig_blocks.suptitle(f"Per-block prediction inspection for channel {selected_name}")
            fig_blocks.tight_layout()
            plt.show()

        print()
        print(
            "Interpretation note: predictions correspond to user-defined calibration labels "
            "and should be interpreted within that calibration context. "
            "Synthetic reference labels are available here only because synthetic "
            "state patterns are injected for end-to-end validation. "
            "In real inference on user EEG, reference labels are not available."
        )

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

