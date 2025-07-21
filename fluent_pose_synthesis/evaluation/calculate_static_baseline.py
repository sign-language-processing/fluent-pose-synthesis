import argparse
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
from pose_evaluation.metrics.pose_processors import NormalizePosesProcessor


def compute_static_baseline_dtw(split_dir: Path, min_length_threshold: int):
    """
    Calculate the Static Baseline DTW score.
    For each fluent sequence, generate a static sequence by repeating its first frame.
    Compare this static sequence with the original complete fluent sequence.
    The sample filtering logic is consistent with direct baseline and model inference.
    """
    all_original_files = sorted(list(split_dir.glob("*_original.pose")))

    if not all_original_files:
        print(f"Error: No '*_original.pose' files found in {split_dir}")
        return 0, []

    valid_fluent_gt_paths = []
    print(
        f"Found {len(all_original_files)} potential samples. Now filtering based on disfluent length >= {min_length_threshold}..."
    )

    for fluent_gt_path in all_original_files:
        file_id_base = fluent_gt_path.stem.replace('_original', '')
        metadata_path = split_dir / f"{file_id_base}_metadata.json"

        if not metadata_path.exists():
            continue

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        disfluent_len = metadata.get("disfluent_pose_length", 0)

        if disfluent_len >= min_length_threshold:
            valid_fluent_gt_paths.append(fluent_gt_path)

    total_valid_files = len(valid_fluent_gt_paths)
    if total_valid_files == 0:
        print(f"No samples met the length requirement (disfluent_length >= {min_length_threshold}).")
        return 0, []

    dtw_metric = DistanceMetric(
        name="static_baseline_DTW",
        distance_measure=DTWDTAIImplementationDistanceMeasure(
            name="dtaiDTW",
            use_fast=True,
            default_distance=0.0,
        ),
        pose_preprocessors=[NormalizePosesProcessor()],
    )

    dtw_scores = []

    print(f"\nCalculating Static Baseline DTW for {total_valid_files} valid samples...")
    for fluent_gt_path in tqdm(valid_fluent_gt_paths, desc=f"Static Baseline DTW on '{split_dir.name}'"):
        with open(fluent_gt_path, "rb") as f:
            fluent_gt_pose = Pose.read(f.read())
        fluent_gt_pose.body.data = fluent_gt_pose.body.data.astype(np.double)

        if len(fluent_gt_pose.body.data) == 0:
            continue

        first_frame = fluent_gt_pose.body.data[0:1]
        num_frames = len(fluent_gt_pose.body.data)

        static_pose_data = np.tile(first_frame, (num_frames, 1, 1, 1))
        # Confidence is 3D array (T, 1, K)
        static_confidence = np.ones((num_frames, 1, fluent_gt_pose.body.data.shape[2]), dtype=np.float32)

        static_pose_body = NumPyPoseBody(fps=fluent_gt_pose.body.fps, data=static_pose_data,
                                         confidence=static_confidence)
        static_prediction_pose = Pose(header=fluent_gt_pose.header, body=static_pose_body)

        dtw = dtw_metric(static_prediction_pose, fluent_gt_pose)
        dtw_scores.append(dtw)

    if not dtw_scores:
        print("No valid DTW scores were calculated.")
        return 0, []

    mean_dtw = np.mean(dtw_scores)
    print("\n" + "=" * 50)
    print(f"=== Static Baseline Mean DTW on '{split_dir.name}' ===")
    print(f"Total file pairs evaluated (after filtering): {len(dtw_scores)}")
    print(f"Mean DTW: {mean_dtw:.4f}")
    print("=" * 50)

    return mean_dtw, dtw_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the Static Baseline DTW score.")
    parser.add_argument("-d", "--data_root", type=str, required=True,
                        help="Path to the root of pose data directory (e.g., './pose_data/output').")
    parser.add_argument("-s", "--split", type=str, default="test",
                        help="The data split to evaluate ('validation' or 'test').")
    parser.add_argument(
        "--min_length", type=int, default=25, help=
        "Minimum disfluent length threshold for a sample to be included, ensuring consistency with model evaluation.")

    args = parser.parse_args()

    data_root_path = Path(args.data_root)
    split_dir_path = data_root_path / args.split

    if not split_dir_path.exists():
        print(f"Error: Split directory not found at {split_dir_path}")
    else:
        compute_static_baseline_dtw(split_dir_path, args.min_length)
