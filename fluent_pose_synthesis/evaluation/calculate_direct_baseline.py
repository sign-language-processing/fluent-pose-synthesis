import argparse
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm

from pose_format import Pose
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
from pose_evaluation.metrics.pose_processors import NormalizePosesProcessor


def compute_direct_baseline_dtw(split_dir: Path, min_length_threshold: int):
    """
    Calculate the Direct Baseline DTW score. The sample filtering logic is consistent with inference.
    For each sample, check its metadata and only keep samples where disfluent_pose_length >= min_length_threshold.
    Directly compare the full Disfluent Pose and the full Fluent Pose.
    """
    # 1. Find all possible sample pairs
    all_original_files = sorted(list(split_dir.glob("*_original.pose")))

    if not all_original_files:
        print(f"Error: No '*_original.pose' files found in {split_dir}")
        return 0, []

    # 2. Filter samples to ensure consistency with the dataset loading logic in infer.py
    valid_pairs_for_evaluation = []
    print(
        f"Found {len(all_original_files)} potential samples. Now filtering based on disfluent length >= {min_length_threshold}..."
    )

    for fluent_gt_path in all_original_files:
        file_id_base = fluent_gt_path.stem.replace('_original', '')
        disfluent_path = split_dir / f"{file_id_base}_updated.pose"
        metadata_path = split_dir / f"{file_id_base}_metadata.json"

        if not (disfluent_path.exists() and metadata_path.exists()):
            continue  # Skip if corresponding files are incomplete

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        disfluent_len = metadata.get("disfluent_pose_length", 0)

        # Filtering logic
        if disfluent_len >= min_length_threshold:
            valid_pairs_for_evaluation.append({"disfluent": disfluent_path, "fluent": fluent_gt_path})

    total_valid_files = len(valid_pairs_for_evaluation)
    if total_valid_files == 0:
        print(f"No samples met the length requirement (disfluent_length >= {min_length_threshold}).")
        return 0, []

    # 3. Initialize the DTW metric
    dtw_metric = DistanceMetric(
        name="direct_baseline_DTW",
        distance_measure=DTWDTAIImplementationDistanceMeasure(
            name="dtaiDTW",
            use_fast=True,
            default_distance=0.0,
        ),
        pose_preprocessors=[NormalizePosesProcessor()],
    )

    dtw_scores = []

    print(f"\nCalculating Direct Baseline DTW for {total_valid_files} valid samples...")
    for pair in tqdm(valid_pairs_for_evaluation, desc=f"Direct Baseline DTW on '{split_dir.name}'"):
        # Load the full disfluent sequence
        with open(pair["disfluent"], "rb") as f:
            disfluent_pose = Pose.read(f.read())
        disfluent_pose.body.data = disfluent_pose.body.data.astype(np.double)

        # Load the full fluent reference sequence
        with open(pair["fluent"], "rb") as f:
            fluent_gt_pose = Pose.read(f.read())
        fluent_gt_pose.body.data = fluent_gt_pose.body.data.astype(np.double)

        # Directly compare the two full sequences
        dtw = dtw_metric(disfluent_pose, fluent_gt_pose)
        dtw_scores.append(dtw)

    if not dtw_scores:
        print("No valid DTW scores were calculated.")
        return 0, []

    mean_dtw = np.mean(dtw_scores)
    print("\n" + "=" * 50)
    print(f"=== Direct Baseline Mean DTW on '{split_dir.name}' ===")
    print(f"Total file pairs evaluated (after filtering): {len(dtw_scores)}")
    print(f"Mean DTW: {mean_dtw:.4f}")
    print("=" * 50)

    return mean_dtw, dtw_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the Direct Baseline DTW score.")
    parser.add_argument("-d", "--data_root", type=str, required=True,
                        help="Path to the root of pose data directory (e.g., './pose_data/output').")
    parser.add_argument("-s", "--split", type=str, default="test",
                        help="The data split to evaluate ('validation' or 'test').")
    parser.add_argument("--min_length", type=int, default=25,
                        help="Minimum disfluent length threshold for a sample to be included.")

    args = parser.parse_args()

    data_root_path = Path(args.data_root)
    split_dir_path = data_root_path / args.split

    if not split_dir_path.exists():
        print(f"Error: Split directory not found at {split_dir_path}")
    else:
        compute_direct_baseline_dtw(split_dir_path, args.min_length)
