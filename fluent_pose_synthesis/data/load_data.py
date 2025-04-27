import time
import json
from pathlib import Path
from typing import Any, Dict

import torch
import numpy as np
from torch.utils.data import Dataset
from pose_format import Pose
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import normalize_mean_std


class SignLanguagePoseDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        fluent_frames: int,
        dtype=np.float32,
        limited_num: int = -1,
    ):
        """
        Args:
            data_dir (Path): Root directory where the data is saved. Each split should be in its own subdirectory.
            split (str): Dataset split name, including "train", "validation", and "test".
            fluent_frames (int): Frames numbers from the fluent (target) sequence to use as target.
            dtype: Data type for the arrays, default is np.float32.
            limited_num (int): Limit the number of samples to load; default -1 loads all samples.
        """
        self.data_dir = data_dir
        self.split = split
        self.fluent_frames = fluent_frames
        self.dtype = dtype

        # Store only file paths for now, load data on-the-fly
        # Each sample should have fluent (original), disfluent (updated), and metadata files
        self.examples = []
        split_dir = self.data_dir / split
        fluent_files = sorted(list(split_dir.glob(f"{split}_*_original.pose")))
        if limited_num > 0:
            fluent_files = fluent_files[
                :limited_num
            ]  # Limit the number of samples to load

        for fluent_file in fluent_files:
            # Construct corresponding disfluent and metadata file paths based on the file name
            disfluent_file = fluent_file.with_name(
                fluent_file.name.replace("_original.pose", "_updated.pose")
            )
            metadata_file = fluent_file.with_name(
                fluent_file.name.replace("_original.pose", "_metadata.json")
            )
            self.examples.append(
                {
                    "fluent_path": fluent_file,
                    "disfluent_path": disfluent_file,
                    "metadata_path": metadata_file,
                }
            )

        print(f"Dataset initialized with {len(self.examples)} samples. Split: {split}")

        # Initialize pose_header from the first fluent .pose file
        if self.examples:
            first_fluent_path = self.examples[0]["fluent_path"]
            try:
                with open(first_fluent_path, "rb") as f:
                    first_pose = Pose.read(f.read())
                    self.pose_header = first_pose.header
            except Exception as e:
                print(
                    f"[WARNING] Failed to read pose_header from {first_fluent_path}: {e}"
                )
                self.pose_header = None
        else:
            self.pose_header = None

        # Repeat the dataset for testing purposes
        self.examples = self.examples * 10
        print(f"Dataset expanded to {len(self.examples)} samples after repeating.")

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a sample from the dataset. For each sample, load the entire disfluent sequence as condition,
        and randomly sample a cip from the fluent sequence of fixed length (fluent_frames) as target.
        Args:
            idx (int): Index of the sample to retrieve.
        """
        sample = self.examples[idx]

        # Load pose sequences and metadata from disk
        with open(sample["fluent_path"], "rb") as f:
            fluent_pose = Pose.read(f.read())
        with open(sample["disfluent_path"], "rb") as f:
            disfluent_pose = Pose.read(f.read())
        with open(sample["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Before normalization: extract raw data for statistics
        fluent_data_raw = np.array(fluent_pose.body.data.astype(self.dtype))
        disfluent_data_raw = np.array(disfluent_pose.body.data.astype(self.dtype))

        print(f"[DEBUG][Before Norm] Fluent raw data mean: {fluent_data_raw.mean()}, std: {fluent_data_raw.std()}")
        print(f"[DEBUG][Before Norm] Disfluent raw data mean: {disfluent_data_raw.mean()}, std: {disfluent_data_raw.std()}")

        # Apply normalization by shoulders width first (spatial normalization)
        fluent_pose.normalize()
        disfluent_pose.normalize()

        # Apply full-pose normalization using global mean/std (scale normalization)
        fluent_pose = normalize_mean_std(fluent_pose)
        disfluent_pose = normalize_mean_std(disfluent_pose)

        fluent_data = np.array(fluent_pose.body.data.astype(self.dtype))
        # Use the entire disfluent sequence as condition
        disfluent_data = np.array(disfluent_pose.body.data.astype(self.dtype))

        print(f"[DEBUG][After Norm] Fluent data mean: {fluent_data.mean()}, std: {fluent_data.std()}")
        print(f"[DEBUG][After Norm] Disfluent data mean: {disfluent_data.mean()}, std: {disfluent_data.std()}")

        fluent_length = len(fluent_data)

        # If the fluent sequence is shorter than fluent_frames, pad it with the last frame
        if fluent_length < self.fluent_frames:
            fluent_padding = np.repeat(fluent_data[[-1]], self.fluent_frames - fluent_length, axis=0)
            fluent_clip = np.concatenate([fluent_data, fluent_padding], axis=0)
        else:
            fluent_clip = fluent_data[:self.fluent_frames]

        disfluent_seq = disfluent_data

        # Frame-level mask generation
        target_mask = np.any(fluent_clip != 0, axis=(1, 2, 3))  # shape: [T]

        return {
            "data": torch.tensor(
                fluent_clip, dtype=torch.float32
            ),  # Fluent target clip
            "conditions": {
                "input_sequence": torch.tensor(
                    disfluent_seq, dtype=torch.float32
                ),  # Full disfluent input
                "target_mask": torch.tensor(
                    target_mask, dtype=torch.bool
                ),  # Per-frame valid mask
                "metadata": metadata,
            },
        }


def example_dataset():
    """
    Example function to demonstrate the dataset class and its DataLoader.
    """
    # Create an instance of the dataset
    dataset = SignLanguagePoseDataset(
        data_dir=Path("/scratch/ronli/output"),
        split="train",
        fluent_frames=50,
        limited_num=128,
    )

    # Create a DataLoader using zero-padding collator
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
        collate_fn=zero_pad_collator,
    )

    # Flag to indicate whether to display batch information
    display_batch_info = True
    # Flag to indicate whether to measure data loading time
    measure_loading_time = True

    if display_batch_info:
        # Display shapes of a batch for debugging purposes
        batch = next(iter(dataloader))
        print("Batch size:", len(batch))
        print("Normalized target clip:", batch["data"].shape)
        print("Input sequence:", batch["conditions"]["input_sequence"].shape)
        print("Target mask:", batch["conditions"]["target_mask"].shape)

    if measure_loading_time:
        loading_times = []
        start_time = time.time()
        for batch in dataloader:
            end_time = time.time()
            batch_loading_time = end_time - start_time
            print(f"Data loading time for each iteration: {batch_loading_time:.4f}s")
            loading_times.append(batch_loading_time)
            start_time = end_time
        avg_loading_time = sum(loading_times) / len(loading_times)
        print(f"Average data loading time: {avg_loading_time:.4f}s")
        print(f"Total data loading time: {sum(loading_times):.4f}s")


# if __name__ == '__main__':
#     example_dataset()
