import time
import json
import random
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
        chunk_len: int,
        dtype=np.float32,
        limited_num: int = -1,
    ):
        """
        Args:
            data_dir (Path): Root directory where the data is saved. Each split should be in its own subdirectory.
            split (str): Dataset split name, including "train", "validation", and "test".
            chunk_len (int): Frames numbers from the fluent (target) sequence to use as target (chunk length).
            dtype: Data type for the arrays, default is np.float32.
            limited_num (int): Limit the number of samples to load; default -1 loads all samples.
        """
        self.data_dir = data_dir
        self.split = split
        self.chunk_len = chunk_len
        self.dtype = dtype

        # Store only file paths for now, load data on-the-fly
        # Each sample should have fluent (original), disfluent (updated), and metadata files
        self.examples = []
        split_dir = self.data_dir / split
        fluent_files = sorted(list(split_dir.glob(f"{split}_*_original.pose")))
        if limited_num > 0:
            fluent_files = fluent_files[:limited_num]  # Limit the number of samples to load

        for fluent_file in fluent_files:
            # Construct corresponding disfluent and metadata file paths based on the file name
            disfluent_file = fluent_file.with_name(fluent_file.name.replace("_original.pose", "_updated.pose"))
            metadata_file = fluent_file.with_name(fluent_file.name.replace("_original.pose", "_metadata.json"))
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
                print(f"[WARNING] Failed to read pose_header from {first_fluent_path}: {e}")
                self.pose_header = None
        else:
            self.pose_header = None

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a sample from the dataset. For each sample, load the entire disfluent sequence as condition,
        and randomly sample a clip from the fluent sequence of fixed length (chunk_len) as target.
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

        # print(f"[DEBUG][Before Norm] Fluent raw data mean: {fluent_pose.body.data.mean(axis=(0, 1, 2))} std {fluent_pose.body.data.std(axis=(0, 1, 2))}")
        # print(f"[DEBUG][Before Norm] Disfluent raw data mean: {disfluent_pose.body.data.mean(axis=(0, 1, 2))} std {disfluent_pose.body.data.std(axis=(0, 1, 2))}")

        # Normalize the pose data
        fluent_pose = normalize_mean_std(fluent_pose)
        disfluent_pose = normalize_mean_std(disfluent_pose)

        # print(f"DEBUG][After Norm] Fluent normalized data mean:: {fluent_pose.body.data.mean(axis=(0, 1, 2))} std {fluent_pose.body.data.std(axis=(0, 1, 2))}")
        # print(f"[DEBUG][After Norm] Disfluent normalized data mean: {disfluent_pose.body.data.mean(axis=(0, 1, 2))} std {disfluent_pose.body.data.std(axis=(0, 1, 2))}")

        fluent_data = np.array(fluent_pose.body.data.astype(self.dtype))
        fluent_mask = fluent_pose.body.data.mask
        disfluent_data = np.array(disfluent_pose.body.data.astype(self.dtype))

        fluent_length = len(fluent_data)

        # 1. Randomly sample the start index for the fluent (target) chunk
        if fluent_length <= self.chunk_len:
            start_idx = 0
            target_len = fluent_length
            history_len = 0
        else:
            start_idx = random.randint(0, fluent_length - self.chunk_len)
            target_len = self.chunk_len
            history_len = start_idx

        # 2. Extract target chunk (y_k) and history chunk (y_1, ..., y_{k-1})
        target_chunk = fluent_data[start_idx : start_idx + target_len]
        target_mask = fluent_mask[start_idx : start_idx + target_len]

        if history_len > 0:
            history_chunk = fluent_data[:history_len]
        else:
            # MODIFICATION: Force minimum length of 1 for previous_output if empty
            history_chunk = np.zeros((1,) + fluent_data.shape[1:], dtype=self.dtype) # create a single empty frame
            # The purpose of this is to ensure the current collate_fn works
        # else:
        #     # No history chunk available, create an empty array with time dimension 0
        #     history_chunk = np.empty((0,) + fluent_data.shape[1:], dtype=self.dtype)

        # 3. Prepare the entire disfluent sequence as condition
        disfluent_seq = disfluent_data

        # 4. Pad target chunk if its actual length is less than chunk_len
        if target_chunk.shape[0] < self.chunk_len:
            pad_len = self.chunk_len - target_chunk.shape[0]
            # Padding 0s for target chunk
            padding_shape_data = (pad_len,) + target_chunk.shape[1:]
            target_padding = np.zeros(padding_shape_data, dtype=self.dtype)
            target_chunk = np.concatenate([target_chunk, target_padding], axis=0)
            # Padding for mask (True for masked)
            mask_padding = np.ones((pad_len,) + target_mask.shape[1:], dtype=bool)
            target_mask = np.concatenate([target_mask, mask_padding], axis=0)

        # 5. Convert numpy arrays to torch tensors
        target_chunk = torch.from_numpy(target_chunk.astype(np.float32))
        history_chunk = torch.from_numpy(history_chunk.astype(np.float32))
        disfluent_seq = torch.from_numpy(disfluent_seq.astype(np.float32))
        target_mask = torch.from_numpy(target_mask) # Bool tensor

        # 6. Squeeze person dimension
        target_chunk = target_chunk.squeeze(1)    # (T_chunk, K, D)
        history_chunk = history_chunk.squeeze(1)  # (T_hist, K, D)
        disfluent_seq = disfluent_seq.squeeze(1)  # (T_disfl, K, D)
        target_mask = target_mask.squeeze(1)      # (T_chunk, K, D)

        # 7. Create conditions dictionary
        # Later, zero_pad_collator will handle padding T_disfl and T_hist across the batch
        conditions = {
            "input_sequence": disfluent_seq,     # (T_disfl, K, D)
            "previous_output": history_chunk,    # (T_hist, K, D)
            "target_mask": target_mask           # (T_chunk, K, D)
        }

        # print(f"DEBUG Dataset idx {idx}:")
        # print(f"  target_chunk shape: {target_chunk.shape}")
        # print(f"  input_sequence shape: {disfluent_seq.shape}")
        # print(f"  previous_output shape: {history_chunk.shape}")
        # print(f"  target_mask shape: {target_mask.shape}")

        return {
            "data": target_chunk,   # (T_chunk, K, D)
            "conditions": conditions,
        }


def example_dataset():
    """
    Example function to demonstrate the dataset class and its DataLoader.
    """
    # Create an instance of the dataset
    dataset = SignLanguagePoseDataset(
        data_dir=Path("/scratch/ronli/fluent-pose-synthesis/pose_data/output"),
        split="train",
        chunk_len=40,
        limited_num=128,
    )

    # Create a DataLoader using zero-padding collator
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
        collate_fn=zero_pad_collator,
    )

    print(f"\n--- Example Batch Info (Batch Size: {dataloader.batch_size}) ---")

    batch = next(iter(dataloader))
    print("Batch Keys:", batch.keys())
    print("Conditions Keys:", batch['conditions'].keys())

    print("\nShapes:")
    print(f"  data (Target Chunk): {batch['data'].shape}")
    print(f"  conditions['input_sequence'] (Disfluent): {batch['conditions']['input_sequence'].shape} ")
    print(f"  conditions['previous_output'] (History): {batch['conditions']['previous_output'].shape} ")
    print(f"  conditions['target_mask']: {batch['conditions']['target_mask'].shape}")

    print("\nNormalization Stats (Shapes):")
    print(f"  Fluent Mean: {dataset.fluent_mean.shape}")
    print(f"  Fluent Std: {dataset.fluent_std.shape}")
    print(f"  Disfluent Mean: {dataset.disfluent_mean.shape}")
    print(f"  Disfluent Std: {dataset.disfluent_std.shape}")

    print("\nSample Values (first element of first sequence):")
    print(f"  Target Chunk (first 5 flattened): {batch['data'][0].flatten()[:5]}")
    # Check if history chunk is not empty
    if batch["conditions"]["previous_output"].shape[1] > 0:
        print(f"  History Chunk (first 5 flattened): {batch['conditions']['previous_output'][0].flatten()[:5]}")
    else:
        print("  History Chunk: Empty")
    print(f"  Disfluent Seq (first 5 flattened): {batch['conditions']['input_sequence'][0].flatten()[:5]}")
    print(f"  Target Mask (first 5 flattened): {batch['conditions']['target_mask'][0].flatten()[:5]}")


# if __name__ == '__main__':
#     example_dataset()


# Example Output:
# Dataset initialized with 128 samples. Split: train
# Batch Keys: dict_keys(['data', 'conditions'])
# Conditions Keys: dict_keys(['input_sequence', 'previous_output', 'target_mask'])

# Shapes:
#   data (Target Chunk): torch.Size([32, 40, 178, 3])
#   conditions['input_sequence'] (Disfluent): torch.Size([32, 359, 178, 3])
#   conditions['previous_output'] (History): torch.Size([32, 110, 178, 3])
#   conditions['target_mask']: torch.Size([32, 40, 178, 3])

# Normalization Stats (Shapes):
#   Fluent Mean: torch.Size([1, 178, 3])
#   Fluent Std: torch.Size([1, 178, 3])
#   Disfluent Mean: torch.Size([1, 178, 3])
#   Disfluent Std: torch.Size([1, 178, 3])

# Sample Values (first element of first sequence):
#   Target Chunk (first 5 flattened): tensor([ 9.0694e-02,  7.7781e-01, -7.0343e+02,  1.9091e-01, -8.7535e-01])
#   History Chunk (first 5 flattened): tensor([0., 0., 0., 0., 0.])
#   Disfluent Seq (first 5 flattened): tensor([ 0.1327,  1.0505, -1.7174,  0.2764, -0.7866])
#   Target Mask (first 5 flattened): tensor([False, False, False, False, False])
