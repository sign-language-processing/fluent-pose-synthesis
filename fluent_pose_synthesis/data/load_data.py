import time
import json
import random
from pathlib import Path
from typing import Any, Dict

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from pose_format import Pose
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import normalize_mean_std
import pickle
import hashlib


class SignLanguagePoseDataset(Dataset):

    def __init__(
        self,
        data_dir: Path,
        split: str,
        chunk_len: int,
        dtype=np.float32,
        history_len: int = 5,
        limited_num: int = -1,
        use_cache: bool = True,
        cache_dir: Path = None,
        force_reload: bool = False,
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
        self.history_len = history_len
        self.window_len = chunk_len + history_len
        self.dtype = dtype

        # Cache controls
        self.use_cache = use_cache
        self.force_reload = force_reload
        # Determine cache directory
        if cache_dir is None:
            self.cache_dir = self.data_dir / "cache"
        else:
            self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        # Collect data file mtimes for invalidation
        split_dir = self.data_dir / self.split
        all_files = list(split_dir.glob(f"{split}_*_original.pose")) + \
                    list(split_dir.glob(f"{split}_*_updated.pose")) + \
                    list(split_dir.glob(f"{split}_*_metadata.json"))
        mtimes = [f.stat().st_mtime for f in all_files if f.exists()]
        data_mtime = max(mtimes) if mtimes else 0
        # Build cache key
        cache_params = {
            'data_dir': str(data_dir),
            'split': split,
            'chunk_len': chunk_len,
            'history_len': history_len,
            'dtype': str(dtype),
            'limited_num': limited_num,
            'data_mtime': data_mtime,
        }
        cache_key = hashlib.md5(str(cache_params).encode()).hexdigest()
        self.cache_file = self.cache_dir / f"dataset_cache_{split}_{cache_key}.pkl"
        # Try loading from cache
        if self.use_cache and not self.force_reload and self.cache_file.exists():
            print(f"Loading dataset from cache: {self.cache_file}")
            self._load_from_cache()
            print(f"Dataset loaded from cache: {len(self.examples)} samples, split={split}")
            return

        # Store only file paths for now, load data on-the-fly
        # Each sample should have fluent (original), disfluent (updated), and metadata files
        self.examples = []
        split_dir = self.data_dir / split
        fluent_files = sorted(list(split_dir.glob(f"{split}_*_original.pose")))
        if limited_num > 0:
            fluent_files = fluent_files[:limited_num]  # Limit the number of samples to load

        for fluent_file in tqdm(fluent_files, desc=f"Loading {split} examples"):
            # Construct corresponding disfluent and metadata file paths based on the file name
            disfluent_file = fluent_file.with_name(fluent_file.name.replace("_original.pose", "_updated.pose"))
            metadata_file = fluent_file.with_name(fluent_file.name.replace("_original.pose", "_metadata.json"))
            self.examples.append({
                "fluent_path": fluent_file,
                "disfluent_path": disfluent_file,
                "metadata_path": metadata_file,
            })

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

        self.fluent_clip_list = []
        self.fluent_mask_list = []
        self.disfluent_clip_list = []

        self.train_indices = []

        for example_idx, example in enumerate(
                tqdm(self.examples, desc=f"Processing pose files for {split}", total=len(self.examples))):
            with open(example["fluent_path"], "rb") as f:
                fluent_pose = Pose.read(f.read())
            with open(example["disfluent_path"], "rb") as f:
                disfluent_pose = Pose.read(f.read())

            fluent_data = np.array(fluent_pose.body.data.astype(self.dtype))
            fluent_mask = fluent_pose.body.data.mask
            disfluent_data = np.array(disfluent_pose.body.data.astype(self.dtype))
            fluent_length = fluent_data.shape[0]

            self.fluent_clip_list.append(fluent_data[:, 0])
            self.fluent_mask_list.append(fluent_mask[:, 0])
            self.disfluent_clip_list.append(disfluent_data[:, 0])

        if self.split == "validation":
            self.train_indices = np.arange(len(self.examples)).reshape(-1, 1)
        else:
            for example_idx, example in enumerate(
                    tqdm(self.examples, desc=f"Building indices for {split}", total=len(self.examples))):
                fluent_data = self.fluent_clip_list[example_idx]
                fluent_length = fluent_data.shape[0]

                if fluent_length >= self.chunk_len:
                    zero_indices = np.array([-1] * self.history_len + list(range(self.chunk_len))).reshape(1, -1)
                    clip_indices = np.arange(0, fluent_length - self.window_len + 1, 1)[:, None] + np.arange(
                        self.window_len)
                    clip_indices = np.concatenate((zero_indices, clip_indices), axis=0)
                    clip_indices_with_idx = np.hstack((
                        np.full(
                            (len(clip_indices), 1),
                            example_idx,
                            dtype=clip_indices.dtype,
                        ),
                        clip_indices,
                    ))
                else:
                    zero_indices = np.array([-1] * self.history_len + list(range(fluent_length)) + [-2] *
                                            (self.chunk_len - fluent_length)).reshape(1, -1)
                    clip_indices_list = []
                    for i in range(self.window_len):
                        is_history_part = i < self.history_len
                        if i < fluent_length:
                            clip_indices_list.append(i)
                        else:
                            if is_history_part:
                                clip_indices_list.append(-1)
                            else:
                                clip_indices_list.append(-2)
                    clip_indices = np.array(clip_indices_list).reshape(1, -1)
                    clip_indices = np.concatenate((zero_indices, clip_indices), axis=0)
                    clip_indices_with_idx = np.hstack((
                        np.full(
                            (len(clip_indices), 1),
                            example_idx,
                            dtype=clip_indices.dtype,
                        ),
                        clip_indices,
                    ))

                self.train_indices.append(clip_indices_with_idx)

            self.train_indices = np.concatenate(self.train_indices, axis=0)

        concatenated_fluent_clips = np.concatenate(self.fluent_clip_list, axis=0)
        self.input_mean = concatenated_fluent_clips.mean(axis=0, keepdims=True)  # axis=0
        self.input_std = concatenated_fluent_clips.std(axis=0, keepdims=True)

        concatenated_disfluent_clips = np.concatenate(self.disfluent_clip_list, axis=0)
        self.condition_mean = concatenated_disfluent_clips.mean(axis=0, keepdims=True)
        self.condition_std = concatenated_disfluent_clips.std(axis=0, keepdims=True)

        for i in range(len(self.examples)):
            self.fluent_clip_list[i] = (self.fluent_clip_list[i] - self.input_mean) / self.input_std
            self.disfluent_clip_list[i] = (self.disfluent_clip_list[i] - self.condition_mean) / self.condition_std

        # Save cache for future runs
        if self.use_cache:
            print(f"Saving dataset to cache: {self.cache_file}")
            self._save_to_cache()
        print("Dataset initialized with {} samples. Split: {}".format(len(self.examples), split))

    def _save_to_cache(self):
        """Serialize dataset to cache file."""
        data = {
            'examples': self.examples,
            'pose_header': self.pose_header,
            'fluent_clip_list': self.fluent_clip_list,
            'fluent_mask_list': self.fluent_mask_list,
            'disfluent_clip_list': self.disfluent_clip_list,
            'train_indices': self.train_indices,
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'condition_mean': self.condition_mean,
            'condition_std': self.condition_std,
        }
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[WARNING] Failed to save cache: {e}")

    def _load_from_cache(self):
        """Load dataset from cache file."""
        try:
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
            self.examples = data['examples']
            self.pose_header = data['pose_header']
            self.fluent_clip_list = data['fluent_clip_list']
            self.fluent_mask_list = data['fluent_mask_list']
            self.disfluent_clip_list = data['disfluent_clip_list']
            self.train_indices = data['train_indices']
            self.input_mean = data['input_mean']
            self.input_std = data['input_std']
            self.condition_mean = data['condition_mean']
            self.condition_std = data['condition_std']
        except Exception as e:
            print(f"[WARNING] Failed to load cache, rebuilding: {e}")
            # Fall back to fresh build

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.train_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a sample from the dataset. For each sample, load the entire disfluent sequence as condition,
        and randomly sample a clip from the fluent sequence of fixed length (chunk_len) as target.
        Args:
            idx (int): Index of the sample to retrieve.
        """
        # Load metadata JSON for the current example to access original lengths
        motion_idx = self.train_indices[idx][0]
        metadata_path = self.examples[motion_idx]["metadata_path"]
        with open(metadata_path, "r", encoding="utf-8") as mf:
            meta_json = json.load(mf)
        orig_fluent_len = meta_json.get("fluent_pose_length", None)
        orig_disfluent_len = meta_json.get("disfluent_pose_length", None)

        if self.split == "validation":
            motion_idx = self.train_indices[idx][0]
            full_seq = torch.from_numpy(self.fluent_clip_list[motion_idx].astype(np.float32))
            disfluent_seq = torch.from_numpy(self.disfluent_clip_list[motion_idx].astype(np.float32))

            # Construct previous_output for validation
            history_len = self.history_len
            num_keypoints = self.fluent_clip_list[motion_idx].shape[1]  # K
            num_dims = self.fluent_clip_list[motion_idx].shape[2]  # D

            # Create a zero tensor for previous_output
            # Its shape should be (history_len, K, D)
            previous_output = torch.zeros((history_len, num_keypoints, num_dims), dtype=full_seq.dtype)

            metadata = {
                "original_example_index": int(motion_idx), "fluent_pose_length": orig_fluent_len,
                "disfluent_pose_length": orig_disfluent_len
            }
            result = {
                "data": full_seq,  # Full sequence, mainly used as reference for validation metrics
                "conditions": {
                    "input_sequence": disfluent_seq,  # Disfluent sequence as condition
                    "previous_output": previous_output,  # Now the initial history is all zeros
                },
                "full_fluent_reference": full_seq,  # Full reference for DTW evaluation
                "metadata": metadata,
            }
            return result

        item_frame_indice = self.train_indices[idx]
        motion_idx, frame_indices = item_frame_indice[0], item_frame_indice[1:]
        history_indices = frame_indices[:self.history_len]
        target_indices = frame_indices[self.history_len:]

        history_chunk = self.fluent_clip_list[motion_idx][history_indices]
        disfluent_seq = self.disfluent_clip_list[motion_idx]
        # Process target_chunk and target_mask, set frame at -2 to all-zero frame with mask True, others remain unchanged
        target_chunk_frames = []
        target_mask_frames = []
        single_frame_shape = self.fluent_clip_list[motion_idx][0].shape
        single_mask_shape = self.fluent_mask_list[motion_idx][0].shape
        for t_idx in target_indices:
            if t_idx == -2:
                target_chunk_frames.append(np.zeros(single_frame_shape, dtype=np.float32))
                target_mask_frames.append(np.ones(single_mask_shape, dtype=bool))
            else:
                target_chunk_frames.append(self.fluent_clip_list[motion_idx][t_idx])
                target_mask_frames.append(self.fluent_mask_list[motion_idx][t_idx])
        target_chunk = np.stack(target_chunk_frames, axis=0)
        target_mask = np.stack(target_mask_frames, axis=0)

        history_chunk[history_indices == -1].fill(0)

        # Convert numpy arrays to torch tensors
        target_chunk = torch.from_numpy(target_chunk.astype(np.float32))
        history_chunk = torch.from_numpy(history_chunk.astype(np.float32))
        disfluent_seq = torch.from_numpy(disfluent_seq.astype(np.float32))
        target_mask = torch.from_numpy(target_mask)  # Bool tensor

        # Create conditions dictionary
        conditions = {
            "input_sequence": disfluent_seq,  # (T_disfl, K, D)
            "previous_output": history_chunk,  # (T_hist, K, D)
            "target_mask": target_mask,  # (T_chunk, K, D)
        }

        # Get motion_idx, which points to the original sample index in self.examples
        item_frame_indice = self.train_indices[idx]  # Assume split is not 'test', or test also uses train_indices
        motion_idx = item_frame_indice[0]

        # Create metadata dictionary
        metadata = {
            "original_example_index": int(motion_idx),  # Ensure it is Python int type
            "original_disfluent_filepath": str(self.examples[motion_idx]["disfluent_path"]),
            "fluent_pose_length": orig_fluent_len,
            "disfluent_pose_length": orig_disfluent_len
        }

        # Build base return dictionary
        result = {
            "data": target_chunk,  # (T_chunk, K, D)
            "conditions": conditions,
            "metadata": metadata,
        }
        # If validation set, append full fluent reference sequence
        if self.split == "validation":
            # Take the full sequence from pre-normalized fluent_clip_list and convert to tensor
            full_seq = torch.from_numpy(self.fluent_clip_list[motion_idx].astype(np.float32))
            result["full_fluent_reference"] = full_seq  # (T_full, K, D)
        return result


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
    print("Conditions Keys:", batch["conditions"].keys())

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
