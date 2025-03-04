import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, Union, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset
from pose_format import Pose
from pose_format.torch.masked.collator import zero_pad_collator


class SignLanguagePoseDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, fluent_frames: int, dtype=np.float32, limited_num: int=-1):
        """
        Args:
            data_dir (Path): Root directory where the data is saved. Each split should be in its own subdirectory.
            split (str): Dataset split name, including "train", "validation", and "test".
            fluent_frames (int): Number of frames from the fluent (target) sequence (original pose sequence) to use as target.
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
            fluent_files = fluent_files[:limited_num]   # Limit the number of samples to load

        for fluent_file in fluent_files:
            # Construct corresponding disfluent and metadata file paths based on the file name
            disfluent_file = fluent_file.with_name(fluent_file.name.replace("_original.pose", "_updated.pose"))
            metadata_file = fluent_file.with_name(fluent_file.name.replace("_original.pose", "_metadata.json"))
            self.examples.append({
                'fluent_path': fluent_file,
                'disfluent_path': disfluent_file,
                'metadata_path': metadata_file
            })
        
        print(f"Dataset initialized with {len(self.examples)} samples. Split: {split}")

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
        with open(sample['fluent_path'], 'rb') as f:
            fluent_pose = Pose.read(f.read())
        with open(sample['disfluent_path'], 'rb') as f:
            disfluent_pose = Pose.read(f.read())
        with open(sample['metadata_path'], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Use the entire disfluent sequence as condition
        disfluent_seq = disfluent_pose.body.data.astype(self.dtype)
        disfluent_mask = disfluent_pose.body.mask
        
        fluent_data = fluent_pose.body.data.astype(self.dtype)
        fluent_mask_full = fluent_pose.body.mask
        fluent_length = len(fluent_data)
        
        if fluent_length > self.fluent_frames:
            # Dynamic windowing: randomly select a window of length fluent_frames
            start = np.random.randint(0, fluent_length - self.fluent_frames + 1)
            fluent_clip = fluent_data[start: start + self.fluent_frames]
            fluent_mask = fluent_mask_full[start: start + self.fluent_frames]
        else:
            # If the fluent sequence is shorter than fluent_frames, zero_pad_collator will handle it
            fluent_clip = fluent_data
            fluent_mask = fluent_mask_full

        return {
            'data': torch.tensor(fluent_clip, dtype=torch.float32),
            'conditions': {
                'input_sequence': torch.tensor(disfluent_seq, dtype=torch.float32),   # entire disfluent sequence
                'input_mask': torch.tensor(disfluent_mask, dtype=torch.bool),           # mask for the disfluent sequence
                'target_mask': torch.tensor(fluent_mask, dtype=torch.bool),             # mask for the fluent clip
                'metadata': metadata
            }
        }


def example_dataset():
    """
    Example function to demonstrate the dataset class and its DataLoader.
    """
    # Create an instance of the dataset
    dataset = SignLanguagePoseDataset(
        data_dir=Path("/scratch/ronli/output"),
        split="train",
        fluent_frames=20,
        limited_num=1000
    )
    
    # Create a DataLoader using zero-padding collator
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=0, 
        drop_last=False, 
        pin_memory=True,
        collate_fn=zero_pad_collator
    )

    # Flag to indicate whether to display batch information
    display_batch_info = True
    # Flag to indicate whether to measure data loading time
    measure_loading_time = True

    if display_batch_info:
        # Display shapes of a batch for debugging purposes
        batch = next(iter(dataloader))
        print("Target clip shape:", batch['data'].shape)             # Fluent (target) clip shape
        print("Input clip shape:", batch['conditions']['input_sequence'].shape)   # Disfluent (condition) sequence shape
        print("Input mask shape:", batch['conditions']['input_mask'].shape)
        print("Target mask shape:", batch['conditions']['target_mask'].shape)
        # print("Metadata:", batch['conditions']['metadata'])

    if measure_loading_time:
        loading_times = []
        start_time = time.time()
        for batch in dataloader:
            end_time = time.time()
            batch_loading_time = end_time - start_time
            print('Data loading time for each iteration: {:.4f}s'.format(batch_loading_time))
            loading_times.append(batch_loading_time)
            start_time = end_time
        avg_loading_time = sum(loading_times) / len(loading_times)
        print('Average data loading time: {:.4f}s'.format(avg_loading_time))
        print('Total data loading time: {:.4f}s'.format(sum(loading_times)))


if __name__ == '__main__':
    example_dataset()