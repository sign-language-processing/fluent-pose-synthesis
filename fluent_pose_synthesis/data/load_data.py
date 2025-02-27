import sys
import time
import json
from pathlib import Path
from typing import Any, Dict

sys.path.append('./')

import numpy as np
import torch
from torch.utils.data import Dataset
from pose_format import Pose


class SignLanguagePoseDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, disfluent_frame: int, fluent_frame: int, offset_frame: int = 1, dtype=np.float32, limited_num: int = -1):
        """
        Args:
            data_dir (Path): Root directory where the data is saved. Each split should be in its own subdirectory.
            split (str): Dataset split name, including "train", "validation", and "test".
            disfluent_frame (int): Number of frames from the potentially disfluent sequence (updated pose sequence) to use as condition.
            fluent_frame (int): Number of frames from the fluent (target) sequence (original pose sequence) to use as target.
            offset_frame (int): Stride to use when generating sliding windows (default is 1).
            dtype: Data type for the arrays, default is np.float32.
            limited_num (int): Limit the number of samples to load; default -1 loads all samples.
        """
        self.data_dir = data_dir
        self.split = split
        self.disfluent_frame = disfluent_frame
        self.fluent_frame = fluent_frame
        self.window_size = disfluent_frame + fluent_frame
        self.offset_frame = offset_frame
        self.dtype = dtype

        # Load all samples for the given split
        # Each sample should have fluent (original), disfluent (updated), and metadata files
        self.examples = []
        split_dir = self.data_dir / split
        fluent_files = sorted(list(split_dir.glob(f"{split}_*_original.pose")))
        if limited_num > 0:
            fluent_files = fluent_files[:limited_num]   # Limit the number of samples to load

        frame_nums_list = []  # To record frame counts per sample
        for fluent_file in fluent_files:
            # Construct corresponding disfluent and metadata file paths based on the file name
            disfluent_file = fluent_file.with_name(fluent_file.name.replace("_original.pose", "_updated.pose"))
            metadata_file = fluent_file.with_name(fluent_file.name.replace("_original.pose", "_metadata.json"))
            # Load pose objects
            with open(fluent_file, 'rb') as f:
                fluent_pose = Pose.read(f.read())
            with open(disfluent_file, 'rb') as f:
                disfluent_pose = Pose.read(f.read())
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.examples.append({
                'fluent': fluent_pose,
                'disfluent': disfluent_pose,
                'metadata': metadata
            })
            # Record the frame count for this sample
            min_length = min(fluent_pose.body.data.shape[0], disfluent_pose.body.data.shape[0])
            frame_nums_list.append(min_length)

        # Generate sliding window indices using vectorized method
        # Create a list to hold indices arrays for each sample
        self.clip_indices_list = []
        sample_idx = 0
        for i, example in enumerate(self.examples):
            # Get the number of frames in the fluent sequence; Skip if the sequence is too short
            min_length = min(example['fluent'].body.data.shape[0], example['disfluent'].body.data.shape[0])
            if min_length < self.window_size:   
                continue
            # Generate sliding window start indices with the given offset
            clip_indices = np.arange(0, min_length - self.window_size + 1, self.offset_frame)[:, None] + np.arange(self.window_size)
            # Prepend the sample index (sample_idx) to each clip index row
            clip_indices_with_idx = np.hstack((np.full((len(clip_indices), 1), i, dtype=clip_indices.dtype), clip_indices))
            self.clip_indices_list.append(clip_indices_with_idx)
            sample_idx += 1
        # Concatenate all clip indices from different samples
        self.clip_indices = np.concatenate(self.clip_indices_list, axis=0)

        total_clips = len(self.clip_indices)
        total_frames = sum(frame_nums_list)
        print('Dataset loaded, trained with %d clips, %d frames, %d mins in total' % (total_clips, total_frames, total_frames/25/60))   # All sequences are sampled at 25fps

    def __len__(self) -> int:
        """
        Returns the total number of sliding-window clips in the dataset.
        """
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a sliding-window clip from the dataset.
        Args:
            idx (int): Index of the clip to retrieve.
        """
        # Each row of clip_indices: [example_index, frame0, frame1, ..., frame(window_size-1)]
        indices = self.clip_indices[idx]
        ex_idx = indices[0]
        # The remaining indices are the frame indices for this clip
        clip_frame_indices = indices[1:]
        example = self.examples[ex_idx]
        
        # Slice the pose sequences using the precomputed indices
        disfluent_seq = example['disfluent'].body.data[clip_frame_indices].astype(self.dtype)
        fluent_seq = example['fluent'].body.data[clip_frame_indices].astype(self.dtype)
        
        input_clip = disfluent_seq[:self.disfluent_frame]
        target_clip = fluent_seq[self.disfluent_frame:]
                
        # Get the mask for the fluent pose sequence over these frames
        mask = example['fluent'].body.mask[clip_frame_indices]
        
        return {
            'data': target_clip,
            'conditions': {
                'input_clip': input_clip,
                'mask': mask,
                'metadata': example['metadata']
            }
        }


def custom_collate(batch):
    """
    Custom collate function to handle the metadata field properly.
    Stacks the numerical arrays while merging metadata dicts into one dict, where each key maps to a list of values from the batch.
    """
    collated = {}
    # Stack the 'data' field (target clips)
    collated['data'] = np.stack([item['data'] for item in batch])
    conditions = {}
    # Stack input_clip and mask (both are NumPy arrays of equal shape)
    conditions['input_clip'] = np.stack([item['conditions']['input_clip'] for item in batch])
    conditions['mask'] = np.stack([item['conditions']['mask'] for item in batch])
    
    # Merge metadata (each is a dict) into a single dict where each key maps to a list of values
    merged_metadata = {}
    for item in batch:
        meta: dict = item['conditions']['metadata']
        for key, value in meta.items():
            if key not in merged_metadata:
                merged_metadata[key] = []
            merged_metadata[key].append(value)
    conditions['metadata'] = merged_metadata
    
    collated['conditions'] = conditions
    
    return collated


def test_dataset():
    """
    Test the dataset class and its DataLoader.
    """
    # Create an instance of the dataset
    dataset = SignLanguagePoseDataset(
        data_dir=Path("/scratch/ronli/output"),
        split="train",
        disfluent_frame=10,
        fluent_frame=45,
        offset_frame=1,
        limited_num=100
    )
    
    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=0, 
        drop_last=False, 
        pin_memory=True,
        collate_fn=custom_collate
    )

    do_export_test = True
    do_loop_test = True

    if do_export_test:
        # Example: Print shapes of a batch.
        batch = next(iter(dataloader))
        print("Target clip shape:", batch['data'].shape)    # Fluent (target) clip
        print("Input clip shape:", batch['conditions']['input_clip'].shape)    # Disfluent (input) clip
        print("Mask shape:", batch['conditions']['mask'].shape)
        # print("Metadata:", batch['conditions']['metadata'])

    if do_loop_test:
        times = []
        start_time = time.time()
        for batch in dataloader:
            end_time = time.time()
            print('Data loading time for each iteration: {:.4f}s'.format(end_time - start_time))
            times.append(end_time - start_time)
            start_time = end_time
        avg_time = sum(times) / len(times)
        print('Average data loading time: {:.4f}s'.format(avg_time))
        print('Entire data loading time: {:.4f}s'.format(sum(times)))


if __name__ == '__main__':
    test_dataset()