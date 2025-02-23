import os
import math
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import Dataset
from pose_format import Pose
from pose_format.torch.pose_body import TorchPoseBody

from process_mediapipe_faces import mediapipe_face_regions_dict, MEDIAPIPE_FACE_KEYPOINTS


class SignLanguagePoseDataset(Dataset):
    """
    A dataset class for loading Sign Language Pose data.
    """
    def __init__(self, data_dir: Path, split: str, past_frame: int, future_frame: int, augmentation: bool = True):
        """
        Initialize the dataset with the data directory and window size.
        Args:
            data_dir (Path): The root data directory.
            split (str): The dataset split (train, validation, test).
            past_frame (int): Number of past frames to include.
            future_frame (int): Number of future frames to include.
            augmentation (bool): Whether to apply data augmentations.
        """
        self.data_dir = data_dir / split
        self.past_frame = past_frame
        self.future_frame = future_frame
        self.window_size = past_frame + future_frame
        self.augmentation = augmentation
        self.offset_frame = 1  # Offset for clip extraction
        self.clips = self._discover_clips()
        print(f"Dataset loaded | Total clips: {len(self.clips)} | Window Size: {self.window_size} frames")

    def _discover_clips(self) -> list:
        """
        Discover all valid clips from the data directory.
        For each pose file, if both the updated and original pose files have a frame count
        greater than or equal to window_size, generate multiple clip entries using a specified offset.
        """
        clips = []
        for pose_file in self.data_dir.glob("*_updated.pose"):
            original_pose_file = pose_file.with_name(pose_file.name.replace("_updated", "_original"))
            metadata_file = pose_file.with_name(pose_file.name.replace("_updated.pose", "_metadata.json"))
            if original_pose_file.exists() and metadata_file.exists():
                # Use _load_pose method to load updated and original poses
                updated_pose = self._load_pose(pose_file)
                original_pose = self._load_pose(original_pose_file)
                total_frames_updated = updated_pose.body.data.tensor.shape[0]
                total_frames_original = original_pose.body.data.tensor.shape[0]
                if min(total_frames_updated, total_frames_original) < self.window_size:
                    continue
                # Use the minimum of the two frame counts for valid clip extraction
                total_frames = min(total_frames_updated, total_frames_original)
                for start_idx in range(0, total_frames - self.window_size + 1, self.offset_frame):
                    clips.append({
                        "updated": pose_file,
                        "original": original_pose_file,
                        "metadata": metadata_file,
                        "start_idx": start_idx
                    })
        
        return clips
    
    def _split_clip(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Split the continuous clip into past and future segments.
        """
        past = clip[:self.past_frame]
        future = clip[self.past_frame:]
        
        return {"past": past, "future": future}
    
    def _load_pose(self, file_path: Path) -> Pose:
        """
        Load a pose file as TorchPoseBody object.
        Args:
            file_path (Path): The path to the pose file.
        """
        with open(file_path, "rb") as f:
            data_buffer = f.read()
        
        return Pose.read(data_buffer, TorchPoseBody)
    
    def _get_sliding_window(self, pose: Pose) -> torch.Tensor:
        """
        Extract a continuous clip from the pose sequence using a sliding window.
        Args:
            pose (Pose): Pose object containing the sequence.
        """
        pose_data_tensor = pose.body.data.tensor    # Extract the pose data tensor from MaskedBody
        total_frames = pose_data_tensor.shape[0]
        print("total_frames", total_frames)
        if total_frames < self.window_size:
            raise ValueError(f"Pose sequence length {total_frames} is less than required window size {self.window_size}")
        # For fixed clip extraction, we use the provided start_idx from clip info (handled in __getitem__)
        # This function will simply be used to extract a clip if needed.
        start_idx = random.randint(0, total_frames - self.window_size)
        return pose_data_tensor[start_idx : start_idx + self.window_size]

    def _apply_augmentation(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation on the clip:
        1. Random small-angle rotation (simulate slight errors).
        2. Optional Gaussian smoothing over time.
        Args:
            clip (torch.Tensor): Clip tensor of shape (window_size, N, 3) where N is the number of keypoints.
        """
        angle = random.uniform(-5, 5) * (math.pi / 180)  # Convert degrees to radians
        angle = torch.tensor(angle, dtype=clip.dtype, device=clip.device)  # Ensure tensor format
        
        # 3D rotation around Z-axis (most natural for sign language, since signing occurs in front of the body)
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        rotation_matrix = torch.tensor([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=clip.dtype, device=clip.device)  # 3x3 rotation matrix

        # Reshape clip to apply rotation correctly
        T, B, N, D = clip.shape  # (window_size, batch, num_keypoints, 3)
        clip_reshaped = clip.view(-1, 3)  # Flatten to (T * B * N, 3)

        # Apply rotation
        augmented_clip = torch.matmul(clip_reshaped, rotation_matrix.T)  # (T * B * N, 3)
        augmented_clip = augmented_clip.view(T, B, N, D)  # Reshape back to original shape

        # Randomly apply Gaussian smoothing
        if random.random() < 0.5:
            augmented_clip = gaussian_filter1d_torch(augmented_clip, sigma=1)

        return augmented_clip

    def _parse_pose(self, pose: Pose) -> Dict[str, Any]:
        """
        Parse the pose data and return a structured dictionary.
        Args:
            pose (Pose): The Pose object.
        """
        data_dict = {}
        # The pose components to extract
        components = {
            "body": "POSE_LANDMARKS",
            "left_hand": "LEFT_HAND_LANDMARKS",
            "right_hand": "RIGHT_HAND_LANDMARKS",
            "face": "FACE_LANDMARKS"
        }
        for part, component_name in components.items():
            comp = next((c for c in pose.header.components if c.name == component_name), None)
            # Generate a mapping of point names to indices
            point_map = {p: i for i, p in enumerate(comp.points)} if comp is not None else {}
            # Body and hands: Extract keypoints for body, left hand, and right hand
            if part in ["body", "left_hand", "right_hand"]:
                indices = list(point_map.values())
                if indices:
                    data_dict[part] = {
                        "xy": pose.body.data[:, :, indices, :2].squeeze(1),
                        "conf": pose.body.confidence[:, :, indices].squeeze(1)
                    }
            # Face: Extract facial landmarks for different regions
            elif part == "face":
                face_data = {}
                for region, region_indices_set in mediapipe_face_regions_dict.items():
                    indices = [i for i, point in enumerate(comp.points) if int(point) in region_indices_set]
                    print(f"[DEBUG] Processing region '{region}', expected keypoints: {region_indices_set}, found indices: {indices}")
                    if indices:
                        face_data[region] = {
                            "xy": pose.body.data[:, :, indices, :2].squeeze(1),
                            "conf": pose.body.confidence[:, :, indices].squeeze(1)
                        }
                print(f"[DEBUG] Extracted face data keys: {list(face_data.keys())}")
            data_dict["face"] = face_data
    
        return convert_masked_to_tensor(data_dict)

    def compute_sliding_window_aggregate(self, x: torch.Tensor, win_size: int, step: int = 1) -> torch.Tensor:
        """
        Compute sliding window aggregate statistics for a given time series.
        Statistics include mean, standard deviation, minimum, and maximum.
        Args:
            x (torch.Tensor): Input time series of shape (T,) or (T, F) or (T, ...).
            win_size (int): Window size.
            step (int): Step size.
        """
        # Flatten all dimensions except the time dimension (dimension 0)
        if x.dim() > 1:
            T = x.shape[0]
            # Flatten the rest of the dimensions into a single dimension F
            x = x.reshape(T, -1)
        # If the tensor is 1D, add a feature dimension so shape becomes (T, 1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)        
        # Create sliding windows along the time dimension.
        # windows shape: (num_windows, win_size, F)
        windows = x.unfold(0, win_size, step)        
        # Compute statistics along the window (time) dimension
        mean_val = windows.mean(dim=1)
        std_val = windows.std(dim=1)
        min_val, _ = windows.min(dim=1)
        max_val, _ = windows.max(dim=1)        
        # Concatenate the statistics: shape (num_windows, 4*F)
        stats = torch.cat([mean_val, std_val, min_val, max_val], dim=1)
       
        # Return the mean statistics over all windows, resulting in a 1D tensor.
        return stats.mean(dim=0)

    def extract_prosody_features(self, pose: Pose, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Extract prosody/annotation features including:
            1. Kinematic features: velocity, acceleration, curvature, trajectory smoothness
            2. Facial dynamic features: eyebrow movement, mouth opening, blink rate
            3. Hand movement: gesture speed, smoothness
            4. Pause detection (pause_mask)
        Args:
            pose (Pose): The Pose object.
            data_dict (Dict): Parsed pose data.
        """
        features = {}

        # ====== Compute kinematic features (for body, hands, and facial regions) ======
        def compute_kinematics(data: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Compute velocity, acceleration, curvature, and smoothness for a given data tensor.
            Args:
                data (torch.Tensor): Input data tensor.
            """
            print("Before conversion:", type(data))
            data = data.tensor if hasattr(data, "tensor") else data
            print("After conversion:", type(data))

            velocity = torch.diff(data, dim=0, n=1)
            acceleration = torch.diff(data, dim=0, n=2)
            dx = velocity[:acceleration.shape[0]]  # Match acceleration's first dim
            curvature = torch.abs(dx[..., 0] * acceleration[..., 1] - dx[..., 1] * acceleration[..., 0])
            # Ensure normalization shape matches
            norm_dx = torch.linalg.norm(dx, dim=-1).unsqueeze(-1) + 1e-6  # Avoid div by zero
            curvature = curvature / (norm_dx ** 3).squeeze(-1)
            smoothed_trajectory = gaussian_filter1d_torch(data, sigma=5)

            return {
                "velocity": velocity,
                "acceleration": acceleration,
                "curvature": curvature.unsqueeze(-1),
                "smoothness": smoothed_trajectory[2:]
            }

        features["body"] = compute_kinematics(data_dict["body"]["xy"])
        features["left_hand"] = compute_kinematics(data_dict["left_hand"]["xy"])
        features["right_hand"] = compute_kinematics(data_dict["right_hand"]["xy"])

        # For the face part, first build a relative index mapping for each region.
        # We use the FACE_LANDMARKS component from pose.header.components.
        face_comp = next((c for c in pose.header.components if c.name == "FACE_LANDMARKS"), None)
        # Build a mapping for each region: {absolute keypoint id: relative index within that region}
        region_index_maps = {}
        for region in mediapipe_face_regions_dict.keys():
            region_index_map = {}
            for i, point in enumerate(face_comp.points):
                try:
                    p_int = int(point)  # Convert string to integer
                except Exception as e:
                    print(f"[DEBUG] Error converting point {point}: {e}")
                    continue
                if p_int in mediapipe_face_regions_dict[region]:
                    if p_int not in region_index_map:
                        region_index_map[p_int] = len(region_index_map)
            region_index_maps[region] = region_index_map
            print(f"[DEBUG] Region '{region}' index mapping: {region_index_map}")
        
        # In _parse_pose, the face data for each region is extracted using relative indices.
        features["face"] = {}
        if "face" in data_dict:
            for region, region_feats in data_dict["face"].items():
                features["face"][region] = compute_kinematics(region_feats["xy"])
        else:
            print("[DEBUG] Warning: no 'face' data in data_dict, skipping face kinematic feature extraction")

        # ====== Compute facial prosody features (eyebrows, mouth, eyes) ======
        # Eyebrow movement features (raw time series per frame)
        left_brow = data_dict["face"]["left_eyebrow"]["xy"][:, :, 1]  # Y-coordinate for left eyebrow, shape: (T, N)
        right_brow = data_dict["face"]["right_eyebrow"]["xy"][:, :, 1]  # Y-coordinate for right eyebrow, shape: (T, N)
        brow_movement_left = torch.abs(left_brow - left_brow[:, 0:1]).mean(dim=1)  # Shape: (T,)
        brow_movement_right = torch.abs(right_brow - right_brow[:, 0:1]).mean(dim=1)  # Shape: (T,)
        features["brow_movement"] = (brow_movement_left + brow_movement_right) / 2  # Shape: (T,)

        brow_velocity_left = torch.abs(torch.diff(left_brow, dim=0)).mean(dim=1)  # Shape: (T-1,)
        brow_velocity_right = torch.abs(torch.diff(right_brow, dim=0)).mean(dim=1)  # Shape: (T-1,)
        features["brow_movement_velocity"] = (brow_velocity_left + brow_velocity_right) / 2  # Shape: (T-1,)

        # Build a mapping for the lips region using the FACE_LANDMARKS component
        face_comp = next((c for c in pose.header.components if c.name == "FACE_LANDMARKS"), None)
        if face_comp is None:
            raise ValueError("FACE_LANDMARKS component not found in pose header")
        lips_index_map = {}
        for point in face_comp.points:
            if int(point) in mediapipe_face_regions_dict["lips"]:
                if int(point) not in lips_index_map:
                    lips_index_map[int(point)] = len(lips_index_map)
        # Retrieve indices for required lips keypoints using MEDIAPIPE_FACE_KEYPOINTS
        upper_lip_center = lips_index_map[MEDIAPIPE_FACE_KEYPOINTS["upper_lip_center"]]
        lower_lip_center = lips_index_map[MEDIAPIPE_FACE_KEYPOINTS["lower_lip_center"]]
        upper_lip_left = lips_index_map[MEDIAPIPE_FACE_KEYPOINTS["upper_lip_left"]]
        lower_lip_left = lips_index_map[MEDIAPIPE_FACE_KEYPOINTS["lower_lip_left"]]
        upper_lip_right = lips_index_map[MEDIAPIPE_FACE_KEYPOINTS["upper_lip_right"]]
        lower_lip_right = lips_index_map[MEDIAPIPE_FACE_KEYPOINTS["lower_lip_right"]]
        lip_left_corner = lips_index_map[MEDIAPIPE_FACE_KEYPOINTS["lip_left_corner"]]
        lip_right_corner = lips_index_map[MEDIAPIPE_FACE_KEYPOINTS["lip_right_corner"]]

        lips = data_dict["face"]["lips"]["xy"]
        # Compute mouth open distance as the sum of vertical distances in three regions (raw time series)
        mouth_open_dist = (
            torch.norm(lips[:, upper_lip_center, :] - lips[:, lower_lip_center, :], dim=1) +
            torch.norm(lips[:, upper_lip_left, :] - lips[:, lower_lip_left, :], dim=1) +
            torch.norm(lips[:, upper_lip_right, :] - lips[:, lower_lip_right, :], dim=1)
        )
        # Compute mouth width as a normalization factor (raw time series)
        mouth_width = torch.norm(lips[:, lip_left_corner, :] - lips[:, lip_right_corner, :], dim=1)
        # Compute mouth open ratio (avoid division by zero) as a raw time series per frame
        features["mouth_open_ratio"] = mouth_open_dist / (mouth_width + 1e-6)

        # Compute a raw blink indicator per frame
        left_eye = data_dict["face"]["left_eye"]["xy"]
        right_eye = data_dict["face"]["right_eye"]["xy"]
        left_eye_upper_idx = region_index_maps["left_eye"][MEDIAPIPE_FACE_KEYPOINTS["left_eye_upper"]]
        left_eye_lower_idx = region_index_maps["left_eye"][MEDIAPIPE_FACE_KEYPOINTS["left_eye_lower"]]
        right_eye_upper_idx = region_index_maps["right_eye"][MEDIAPIPE_FACE_KEYPOINTS["right_eye_upper"]]
        right_eye_lower_idx = region_index_maps["right_eye"][MEDIAPIPE_FACE_KEYPOINTS["right_eye_lower"]]
        left_eye_open = torch.abs(left_eye[:, left_eye_upper_idx, 1] - left_eye[:, left_eye_lower_idx, 1])
        right_eye_open = torch.abs(right_eye[:, right_eye_upper_idx, 1] - right_eye[:, right_eye_lower_idx, 1])
        eye_opening = (left_eye_open + right_eye_open) / 2  # Shape: (T,)
        blink_indicator = (torch.diff(eye_opening, dim=0) < -torch.std(eye_opening)).float()  # Shape: (T-1,)
        features["eye_blink_indicator"] = blink_indicator

        # Compute pause mask based on body velocity (raw time series)
        speed_magnitude = torch.linalg.norm(features["body"]["velocity"], dim=-1)
        features["pause_mask"] = (speed_magnitude < 0.01).float()

        return features

    def concatenate_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        For each dynamic feature, compute sliding window aggregate statistics and concatenate them into one vector.
        Args:
            features (Dict): Dictionary containing raw time series features.
        """
        # Define sliding window size for aggregation (tunable parameter)
        stat_win = max(1, self.window_size // 2)
        agg_features = []

        def process_feature_array(arr: torch.Tensor) -> torch.Tensor:
            if arr.dim() == 1:
                arr = arr.unsqueeze(-1)
            return self.compute_sliding_window_aggregate(arr, win_size=stat_win, step=1)

        # Process kinematic features for body, left_hand, right_hand
        for part in ["body", "left_hand", "right_hand"]:
            for key, val in features[part].items():
                agg_feat = process_feature_array(val)
                agg_features.append(agg_feat)

        # Process kinematic features for each face region
        for region, region_feats in features["face"].items():
            for key, val in region_feats.items():
                agg_feat = process_feature_array(val)
                agg_features.append(agg_feat)

        # Process eyebrow features
        for key in ["brow_movement", "brow_movement_velocity"]:
            agg_feat = process_feature_array(features[key])
            agg_features.append(agg_feat)

        # Process mouth open ratio
        agg_feat = process_feature_array(features["mouth_open_ratio"])
        agg_features.append(agg_feat)

        # Process eye blink indicator
        agg_feat = process_feature_array(features["eye_blink_indicator"])
        agg_features.append(agg_feat)

        # Process pause mask (simply take its mean since it is binary)
        agg_features.append(torch.tensor([features["pause_mask"].mean()]))

        # Concatenate all aggregated features into one vector
        concatenated = torch.cat(agg_features, dim=0)
        
        return concatenated

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        Args:
            idx (int): The index of the sample.
        """
        clip_info = self.clips[idx]
        
        # Load the updated and original poses
        updated_pose = self._load_pose(clip_info["updated"])
        original_pose = self._load_pose(clip_info["original"])

        total_frames_updated = updated_pose.body.data.tensor.shape[0]
        total_frames_original = original_pose.body.data.tensor.shape[0]
        if min(total_frames_updated, total_frames_original) < self.window_size:
            print(f"Skipping clip {idx} due to insufficient frames")
            return self.__getitem__((idx + 1) % len(self.clips))
        
        start_idx = clip_info["start_idx"]
        # Extract a fixed window clip based on start_idx
        updated_clip = updated_pose.body.data.tensor[start_idx : start_idx + self.window_size]
        original_clip = original_pose.body.data.tensor[start_idx : start_idx + self.window_size]

        print(f"🔹 Clip {idx}: Pose Loaded, start_idx: {start_idx}")
        print(f"   - Updated Clip Type: {type(updated_clip)}")
        print(f"   - Updated Clip Shape: {getattr(updated_clip, 'shape', 'N/A')}")

        if hasattr(updated_clip, "tensor"):
            updated_clip = updated_clip.tensor  # Convert MaskedTensor to Tensor
        if hasattr(original_clip, "tensor"):
            original_clip = original_clip.tensor
        
        # Ensure the clips are torch.Tensors (convert if necessary)
        if not isinstance(updated_clip, torch.Tensor):
            updated_clip = torch.tensor(updated_clip)
        if not isinstance(original_clip, torch.Tensor):
            original_clip = torch.tensor(original_clip)
        
        # Apply augmentation to the updated clip (simulate correction process)
        if self.augmentation:
            updated_clip = self._apply_augmentation(updated_clip)
        
        # Split the clips into past (history) and future segments
        updated_segments = self._split_clip(updated_clip)
        original_segments = self._split_clip(original_clip)
        
        # Parse the raw pose data to extract keypoint structures
        updated_parsed = self._parse_pose(updated_pose)
        original_parsed = self._parse_pose(original_pose)
        
        # Slice the parsed data to a fixed temporal length (window_size)
        updated_parsed = slice_parsed_data(updated_parsed, self.window_size)
        original_parsed = slice_parsed_data(original_parsed, self.window_size)
        
        # Extract detailed prosody and kinematic features
        updated_features = self.extract_prosody_features(updated_pose, updated_parsed)
        original_features = self.extract_prosody_features(original_pose, original_parsed)
        
        # Aggregate features via sliding window statistics
        updated_features_concat = self.concatenate_features(updated_features)
        original_features_concat = self.concatenate_features(original_features)
        
        return {
            "updated_full": updated_clip,       # Full clip from updated (potentially unfluent) sequence
            "original_full": original_clip,       # Full clip from original (fluent) sequence
            "updated_past": updated_segments["past"],
            "updated_future": updated_segments["future"],
            "original_past": original_segments["past"],
            "original_future": original_segments["future"],
            "updated_pose_parsed": updated_parsed,
            "original_pose_parsed": original_parsed,
            "updated_features": updated_features,
            "original_features": original_features,
            "updated_features_concat": updated_features_concat,
            "original_features_concat": original_features_concat,
            "pause_mask": updated_features["pause_mask"]
        }

    def __len__(self) -> int:
        return len(self.clips)
    

def gaussian_filter1d_torch(x: torch.Tensor, sigma: float, kernel_size: int = None) -> torch.Tensor:
    """
    Apply a 1D Gaussian filter to a tensor along the first dimension.
    Args:
        x (torch.Tensor): Input tensor of shape (T, N, D) or (T, 1, N, D), where T is the time dimension.
        sigma (float): Standard deviation of the Gaussian.
        kernel_size (int, optional): Size of the Gaussian kernel. Defaults to int(6*sigma+1).
    """
    print("x.shape", x.shape)
    print("x.dtype", x.dtype)
    print("x.dim()", x.dim())

    if x.dim() == 3:
        x = x.unsqueeze(1)  # If input is (T, N, D), change to (T, 1, N, D) adding a batch dimension

    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

    # Create a 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size)  # (out_channels=1, in_channels=1, kernel_size)

    # Ensure x has correct shape for conv1d
    T, B, N, D = x.shape  # (time, batch, num_keypoints, D)
    
    # Adjust kernel_size and padding if T is too small
    if T < kernel_size:
        kernel_size = T if T % 2 == 1 else T - 1
        # Re-create the kernel with the new kernel_size
        coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size)
    
    padding = kernel_size // 2

    x_reshaped = x.permute(1, 3, 2, 0)  # Change to (batch, D, N, T)
    x_reshaped = x_reshaped.reshape(B * D * N, 1, T)  # Merge dimensions for conv1d

    # Ensure correct padding
    x_padded = F.pad(x_reshaped, (padding, padding), mode='reflect')

    # Apply Gaussian filter with conv1d
    filtered = F.conv1d(x_padded, kernel)  # Shape: (B * D * N, 1, T)

    # Reshape back to original shape
    filtered = filtered.view(B, D, N, T).permute(3, 0, 2, 1)  # (T, B, N, D)

    return filtered

def convert_masked_to_tensor(obj):
    if hasattr(obj, "tensor"):
        return obj.tensor
    elif isinstance(obj, dict):
        return {k: convert_masked_to_tensor(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_masked_to_tensor(item) for item in obj]
    else:
        return obj


def slice_parsed_data(data: Any, window_size: int) -> Any:
    """
    Recursively slice tensors in the parsed pose data along the time dimension.
    If a value is a tensor and its first dimension is greater than window_size,
    it is sliced. If the value is a dict or list, process recursively.
    Args:
        data: The parsed pose data dictionary.
        window_size: The window size for slicing.
    """
    if isinstance(data, torch.Tensor):
        if data.shape[0] > window_size:
            return data[:window_size]
        else:
            return data
    elif isinstance(data, dict):
        return {k: slice_parsed_data(v, window_size) for k, v in data.items()}
    elif isinstance(data, list):
        return [slice_parsed_data(item, window_size) for item in data]
    else:
        return data