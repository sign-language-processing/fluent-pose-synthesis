# pylint: disable=protected-access, arguments-renamed
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import normalize_pose_size
from pose_anonymization.data.normalization import unnormalize_mean_std


from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion
from CAMDM.network.training import BaseTrainingPortal
from CAMDM.utils.common import mkdir


def masked_l2_per_sample(x: Tensor, y: Tensor, mask: Optional[Tensor] = None, reduce: bool = True) -> Tensor:
    """
    Compute masked L2 loss per sample. Correctly handles division by zero for fully masked samples.

    Args:
        x, y: Tensors of shape (batch_size, keypoints, dims, time_steps)
        mask: Tensor of shape (batch_size, keypoints, dims, time_steps).
              True = masked (invalid), False = valid.
        reduce: Whether to average the per-sample loss over the batch dimension.
    """
    diff_sq = (x - y) ** 2  # (B, K, D, T)

    if mask is not None:
        mask = mask.bool()  # Ensure boolean type
        # Invert mask: False (valid) -> 1.0, True (masked) -> 0.0
        valid_mask_elements = (~mask).float()
        # Apply mask to zero out loss contribution from invalid elements
        diff_sq = diff_sq * valid_mask_elements
    else:
        # If no mask is provided, all elements are considered valid
        valid_mask_elements = torch.ones_like(diff_sq)

    # Sum squared errors over all dimensions except batch for each sample
    per_sample_loss_sum = diff_sq.flatten(start_dim=1).sum(dim=1) # Shape: (B,)

    # Count the number of valid elements for each sample
    valid_elements_count = valid_mask_elements.flatten(start_dim=1).sum(dim=1) # Shape: (B,)

    # Compute mean squared error per sample
    # Clamp denominator to avoid division by zero (0/0 = NaN).
    # If valid_elements_count is 0, per_sample_loss_sum is also 0, resulting in 0 loss.
    per_sample_loss = per_sample_loss_sum / valid_elements_count.clamp(min=1) # Shape: (B,)

    if reduce:
        # Return the average loss across the batch
        return per_sample_loss.mean()
    else:
        # Return the loss for each sample in the batch
        return per_sample_loss


class PoseTrainingPortal(BaseTrainingPortal):
    def __init__(
        self,
        config: Any,
        model: torch.nn.Module,
        diffusion: GaussianDiffusion,
        dataloader: DataLoader,
        logger: Optional[Any],
        tb_writer: Optional[Any],
        finetune_loader: Optional[DataLoader] = None,
    ):
        """
        Training portal specialized for pose diffusion tasks.
        Args:
            config: Configuration object with trainer, architecture, etc.
            model: The pose diffusion model.
            diffusion: An instance of GaussianDiffusion or its subclass.
            dataloader: The main training dataloader.
            logger: Logger instance (optional).
            tb_writer: TensorBoard writer (optional).
            finetune_loader: Optional finetuning dataloader.
        """
        super().__init__(
            config, model, diffusion, dataloader, logger, tb_writer, finetune_loader
        )
        self.pose_header = None
        self.device = config.device

    def diffuse(
        self,
        x_start: Tensor,    # Target fluent chunk (from dataloader['data'])
        t: Tensor,  # Diffusion timesteps
        cond: Dict[str, Tensor],    # Conditioning inputs (from dataloader['conditions'])
        noise: Optional[Tensor] = None,
        return_loss: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        Performs one step of the diffusion process and optionally computes training losses.
        Args:
            x_start: Ground truth target pose chunk. Expected shape from loader: (B, T_chunk, K, D).
            t: Diffusion timesteps for each sample. Shape: (B,)
            cond: Conditioning inputs dict. Expected keys:
                  'input_sequence': Full disfluent sequence (B, T_disfl, K, D)
                  'previous_output': Fluent history (B, T_hist, K, D)
                  'target_mask': Mask for the target chunk (B, T_chunk, K, D)
            noise: Optional noise tensor. If None, generated internally.
            return_loss: Whether to compute and return training losses.
        """
        # 1. Permute x_start from (B, T_chunk, K, D) to (B, K, D, T_chunk)
        x_start = x_start.permute(0, 2, 3, 1).to(self.device) # (B, K, D, T_chunk)

        if noise is None:
            noise = torch.randn_like(x_start)

        # 2. Apply forward diffusion process: q_sample(x_start, t) -> x_t
        x_t = self.diffusion.q_sample(x_start, t.to(self.device), noise=noise) # (B, K, D, T_chunk)

        # 3. Prepare conditions for the model
        processed_cond = {}
        for key, val in cond.items():
            processed_cond[key] = val.to(self.device)

        # Permute sequence conditions to (B, K, D, T) expected by MotionProcess encoders
        processed_cond["input_sequence"] = processed_cond["input_sequence"].permute(0, 2, 3, 1) # (B, K, D, T_disfl)
        processed_cond["previous_output"] = processed_cond["previous_output"].permute(0, 2, 3, 1) # (B, K, D, T_hist)

        # 4. Call the model's interface
        model_output = self.model.interface(x_t, self.diffusion._scale_timesteps(t.to(self.device)), processed_cond)
        # Permute output back to (B, T_chunk, K, D) for consistency if not calculating loss
        model_output_original_shape = model_output.permute(0, 3, 1, 2)


        # 5. Compute Loss (if requested)
        if return_loss:
            loss_terms = {}

            # Determine the prediction target based on diffusion settings
            mmt = self.diffusion.model_mean_type
            if mmt.name == "PREVIOUS_X":
                target = self.diffusion.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0]
            elif mmt.name == "START_X":
                target = x_start # Target is the original clean chunk
            elif mmt.name == "EPSILON":
                target = noise # Target is the noise added
            else:
                raise ValueError(f"Unsupported model_mean_type: {mmt}")

            assert (model_output.shape == target.shape == x_start.shape), "Shape mismatch between model output, target, and x_start"

            # Process the target_mask
            mask_from_loader = processed_cond["target_mask"]
            # print(f"[DEBUG diffuse] Received target_mask shape: {mask_from_loader.shape}")

            # Adapt mask shape based on loader output (assuming B, T, K, D)
            mask = mask_from_loader.permute(0, 2, 3, 1) # -> (B, K, D, T_chunk)
            # print(f"[DEBUG diffuse] Final mask shape for loss: {mask.shape}")

            # Calculate loss only for samples that have at least one valid frame/point
            # Sum mask over K, D, T dimensions. Check if sum > 0 for each batch item.
            batch_has_valid = (mask.float().sum(dim=(1, 2, 3)) < mask.shape[1]*mask.shape[2]*mask.shape[3]) # Check if not all masked
            valid_batch_indices = batch_has_valid.nonzero().squeeze()

            if valid_batch_indices.numel() == 0:
                print("[WARNING] All samples in this batch are fully masked. Skipping loss computation.")
                # Returning zero loss
                zero_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
                loss_terms = {"loss": zero_loss, "loss_data": zero_loss, "loss_data_vel": zero_loss}
                # Need to return model_output still
                return model_output_original_shape, loss_terms

            # Calculate Loss Components
            # Use the already computed `mask` (shape B, K, D, T) where True=masked
            if self.config.trainer.use_loss_mse:
                loss_data = masked_l2_per_sample(target, model_output, mask, reduce=True)
                loss_terms["loss_data"] = loss_data

            if self.config.trainer.use_loss_vel:
                # Calculate velocity on time axis (last dimension)
                target_vel = target[..., 1:] - target[..., :-1]
                model_output_vel = model_output[..., 1:] - model_output[..., :-1]
                # Create mask for velocity (same shape as velocity)
                mask_vel = mask[..., 1:] if mask is not None else None

                loss_data_vel = masked_l2_per_sample(target_vel, model_output_vel, mask_vel, reduce=True)
                loss_terms["loss_data_vel"] = loss_data_vel

            # Calculate Total Loss
            total_loss = loss_terms.get("loss_data", 0.0) + loss_terms.get("loss_data_vel", 0.0)
            loss_terms["loss"] = total_loss
            print("Total loss:", total_loss.item()) # Loss is already scalar due to reduce=True

            return model_output_original_shape, loss_terms

        # If return_loss is False, just return the model output
        return model_output_original_shape, None

    def evaluate_sampling(
        self, dataloader: DataLoader, save_folder_name: str = "init_samples"
    ):
        """
        Perform inference and save generated samples from the model.
        This currently evaluates the model in a NON-AUTOREGRESSIVE way, predicting only the first chunk based on conditions.
        Args:
            dataloader (Dataloader): A DataLoader instance containing the evaluation dataset.
            save_folder_name (str): Folder to save generated pose files.
        """
        self.model.eval()
        self.model.training = False

        mkdir(f"{self.save_dir}/{save_folder_name}")

        patched_dataloader = DataLoader(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=self.config.trainer.workers,
            collate_fn=zero_pad_collator,
            pin_memory=True,
        )

        datas = next(iter(patched_dataloader))

        def get_original_dataset(dataset):
            while isinstance(dataset, torch.utils.data.Subset):
                dataset = dataset.dataset
            return dataset

        dataset = get_original_dataset(patched_dataloader.dataset)
        self.pose_header = dataset.pose_header

        fluent_clip = datas["data"].to(self.device)

        cond = {
            key: (val.to(self.device) if torch.is_tensor(val) else val)
            for key, val in datas["conditions"].items()
        }

        time, _ = self.schedule_sampler.sample(
            patched_dataloader.batch_size, self.device
        )
        with torch.no_grad():
            pred_output_tensor, _ = self.diffuse(
                fluent_clip, time, cond, noise=None, return_loss=False
            )
        fluent_clip_array = fluent_clip.cpu().numpy()
        pred_output_array = pred_output_tensor.cpu().numpy()

        unnormed_gt_list = self.export_samples(fluent_clip_array, f"{self.save_dir}/{save_folder_name}", "gt")
        unnormed_pred_list = self.export_samples(pred_output_array, f"{self.save_dir}/{save_folder_name}", "pred")

        # Save the normalized fluent clip and predicted output as numpy arrays
        np.save(f"{self.save_dir}/{save_folder_name}/gt_output_normed.npy", fluent_clip_array)
        np.save(f"{self.save_dir}/{save_folder_name}/pred_output_normed.npy", pred_output_array)
        # Save the unnormalized fluent clip and predicted output as numpy arrays
        unormed_gt_batch = np.stack(unnormed_gt_list, axis=0) # (B, T, K, D)
        unormed_pred_batch = np.stack(unnormed_pred_list, axis=0) # (B, T, K, D)
        np.save(f"{self.save_dir}/{save_folder_name}/gt_output.npy", unormed_gt_batch)
        np.save(f"{self.save_dir}/{save_folder_name}/pred_output.npy", unormed_pred_batch)

        if self.logger:
            self.logger.info(f"Evaluate sampling {save_folder_name} at epoch {self.epoch}")
        else:
            print(f"Evaluate sampling {save_folder_name} at epoch {self.epoch}")

    def export_samples(self, pose_output_normalized_np: np.ndarray, save_path: str, prefix: str) -> list:
        """
        Unnormalizes pose data using unnormalize_mean_std, exports to .pose format, and returns the unnormalized numpy data.
        Args:
            pose_output_normalized_np: A numpy array of normalized pose data.
            save_path: Path (string) where files will be saved.
            prefix: Prefix for file names, e.g., "gt" or "pred".
        """
        unnormalized_arrays = [] # Store unnormalized arrays here

        for i in range(pose_output_normalized_np.shape[0]):

            pose_array = pose_output_normalized_np[i] # (time, keypoints, 3)
            time, keypoints, dim = pose_array.shape
            pose_array = pose_array.reshape(time, 1, keypoints, dim)

            # Set confidence to all 1.0 for all keypoints
            confidence = np.ones((time, 1, keypoints), dtype=np.float32)

            # Wrap and export the pose data
            pose_body = NumPyPoseBody(fps=25, data=pose_array, confidence=confidence)
            pose_obj = Pose(self.pose_header, pose_body)

            # Unnormalize the pose data
            unnorm_pose = unnormalize_mean_std(pose_obj)

            file_path = f"{save_path}/pose_{i}.{prefix}.pose"
            with open(file_path, "wb") as f:
                unnorm_pose.write(f)
            # self.logger.info(f"Saved pose file: {file_path}")


            # Verify the file by reading it back
            with open(file_path, "rb") as f_check:
                Pose.read(f_check.read())

            # Extract and store the unnormalized numpy data
            unnorm_data_np = np.array(unnorm_pose.body.data.data.astype(pose_output_normalized_np.dtype)).squeeze(1) # (T, K, D)
            unnormalized_arrays.append(unnorm_data_np)

            # If error occurs, the file was not written correctly
            # self.logger.info(f"Pose file {file_path} read successfully.")
        return unnormalized_arrays
