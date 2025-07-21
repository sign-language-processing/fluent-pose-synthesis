# pylint: disable=protected-access, arguments-renamed
import itertools
import time
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
import torch.nn as nn
from tqdm import tqdm

from pose_format import Pose
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
from pose_evaluation.metrics.pose_processors import NormalizePosesProcessor

from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion
from CAMDM.network.training import BaseTrainingPortal
from CAMDM.utils.common import mkdir


class _ConditionalWrapper(nn.Module):
    """Wraps a base model and a fixed conditioning dict, forwarding only (x, t)."""

    def __init__(self, base_model: nn.Module, cond: dict):
        super().__init__()
        self.base_model = base_model
        self.cond = cond

    def forward(self, x, t, **kwargs):
        # Ignore incoming kwargs, use fixed cond
        return self.base_model.interface(x, t, self.cond)


def move_to_device(val, device):
    if torch.is_tensor(val):
        return val.to(device)
    elif isinstance(val, dict):
        return {k: move_to_device(v, device) for k, v in val.items()}
    else:
        return val


def masked_l2_per_sample(x: Tensor, y: Tensor, mask: Optional[Tensor] = None, reduce: bool = True) -> Tensor:
    """
    Compute masked L2 loss per sample. Correctly handles division by zero for fully masked samples.

    Args:
        x, y: Tensors of shape (batch_size, keypoints, dims, time_steps)
        mask: Tensor of shape (batch_size, keypoints, dims, time_steps).
              True = masked (invalid), False = valid.
        reduce: Whether to average the per-sample loss over the batch dimension.
    """
    diff_sq = (x - y)**2  # (B, K, D, T)

    if mask is not None:
        mask = mask.bool()  # Ensure boolean type
        # Invert mask: False (valid) -> 1.0, True (masked) -> 0.0
        valid_mask_elements = (~mask).float()
        diff_sq = diff_sq * valid_mask_elements
    else:
        valid_mask_elements = torch.ones_like(diff_sq)

    per_sample_loss_sum = diff_sq.flatten(start_dim=1).sum(dim=1)  # Shape: (B,)

    valid_elements_count = valid_mask_elements.flatten(start_dim=1).sum(dim=1)  # Shape: (B,)
    per_sample_loss = per_sample_loss_sum / valid_elements_count.clamp(min=1)  # Shape: (B,)

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
        dataloader: DataLoader,  # Training dataloader
        logger: Optional[Any],
        tb_writer: Optional[Any],
        validation_dataloader: Optional[DataLoader] = None,
        prior_loader: Optional[DataLoader] = None,
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
            validation_dataloader: Optional validation dataloader.
            prior_loader: Optional prior dataloader.
        """
        super().__init__(config, model, diffusion, dataloader, logger, tb_writer, prior_loader)
        self.pose_header = None
        self.device = config.device
        self.validation_dataloader = validation_dataloader
        self.best_validation_metric = float("inf")

        # Initialize DTW metric calculator
        default_dtw_dist_val = 0.0
        self.validation_metric_calculator = DistanceMetric(
            name="Validation DTW",
            distance_measure=DTWDTAIImplementationDistanceMeasure(
                name="dtaiDTW",
                use_fast=True,
                default_distance=default_dtw_dist_val,
            ),
            pose_preprocessors=[NormalizePosesProcessor()],
        )
        self.logger.info(f"Initialized DTW metric with default_distance: {default_dtw_dist_val}")

        # Store normalization statistics from the training dataset for unnormalization
        self.data_input_mean = torch.tensor(dataloader.dataset.input_mean, device=self.device,
                                            dtype=torch.float32).squeeze()
        self.data_input_std = torch.tensor(dataloader.dataset.input_std, device=self.device,
                                           dtype=torch.float32).squeeze()

        # Store pose_header from validation dataset (for saving poses)
        self.val_pose_header = self.validation_dataloader.dataset.pose_header
        if self.val_pose_header:
            self.logger.info("Pose header loaded from validation dataset.")

    def diffuse(
        self,
        x_start: Tensor,  # Target fluent chunk (from dataloader['data'])
        t: Tensor,  # Diffusion timesteps
        cond: Dict[str, Tensor],  # Conditioning inputs (from dataloader['conditions'])
        noise: Optional[Tensor] = None,
        return_loss: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        Performs one step of the diffusion process and optionally computes training losses.
        Args:
            x_start: Ground truth target pose chunk. Expected shape from loader: (B, T_chunk, K, D).
            t: Diffusion timesteps for each sample. Shape: (B,)
            cond: Conditioning inputs dict. Expected keys:
                  "input_sequence": Full disfluent sequence (B, T_disfl, K, D)
                  "previous_output": Fluent history (B, T_hist, K, D)
                  "target_mask": Mask for the target chunk (B, T_chunk, K, D)
            noise: Optional noise tensor. If None, generated internally.
            return_loss: Whether to compute and return training losses.
        """
        # 1. Permute x_start from (B, T_chunk, K, D) to (B, K, D, T_chunk)
        x_start = x_start.permute(0, 2, 3, 1).to(self.device)  # (B, K, D, T_chunk)

        if noise is None:
            noise = torch.randn_like(x_start)

        # 2. Apply forward diffusion process: q_sample(x_start, t) -> x_t
        x_t = self.diffusion.q_sample(x_start, t.to(self.device), noise=noise)  # (B, K, D, T_chunk)

        # 3. Prepare conditions for the model
        processed_cond = {}
        for key, val in cond.items():
            processed_cond[key] = move_to_device(val, self.device)

        # Permute sequence conditions to (B, K, D, T) expected by MotionProcess encoders
        processed_cond["input_sequence"] = processed_cond["input_sequence"].permute(0, 2, 3, 1)  # (B, K, D, T_disfl)
        processed_cond["previous_output"] = processed_cond["previous_output"].permute(0, 2, 3, 1)  # (B, K, D, T_hist)

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
                target = x_start  # Target is the original clean chunk
            elif mmt.name == "EPSILON":
                target = noise  # Target is the noise added
            else:
                raise ValueError(f"Unsupported model_mean_type: {mmt}")

            assert (model_output.shape == target.shape ==
                    x_start.shape), "Shape mismatch between model output, target, and x_start"

            # Retrieve the optional target_mask which flags padded or invalid frames (True=masked)
            mask_from_loader = processed_cond.get("target_mask", None)
            if mask_from_loader is not None:
                # Permute mask shape from (B, T_chunk, K, D) to (B, K, D, T_chunk)
                mask = mask_from_loader.permute(0, 2, 3, 1)
            else:
                # Create mask based on original fluent lengths
                original_lengths = cond.get("metadata", {}).get("fluent_pose_length", None)
                if original_lengths is not None:
                    original_lengths = original_lengths.to(x_start.device)  # Shape: (B,)
                    B, K, D, T_padded = x_start.shape
                    time_idx = torch.arange(T_padded, device=x_start.device).unsqueeze(0).expand(B, -1)  # (B, T_padded)
                    mask_for_time = time_idx >= original_lengths.unsqueeze(1)  # (B, T_padded)
                    mask = mask_for_time.unsqueeze(1).unsqueeze(1).expand(B, K, D, T_padded)
                else:
                    mask = torch.zeros_like(x_start)
                    print("[WARNING] No target_mask provided. Using zero mask (no frames masked).")

            batch_has_valid = (mask.float().sum(dim=(1, 2, 3))
                               < mask.shape[1] * mask.shape[2] * mask.shape[3])  # Check if not all masked
            valid_batch_indices = batch_has_valid.nonzero().squeeze()

            if valid_batch_indices.numel() == 0:
                print("[WARNING] All samples in this batch are fully masked. Skipping loss computation.")
                # Returning zero loss
                zero_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
                loss_terms = {
                    "loss": zero_loss,
                    "loss_data": zero_loss,
                    "loss_data_vel": zero_loss,
                }
                # Need to return model_output still
                return model_output_original_shape, loss_terms

            # Calculate Loss Components
            # Use the already computed `mask` (shape B, K, D, T) where True=masked
            if self.config.trainer.use_loss_mse:
                loss_data = masked_l2_per_sample(target, model_output, mask, reduce=True)
                # loss_data = torch.nn.MSELoss()(model_output, target)
                loss_terms["loss_data"] = loss_data

            # Velocity Loss
            lambda_vel = getattr(self.config.trainer, "lambda_vel", 1.0)
            if self.config.trainer.use_loss_vel:
                # Compute first-order difference (velocity)
                target_vel = target[..., 1:] - target[..., :-1]
                model_output_vel = model_output[..., 1:] - model_output[..., :-1]
                mask_vel = mask[..., 1:] if mask is not None else None
                loss_data_vel = masked_l2_per_sample(target_vel, model_output_vel, mask_vel, reduce=True)
                loss_terms["loss_data_vel"] = loss_data_vel

                # Optional: Acceleration Loss
                if getattr(self.config.trainer, "use_loss_accel", False):
                    lambda_accel = getattr(self.config.trainer, "lambda_accel", 1.0)
                    # Compute second-order difference (acceleration)
                    target_accel = target_vel[..., 1:] - target_vel[..., :-1]
                    model_output_accel = model_output_vel[..., 1:] - model_output_vel[..., :-1]
                    mask_accel = mask_vel[..., 1:] if mask_vel is not None else None
                    loss_data_accel = masked_l2_per_sample(target_accel, model_output_accel, mask_accel, reduce=True)
                    loss_terms["loss_data_accel"] = loss_data_accel

            # Compute Weighted Total Loss
            # Initialize a new zero tensor for total_loss to avoid in-place side effects
            total_loss = torch.tensor(0.0, device=self.device)

            # Accumulate each component without using in-place operations
            if "loss_data" in loss_terms:
                total_loss = total_loss + loss_terms["loss_data"]

            lambda_vel = getattr(self.config.trainer, "lambda_vel", 1.0)
            if "loss_data_vel" in loss_terms:
                total_loss = total_loss + (lambda_vel * loss_terms["loss_data_vel"])

            lambda_accel = getattr(self.config.trainer, "lambda_accel", 1.0)
            if "loss_data_accel" in loss_terms:
                total_loss = total_loss + (lambda_accel * loss_terms["loss_data_accel"])

            # Store the combined result
            loss_terms["loss"] = total_loss

            return model_output_original_shape, loss_terms

        # If return_loss is False, just return the model output
        return model_output_original_shape, None

    def _run_validation_epoch(self) -> Optional[float]:
        """
        Runs validation using helper methods.
        """
        if self.validation_dataloader is None:
            if self.logger:
                self.logger.info("Validation dataloader not provided. Skipping validation.")
            return None

        self.model.eval()
        with torch.no_grad():
            references, predictions = [], []
            for batch_idx, batch_data in enumerate(self.validation_dataloader):
                batch_refs, batch_preds = self._process_validation_batch(batch_data, batch_idx)
                references.extend(batch_refs)
                predictions.extend(batch_preds)

        if not references:
            if self.logger:
                self.logger.warning("No poses collected during validation for DTW calculation.")
            self.model.train()
            return float("inf")

        if self.logger:
            self.logger.info(f"Calculating DTW for {len(references)} validation samples...")
        dtw_score = self._compute_dtw_score(predictions, references)
        self.model.train()
        return dtw_score

    def _process_validation_batch(self, batch_data: Dict[str, Any], batch_idx: int) -> Tuple[List[Pose], List[Pose]]:
        """
        Process a single validation batch: generate sequences, unnormalize, and build Pose objects.
        """
        with torch.no_grad():
            # 1. Autoregressive inference (extracted from original implementation)
            gt_fluent_full_loader = batch_data["full_fluent_reference"].to(self.device)
            disfluent_cond_seq_loader = batch_data["conditions"]["input_sequence"].to(self.device)
            initial_history_loader = batch_data["conditions"]["previous_output"].to(self.device)
            # Permute formats and prepare history
            disfluent_cond_seq = disfluent_cond_seq_loader.permute(0, 2, 3, 1)
            current_history = initial_history_loader.permute(0, 2, 3, 1)
            history_len = getattr(self.config.arch, "history_len", 10)
            # Trim or pad history
            if current_history.shape[3] > history_len:
                current_history = current_history[:, :, :, -history_len:]
            elif current_history.shape[3] < history_len:
                padding = torch.zeros(current_history.shape[0], current_history.shape[1], current_history.shape[2],
                                      history_len - current_history.shape[3], device=self.device)
                current_history = torch.cat([padding, current_history], dim=3)
            # Prepare autoregressive generation
            K = self.config.arch.keypoints
            D_feat = self.config.arch.dims
            # DYNAMIC MAX_LEN CALCULATION
            # Get the lengths of the disfluent sequences from the batch metadata
            disfluent_lengths = batch_data["metadata"]["disfluent_pose_length"].to(self.device)
            # Apply 200-frame upper bound to disfluent lengths
            len_for_calc = torch.min(disfluent_lengths, torch.tensor(200.0, device=self.device))
            # Linear regression-based mapping from disfluent to fluent length
            slope_a = 0.32  # coefficient `a` from regression
            intercept_b = 16.37  # intercept `b` from regression
            target_fluent_lengths = (slope_a * len_for_calc + intercept_b).clamp(min=1).int()
            # The max_len for this batch is the longest required fluent length
            max_len = torch.max(target_fluent_lengths).item()
            if self.logger and batch_idx == 0:  # Log only for the first batch to avoid spam
                self.logger.info(
                    f'Validation sample {batch_idx}: Original disfluent length: {disfluent_lengths.item()}, Length used for calc: {len_for_calc.item()}, Dynamic target length: {max_len} frames'
                )

            chunk_size = getattr(self.config.trainer, "validation_chunk_size", self.config.arch.chunk_len)
            stop_thresh = getattr(self.config.trainer, "validation_stop_threshold", 1e-4)
            generated = torch.empty(current_history.shape[0], K, D_feat, 0, device=self.device)
            active = torch.ones(current_history.shape[0], dtype=torch.bool, device=self.device)
            num_steps = (max_len + chunk_size - 1) // chunk_size
            for _ in range(num_steps):
                if not active.any():
                    break
                n_frames = min(chunk_size, max_len - generated.shape[3])
                target_shape = (current_history.shape[0], K, D_feat, n_frames)
                # Use classifier-free guidance
                cond_dict = {"input_sequence": disfluent_cond_seq, "previous_output": current_history}
                guidance_scale = getattr(self.config.trainer, "guidance_scale", 2.0)
                # Unconditional input: zero out the disfluent sequence
                uncond_disfluent_seq = torch.zeros_like(disfluent_cond_seq)
                uncond_dict = {"input_sequence": uncond_disfluent_seq, "previous_output": current_history}

                # Conditional sampling
                wrapped_model_cond = _ConditionalWrapper(self.model, cond_dict)
                cond_chunk = self.diffusion.p_sample_loop(
                    model=wrapped_model_cond, shape=target_shape,
                    clip_denoised=getattr(self.config.diff, "clip_denoised",
                                          False), model_kwargs={"y": cond_dict}, progress=False)
                # Unconditional sampling
                wrapped_model_uncond = _ConditionalWrapper(self.model, uncond_dict)
                uncond_chunk = self.diffusion.p_sample_loop(
                    model=wrapped_model_uncond, shape=target_shape,
                    clip_denoised=getattr(self.config.diff, "clip_denoised",
                                          False), model_kwargs={"y": uncond_dict}, progress=False)
                # Combine with guidance scale
                chunk = uncond_chunk + guidance_scale * (cond_chunk - uncond_chunk)
                # Stop condition
                if chunk.numel() > 0:
                    mean_abs = chunk.abs().mean(dim=(1, 2, 3))
                    stopped = (mean_abs < stop_thresh) & active
                    chunk[stopped] = 0
                    active = active & (~stopped)
                generated = torch.cat([generated, chunk], dim=3)
                # Update history
                if generated.shape[3] >= history_len:
                    current_history = generated[:, :, :, -history_len:]
                else:
                    pad = torch.zeros(generated.shape[0], K, D_feat, history_len - generated.shape[3],
                                      device=self.device)
                    current_history = torch.cat([pad, generated], dim=3)
            # Permute back
            pred_normed = generated.permute(0, 3, 1, 2)

            # 2. Unnormalize sequences
            val_ds = self.validation_dataloader.dataset
            val_mean = torch.tensor(val_ds.input_mean, device=self.device).view(1, 1, K, D_feat)
            val_std = torch.tensor(val_ds.input_std, device=self.device).view(1, 1, K, D_feat)
            train_mean = self.data_input_mean.view(1, 1, K, D_feat)
            train_std = self.data_input_std.view(1, 1, K, D_feat)
            gt_unnorm = gt_fluent_full_loader * val_std + val_mean
            pred_unnorm = pred_normed * train_std + train_mean

            # 3. Build Pose lists
            refs, preds = [], []
            for i in range(gt_unnorm.shape[0]):
                # Retrieve original reference length to truncate padded frames
                original_lengths = batch_data["metadata"]["fluent_pose_length"]
                current_original_length = int(original_lengths[i])
                fps = getattr(self.val_pose_header, "fps", 25.0)
                # Truncate to original length before reshaping
                ref_truncated_data = gt_unnorm[i, :current_original_length, :, :]
                ref_np = ref_truncated_data.cpu().numpy().reshape(current_original_length, 1, K,
                                                                  D_feat).astype(np.float64)
                ref_body = NumPyPoseBody(fps=fps, data=ref_np, confidence=np.ones((current_original_length, 1, K)))
                # Use full prediction length without truncation
                pred_length = pred_unnorm.shape[1]
                pred_full = pred_unnorm[i, :pred_length, :, :]  # shape (pred_length, K, D_feat)
                pred_np = pred_full.cpu().numpy().reshape(pred_length, 1, K, D_feat).astype(np.float64)
                preds_body = NumPyPoseBody(fps=fps, data=pred_np, confidence=np.ones((pred_length, 1, K)))
                refs.append(Pose(self.val_pose_header, ref_body))
                preds.append(Pose(self.val_pose_header, preds_body))
            return refs, preds

    def _compute_dtw_score(self, predictions: List[Pose], references: List[Pose]) -> float:
        """
        Compute mean DTW distance between predictions and references using corpus_score, with timing.
        """
        # Time the corpus_score computation
        start_time = time.time()
        # Wrap the entire references list as a single reference corpus
        wrapped_refs = [references]
        mean_score = float(self.validation_metric_calculator.corpus_score(predictions, wrapped_refs))
        elapsed = time.time() - start_time

        # Log timing separately for readability
        if self.logger:
            elapsed_msg = f"Validation DTW corpus_score time: {elapsed:.4f}s"
            self.logger.info(elapsed_msg)

        # Log the actual score
        if self.logger:
            score_msg = f"=== Validation DTW (corpus_score): {mean_score:.4f} ==="
            self.logger.info(score_msg)

        if self.tb_writer:
            self.tb_writer.add_scalar("validation/DTW_distance", mean_score, self.epoch)
        return mean_score

    # Override the run_loop method to include validation
    def run_loop(self, enable_profiler=False, profiler_directory="./logs/tb_profiler"):
        use_amp = getattr(self.config.trainer, "use_amp", False)
        # Initialize gradient scaler for mixed precision
        if use_amp:
            scaler = GradScaler("cuda")
        else:
            scaler = None
        if enable_profiler:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_directory),
            )
            profiler.start()
        else:
            profiler = None

        sampling_num = 16
        sampling_idx = np.random.randint(0, len(self.dataloader.dataset), sampling_num)
        sampling_subset = DataLoader(Subset(self.dataloader.dataset, sampling_idx), batch_size=sampling_num)
        self.evaluate_sampling(sampling_subset, save_folder_name="init_samples")
        # Sample fixed validation indices for saving predictions across epochs
        num_to_save = getattr(self.config.trainer, "validation_save_num", 30)
        val_dataset = self.validation_dataloader.dataset
        if len(val_dataset) > num_to_save:
            self.validation_sample_indices = np.random.choice(len(val_dataset), num_to_save, replace=False).tolist()
        else:
            self.validation_sample_indices = list(range(len(val_dataset)))

        epoch_process_bar = tqdm(range(self.epoch, self.num_epochs), desc=f"Epoch {self.epoch}")
        for epoch_idx in epoch_process_bar:
            self.model.train()
            self.model.training = True
            self.epoch = epoch_idx
            epoch_losses = {}

            data_len = len(self.dataloader)

            for datas in self.dataloader:
                datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
                cond = {
                    key: val.to(self.device) if torch.is_tensor(val) else val
                    for key, val in datas["conditions"].items()
                }
                x_start = datas["data"]

                self.opt.zero_grad()
                t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)

                if use_amp:
                    with autocast("cuda"):
                        _, losses = self.diffuse(x_start, t, cond, noise=None, return_loss=True)
                        total_loss = (losses["loss"] * weights).mean()
                    scaler.scale(total_loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    _, losses = self.diffuse(x_start, t, cond, noise=None, return_loss=True)
                    total_loss = (losses["loss"] * weights).mean()
                    total_loss.backward()
                    self.opt.step()

                if profiler:
                    profiler.step()

                if self.config.trainer.ema:
                    self.ema.update()

                for key_name in losses.keys():
                    if "loss" in key_name:
                        if key_name not in epoch_losses.keys():
                            epoch_losses[key_name] = []
                        epoch_losses[key_name].append(losses[key_name].mean().item())

            # Stop profiling after one epoch
            if profiler:
                profiler.stop()
                profiler = None

            if self.prior_loader is not None:
                for prior_datas in itertools.islice(self.prior_loader, data_len):
                    prior_datas = {
                        key: val.to(self.device) if torch.is_tensor(val) else val
                        for key, val in prior_datas.items()
                    }
                    prior_cond = {
                        key: val.to(self.device) if torch.is_tensor(val) else val
                        for key, val in prior_datas["conditions"].items()
                    }
                    prior_x_start = prior_datas["data"]

                    self.opt.zero_grad()
                    t, weights = self.schedule_sampler.sample(prior_x_start.shape[0], self.device)

                    if use_amp:
                        with autocast("cuda"):
                            _, prior_losses = self.diffuse(prior_x_start, t, prior_cond, noise=None, return_loss=True)
                            total_loss = (prior_losses["loss"] * weights).mean()
                        scaler.scale(total_loss).backward()
                        scaler.step(self.opt)
                        scaler.update()
                    else:
                        _, prior_losses = self.diffuse(prior_x_start, t, prior_cond, noise=None, return_loss=True)
                        total_loss = (prior_losses["loss"] * weights).mean()
                        total_loss.backward()
                        self.opt.step()

                    for key_name in prior_losses.keys():
                        if "loss" in key_name:
                            if key_name not in epoch_losses.keys():
                                epoch_losses[key_name] = []
                            epoch_losses[key_name].append(prior_losses[key_name].mean().item())

            loss_str = (f"loss_data: {np.mean(epoch_losses['loss_data']):.6f}, "
                        f"loss_data_vel: {np.mean(epoch_losses['loss_data_vel']):.6f}, "
                        f"loss: {np.mean(epoch_losses['loss']):.6f}")

            epoch_avg_loss = np.mean(epoch_losses["loss"])

            if self.epoch > 10 and epoch_avg_loss < self.best_loss:
                self.save_checkpoint(filename="best")

            if epoch_avg_loss < self.best_loss:
                self.best_loss = epoch_avg_loss

            epoch_process_bar.set_description(
                f"Epoch {epoch_idx}/{self.config.trainer.epoch} | loss: {epoch_avg_loss:.6f} | best_loss: {self.best_loss:.6f}"
            )
            self.logger.info(
                f"Epoch {epoch_idx}/{self.config.trainer.epoch} | {loss_str} | best_loss: {self.best_loss:.6f}")

            save_freq = max(1, int(getattr(self.config.trainer, "save_freq", 1)))
            if epoch_idx > 0 and epoch_idx % save_freq == 0:
                self.save_checkpoint(filename=f"weights_{epoch_idx}")
                self.evaluate_sampling(sampling_subset, save_folder_name="train_samples")

            for key_name in epoch_losses.keys():
                if "loss" in key_name:
                    self.tb_writer.add_scalar(f"train/{key_name}", np.mean(epoch_losses[key_name]), epoch_idx)

            self.scheduler.step()

            # Validation Phase
            eval_freq = getattr(self.config.trainer, "eval_freq", 1)
            if self.validation_dataloader is not None and (self.epoch % eval_freq == 0
                                                           or self.epoch == self.config.trainer.epoch - 1):
                current_validation_metric = self._run_validation_epoch()
                # Log the validation metric to TensorBoard
                if self.tb_writer and current_validation_metric is not None:
                    self.tb_writer.add_scalar("validation/DTW_distance", current_validation_metric, self.epoch)
                # If the metric is better, save the best model
                if (current_validation_metric is not None and current_validation_metric < self.best_validation_metric):
                    self.best_validation_metric = current_validation_metric
                    self.logger.info(
                        f"*** New best validation metric: {self.best_validation_metric:.4f} at epoch {self.epoch}. Saving best model. ***"
                    )
                    self.save_checkpoint(filename="best_model_validation")

                # Compute validation loss on chunks
                val_losses = []
                for val_batch in self.validation_dataloader:
                    # Recursively move to device and handle dicts
                    val_batch_device = {}
                    for key, v_item in val_batch.items():
                        if torch.is_tensor(v_item):
                            val_batch_device[key] = v_item.to(self.device)
                        elif isinstance(v_item, dict):
                            val_batch_device[key] = {
                                sk: sv.to(self.device) if torch.is_tensor(sv) else sv
                                for sk, sv in v_item.items()
                            }
                        else:
                            val_batch_device[key] = v_item

                    current_cond_for_diffuse = {k: v for k, v in val_batch_device.get("conditions", {}).items()}
                    if "metadata" in val_batch_device and "fluent_pose_length" in val_batch_device["metadata"]:
                        if "metadata" not in current_cond_for_diffuse:
                            current_cond_for_diffuse["metadata"] = {}
                        current_cond_for_diffuse["metadata"]["fluent_pose_length"] = val_batch_device["metadata"][
                            "fluent_pose_length"]

                    x_start = val_batch_device["data"]
                    t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                    with torch.no_grad():
                        _, losses = self.diffuse(x_start, t, current_cond_for_diffuse, noise=None, return_loss=True)
                    batch_loss = (losses["loss"] * weights).mean().item()
                    val_losses.append(batch_loss)
                if self.tb_writer and val_losses:
                    avg_val_loss = np.mean(val_losses)
                    self.tb_writer.add_scalar("validation/loss", avg_val_loss, self.epoch)
                    # Save the fixed 30 (or as defined in validation_save_num) validation predictions and corresponding GT results
                    save_dir = Path(self.config.save) / "validation_samples" / f"epoch_{self.epoch}"
                    mkdir(save_dir)
                    sample_indices = self.validation_sample_indices
                    val_save_loader = DataLoader(
                        Subset(val_dataset, sample_indices),
                        batch_size=1,
                        shuffle=False,
                        num_workers=self.config.trainer.workers,
                        pin_memory=True,
                        collate_fn=zero_pad_collator,
                    )
                    for loader_idx, batch_data in enumerate(val_save_loader):
                        refs, preds = self._process_validation_batch(batch_data, batch_idx=0)
                        idx = sample_indices[loader_idx]
                        ref = refs[0]
                        pred = preds[0]
                        ref_path = save_dir / f"ref_epoch{self.epoch}_idx{idx}.pose"
                        with open(ref_path, "wb") as f:
                            ref.write(f)
                        pred_path = save_dir / f"pred_epoch{self.epoch}_idx{idx}.pose"
                        with open(pred_path, "wb") as f:
                            pred.write(f)
                    self.logger.info(f"Saved {len(sample_indices)} validation GT and predictions to {save_dir}")

        best_path = "%s/best.pt" % (self.config.save)
        self.load_checkpoint(best_path)
        self.evaluate_sampling(sampling_subset, save_folder_name="best")

    def evaluate_sampling(self, dataloader: DataLoader, save_folder_name: str = "init_samples"):
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

        cond = {key: (val.to(self.device) if torch.is_tensor(val) else val) for key, val in datas["conditions"].items()}

        time, _ = self.schedule_sampler.sample(patched_dataloader.batch_size, self.device)
        with torch.no_grad():
            pred_output_tensor, _ = self.diffuse(fluent_clip, time, cond, noise=None, return_loss=False)
        fluent_clip_array = fluent_clip.cpu().numpy()
        pred_output_array = pred_output_tensor.cpu().numpy()

        unnormed_fluent_clip = (fluent_clip_array * dataset.input_std + dataset.input_mean)
        unnormed_pred_output = (pred_output_array * dataset.input_std + dataset.input_mean)

        self.export_samples(unnormed_fluent_clip, f"{self.save_dir}/{save_folder_name}", "gt")
        self.export_samples(unnormed_pred_output, f"{self.save_dir}/{save_folder_name}", "pred")

        # Save the normalized fluent clip and predicted output as numpy arrays
        np.save(
            f"{self.save_dir}/{save_folder_name}/gt_output_normed.npy",
            fluent_clip_array,
        )
        np.save(
            f"{self.save_dir}/{save_folder_name}/pred_output_normed.npy",
            pred_output_array,
        )
        # Save the unnormalized fluent clip and predicted output as numpy arrays
        unormed_gt_batch = np.stack(unnormed_fluent_clip, axis=0)  # (B, T, K, D)
        unormed_pred_batch = np.stack(unnormed_pred_output, axis=0)  # (B, T, K, D)
        np.save(f"{self.save_dir}/{save_folder_name}/gt_output.npy", unormed_gt_batch)
        np.save(f"{self.save_dir}/{save_folder_name}/pred_output.npy", unormed_pred_batch)

        if self.logger:
            self.logger.info(f"Evaluate sampling {save_folder_name} at epoch {self.epoch}")
        else:
            print(f"Evaluate sampling {save_folder_name} at epoch {self.epoch}")

    def export_samples(self, pose_output_np: np.ndarray, save_path: str, prefix: str) -> list:
        """
        Unnormalizes pose data using unnormalize_mean_std, exports to .pose format, and returns the unnormalized numpy data.
        Args:
            pose_output_normalized_np: A numpy array of normalized pose data.
            save_path: Path (string) where files will be saved.
            prefix: Prefix for file names, e.g., "gt" or "pred".
        """
        for i in range(pose_output_np.shape[0]):

            pose_array = pose_output_np[i]  # (time, keypoints, 3)
            time, keypoints, dim = pose_array.shape
            pose_array = pose_array.reshape(time, 1, keypoints, dim)

            # Set confidence to all 1.0 for all keypoints
            confidence = np.ones((time, 1, keypoints), dtype=np.float32)

            # Wrap and export the pose data
            pose_body = NumPyPoseBody(fps=25, data=pose_array, confidence=confidence)
            pose_obj = Pose(self.pose_header, pose_body)

            file_path = f"{save_path}/pose_{i}.{prefix}.pose"
            with open(file_path, "wb") as f:
                pose_obj.write(f)

            # Verify the file by reading it back
            with open(file_path, "rb") as f_check:
                Pose.read(f_check.read())
