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

from CAMDM.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelVarType,
)
from CAMDM.network.training import BaseTrainingPortal
from CAMDM.utils.common import mkdir


def masked_l2_per_sample(x, y, mask=None, reduce=True):
    """
    Compute masked L2 loss per sample.
    Args:
        x, y: Tensors of shape (batch_size, joints, dims, time)
        mask: Tensor of shape (batch_size, joints, dims, time). True = masked (invalid), False = valid.
        reduce: Whether to average over batch dimension.
    """
    diff = (x - y) ** 2  # (B, K, D, T)

    if mask is not None:
        mask = mask.bool()  # Ensure bool
        valid_mask = (~mask).float()  # False=valid -> 1.0, True=masked -> 0.0
        diff = diff * valid_mask
    else:
        valid_mask = torch.ones_like(diff)

    per_sample_loss = diff.flatten(start_dim=1).sum(dim=1)
    valid_elements = valid_mask.flatten(start_dim=1).sum(dim=1)

    per_sample_loss = per_sample_loss / valid_elements.clamp(min=1)

    if reduce:
        return per_sample_loss.mean()
    else:
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

    def diffuse(
        self,
        x_start: Tensor,
        t: Tensor,
        cond: Dict[str, Tensor],
        noise: Optional[Tensor] = None,
        return_loss: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Perform diffusion on the input fluent_clip tensor, and return the model output.
        Args:
            fluent_clip: Input target pose clip. Shape: (batch_size, time, 1, keypoints, 3)
            t: Diffusion timesteps for each sample. Shape: (batch_size,)
            cond: Conditioning inputs, must include "target_mask" and "input_sequence".
            noise: Optional noise to apply during diffusion.
            return_loss: Whether to compute and return training losses.
        """
        x_start = x_start.permute(0, 2, 3, 1) # (batch_size, keypoints, dimensions, time)

        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.diffusion.q_sample(x_start, t, noise=noise)

        cond["input_sequence"] = cond["input_sequence"].permute(0, 2, 3, 1) # (batch_size, keypoints, dimensions, time)

        model_output = self.model.interface(x_t, self.diffusion._scale_timesteps(t), cond)

        if return_loss:

            loss_terms = {}

            mmt = self.diffusion.model_mean_type  # real Enum instance from diffusion
            if mmt.name == "PREVIOUS_X":
                target = self.diffusion.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0]
            elif mmt.name == "START_X":
                target = x_start
            elif mmt.name == "EPSILON":
                target = noise
            else:
                raise ValueError(f"Unsupported model_mean_type: {mmt}")

            assert (model_output.shape == target.shape == x_start.shape), "Target shape mismatch"

            target_mask = cond["target_mask"].permute(0, 3, 4, 1, 2).squeeze(4)  # (batch_size, keypoints, dimensions, time)
            batch_size, keypoints, dimensions, time_steps = model_output.shape
            mask = target_mask

            # print("[DEBUG] mask shape:", mask.shape)
            # print("[DEBUG] mask valid count:", (mask == False).sum().item(), "/", mask.numel())
            # print("[DEBUG] any sample mask all zero:", (mask.sum(dim=(1,2,3)) == 0).any().item())
            # print("[DEBUG] mask mean (coverage):", mask.float().mean().item())

            valid_mask = (mask.sum(dim=(1,2,3)) > 0).float()  # (batch_size,)

            if valid_mask.sum() == 0:
                print("[WARNING] All samples are invalid, skipping loss computation.")
                return None, {}

            if self.config.trainer.use_loss_mse:
                per_sample_loss = masked_l2_per_sample(target, model_output, mask, reduce=False)  # (batch_size,)
                loss_terms["loss_data"] = (per_sample_loss * valid_mask).sum() / valid_mask.sum()

            if self.config.trainer.use_loss_vel:
                target_vel = target[..., 1:] - target[..., :-1]
                model_output_vel = model_output[..., 1:] - model_output[..., :-1]
                mask_vel = mask[..., 1:]

                per_sample_loss_vel = masked_l2_per_sample(target_vel, model_output_vel, mask_vel, reduce=False)
                loss_terms["loss_data_vel"] = (per_sample_loss_vel * valid_mask).sum() / valid_mask.sum()

            total_loss = (
                loss_terms.get("vb", 0.0)
                + loss_terms.get("loss_data", 0.0)
                + loss_terms.get("loss_data_vel", 0.0)
            )
            loss_terms["loss"] = total_loss
            print("Total loss:", total_loss.mean().item())

            return model_output.permute(0, 3, 1, 2), loss_terms

        return model_output.permute(0, 3, 1, 2)


    def evaluate_sampling(
        self, dataloader: DataLoader, save_folder_name: str = "init_samples"
    ):
        """
        Perform inference and save generated samples from the model.
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
            pred_output = self.diffuse(
                fluent_clip, time, cond, noise=None, return_loss=False
            )

        fluent_clip_array, pred_output_array = fluent_clip.cpu().numpy(), pred_output.cpu().numpy()
        unormed_fluent_clip = fluent_clip_array * dataset.input_std + dataset.input_mean
        unormed_pred_output = pred_output_array * dataset.input_std + dataset.input_mean

        self.export_samples(unormed_fluent_clip, f"{self.save_dir}/{save_folder_name}", "gt")
        self.export_samples(unormed_pred_output, f"{self.save_dir}/{save_folder_name}", "pred")

        np.save(f"{self.save_dir}/{save_folder_name}/gt_output_normed.npy", fluent_clip_array)
        np.save(f"{self.save_dir}/{save_folder_name}/pred_output_normed.npy", pred_output_array)
        np.save(f"{self.save_dir}/{save_folder_name}/gt_output.npy", unormed_fluent_clip)
        np.save(f"{self.save_dir}/{save_folder_name}/pred_output.npy", unormed_pred_output)

        if self.logger:
            self.logger.info(f"Evaluate sampling {save_folder_name} at epoch {self.epoch}")
        else:
            print(f"Evaluate sampling {save_folder_name} at epoch {self.epoch}")


    def export_samples(self, pose_output: Tensor, save_path: str, prefix: str):
        """
        Export pose sequences to disk in .pose format.
        Args:
            pose_output: A tensor of shape (batch_size, time, 1, keypoints, 3) or (batch_size, time, keypoints, 3).
            save_path: Path where files will be saved.
            prefix: Prefix for file names, e.g., "gt" or "pred".
        """
        for i in range(pose_output.shape[0]):

            pose_array = pose_output[i] # (time, 1, keypoints, 3)
            time, keypoints, dim = pose_array.shape
            pose_array = pose_array.reshape(time, 1, keypoints, dim)

            # Set confidence to all 1.0 for all keypoints
            confidence = np.ones((time, 1, keypoints), dtype=np.float32)

            # Wrap and export the pose data
            pose_body = NumPyPoseBody(fps=25, data=pose_array, confidence=confidence)
            pose_obj = Pose(self.pose_header, pose_body)

            # Normalize for visualization
            # normalize_pose_size(pose_obj)

            file_path = f"{save_path}/pose_{i}.{prefix}.pose"
            with open(file_path, "wb") as f:
                pose_obj.write(f)
            # self.logger.info(f"Saved pose file: {file_path}")

            # Verify the file by reading it back
            with open(file_path, "rb") as f_check:
                Pose.read(f_check.read())

            # If error occurs, the file was not written correctly
            # self.logger.info(f"Pose file {file_path} read successfully.")
