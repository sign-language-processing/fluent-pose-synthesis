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
        fluent_clip: Tensor,
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
        if return_loss:
            x_start = fluent_clip.squeeze(2)  # (batch_size, time, keypoints, 3)
            if noise is None:
                noise = torch.randn_like(x_start)
            x_t = self.diffusion.q_sample(x_start, t, noise=noise)

            x_start_perm = x_start.permute(0, 2, 3, 1)
            x_t_perm = x_t.permute(0, 2, 3, 1)

            x_t_for_model = x_t.unsqueeze(2)
            model_output = self.model.interface(
                x_t_for_model, self.diffusion._scale_timesteps(t), cond
            )  # pylint: disable=protected-access
            model_output_perm = model_output.permute(0, 2, 3, 1)

            loss_terms = {}

            if self.diffusion.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                batch_size, keypoints = x_t_perm.shape[:2]
                assert model_output_perm.shape == (
                    batch_size,
                    keypoints * 2,
                    *x_t_perm.shape[2:],
                ), "Model output shape mismatch"
                model_output_perm, model_var_values = torch.split(
                    model_output_perm, keypoints, dim=1
                )
                frozen_out = torch.cat(
                    [model_output_perm.detach(), model_var_values], dim=1
                )
                loss_terms["vb"] = self.diffusion._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start_perm,
                    x_t=x_t_perm,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.diffusion.loss_type == LossType.RESCALED_MSE:
                    loss_terms["vb"] *= self.diffusion.num_timesteps / 1000.0
                print("vb loss:", loss_terms["vb"].mean().item())

            mmt = self.diffusion.model_mean_type  # real Enum instance from diffusion

            if mmt.name == "PREVIOUS_X":
                target = self.diffusion.q_posterior_mean_variance(
                    x_start=x_start_perm, x_t=x_t_perm, t=t
                )[0]
            elif mmt.name == "START_X":
                target = x_start_perm
            elif mmt.name == "EPSILON":
                target = noise
            else:
                raise ValueError(f"Unsupported model_mean_type: {mmt}")

            assert (
                model_output_perm.shape == target.shape == x_start_perm.shape
            ), "Target shape mismatch"

            frame_mask = cond["target_mask"]
            batch_size, keypoints, dimensions, time_steps = model_output_perm.shape
            mask = frame_mask[:, None, None, :].expand(
                batch_size, keypoints, dimensions, time_steps
            )

            if self.config.trainer.use_loss_mse:
                loss_terms["loss_data"] = self.diffusion.masked_l2(
                    target, model_output_perm, mask
                )

            if self.config.trainer.use_loss_vel:
                target_vel = target[..., 1:] - target[..., :-1]
                model_output_vel = (
                    model_output_perm[..., 1:] - model_output_perm[..., :-1]
                )
                mask_vel = mask[..., 1:]
                loss_terms["loss_data_vel"] = self.diffusion.masked_l2(
                    target_vel, model_output_vel, mask_vel
                )

            if self.config.trainer.use_loss_3d:
                target_rot = target.permute(0, 3, 1, 2)
                pred_rot = model_output_perm.permute(0, 3, 1, 2)
                frame_mask = cond["target_mask"]
                mask_rot = frame_mask[:, :, None, None].expand(
                    -1, -1, self.config.arch.keypoints, self.config.arch.dims
                )

                loss_terms["loss_geo_xyz"] = self.diffusion.masked_l2(
                    target_rot, pred_rot, mask_rot
                )

                if self.config.trainer.use_loss_vel:
                    target_xyz_vel = target_rot[:, 1:] - target_rot[:, :-1]
                    pred_xyz_vel = pred_rot[:, 1:] - pred_rot[:, :-1]
                    mask_xyz_vel = mask_rot[:, 1:]
                    loss_terms["loss_geo_xyz_vel"] = self.diffusion.masked_l2(
                        target_xyz_vel, pred_xyz_vel, mask_xyz_vel
                    )

            total_loss = (
                loss_terms.get("vb", 0.0)
                + loss_terms.get("loss_data", 0.0)
                + loss_terms.get("loss_data_vel", 0.0)
                + loss_terms.get("loss_geo_xyz", 0.0)
                + loss_terms.get("loss_geo_xyz_vel", 0.0)
            )
            loss_terms["loss"] = total_loss
            print("Total loss:", total_loss.mean().item())

            model_output_final = model_output_perm.permute(0, 3, 1, 2)
            return model_output_final, loss_terms

        with torch.no_grad():
            model_output = self.model.interface(fluent_clip, t, cond)
        return model_output.unsqueeze(2)

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

        self.export_samples(fluent_clip, f"{self.save_dir}/{save_folder_name}", "gt")
        self.export_samples(pred_output, f"{self.save_dir}/{save_folder_name}", "pred")

        self.logger.info(f"Evaluate sampling {save_folder_name} at epoch {self.epoch}")

    def export_samples(self, pose_output: Tensor, save_path: str, prefix: str):
        """
        Export pose sequences to disk in .pose format, with velocity-based confidence scores.
        Args:
            pose_output: A tensor of shape (batch_size, time, 1, keypoints, 3) or (batch_size, time, keypoints, 3).
            save_path: Path where files will be saved.
            prefix: Prefix for file names, e.g., "gt" or "pred".
        """
        for i in range(pose_output.shape[0]):
            pose_array = pose_output[i].cpu().numpy()  # (time, 1, keypoints, 3)
            time, people, keypoints, dim = pose_array.shape

            # Set confidence to all 1.0 for all keypoints
            confidence = np.ones((time, people, keypoints), dtype=np.float32)

            # Wrap and export the pose data
            pose_body = NumPyPoseBody(fps=25, data=pose_array, confidence=confidence)
            pose_obj = Pose(self.pose_header, pose_body)

            # Normalize for visualization
            normalize_pose_size(pose_obj)

            file_path = f"{save_path}/pose_{i}.{prefix}.pose"
            with open(file_path, "wb") as f:
                pose_obj.write(f)
            self.logger.info(f"Saved pose file: {file_path}")

            # Verify the file by reading it back
            with open(file_path, "rb") as f_check:
                Pose.read(
                    f_check.read()
                )  # If error occurs, the file was not written correctly
            self.logger.info(f"Pose file {file_path} read successfully.")
