import os
import sys
import importlib
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMDM_PATH = os.path.join(BASE_DIR, "CAMDM", "PyTorch")
# Add CAMDM directory to Python path
sys.path.append(CAMDM_PATH)

from diffusion.gaussian_diffusion import *
from network.training import BaseTrainingPortal
import utils.common as common

from pose_format import Pose, PoseHeader
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.utils.reader import BufferReader
from pose_format.numpy.pose_body import NumPyPoseBody


class PoseTrainingPortal(BaseTrainingPortal):
    def __init__(
        self,
        config: Any,
        model: torch.nn.Module,
        diffusion: GaussianDiffusion,
        dataloader: DataLoader,
        logger: Optional[Any],
        tb_writer: Optional[Any],
        finetune_loader: Optional[DataLoader] = None
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
        super().__init__(config, model, diffusion, dataloader, logger, tb_writer, finetune_loader)

        dataset_module = importlib.import_module("sign_language_datasets.datasets.dgs_corpus.dgs_corpus")
        with open(dataset_module._POSE_HEADERS["holistic"], "rb") as buffer:
            self.pose_header = PoseHeader.read(BufferReader(buffer.read()))

    def diffuse(self, fluent_clip: Tensor, t: Tensor, cond: Dict[str, Tensor], 
                noise: Optional[Tensor] = None, return_loss: bool = False) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Perform diffusion on the input fluent_clip tensor, and return the model output.
        Args:
            fluent_clip: Input target pose clip. Shape: (B, T, 1, K, 3)
            t: Diffusion timesteps for each sample. Shape: (B,)
            cond: Conditioning inputs, must include "target_mask" and "input_sequence".
            noise: Optional noise to apply during diffusion.
            return_loss: Whether to compute and return training losses.
        """
        if return_loss:
            # Squeeze dimension 2 (people)
            x_start = fluent_clip.squeeze(2)  # (B, T, keypoints, 3)
            if noise is None:
                noise = torch.randn_like(x_start)
            x_t = self.diffusion.q_sample(x_start, t, noise=noise)

            # # Print debug info about the input tensors
            # print("=== Debug: Fluent Clip Info ===")
            # print("x_start shape:", x_start.shape,
            #       "mean:", x_start.mean().item(), "std:", x_start.std().item())
            # print("x_t shape:", x_t.shape,
            #       "mean:", x_t.mean().item(), "std:", x_t.std().item())

            x_start_perm = x_start.permute(0, 2, 3, 1)  # (B, K, 3, T)
            x_t_perm = x_t.permute(0, 2, 3, 1)

            x_t_for_model = x_t.unsqueeze(2)  # (B, T, 1, K, 3)
            model_output = self.model.interface(x_t_for_model, self.diffusion._scale_timesteps(t), cond)  # (B, T, K, 3)
            model_output_perm = model_output.permute(0, 2, 3, 1)  # (B, K, 3, T)

            loss_terms = {}

            if self.diffusion.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, K = x_t_perm.shape[:2]
                assert model_output_perm.shape == (B, K * 2, *x_t_perm.shape[2:]), "Model output shape mismatch"
                model_output_perm, model_var_values = torch.split(model_output_perm, K, dim=1)
                frozen_out = torch.cat([model_output_perm.detach(), model_var_values], dim=1)
                loss_terms["vb"] = self.diffusion._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start_perm,
                    x_t=x_t_perm,
                    t=t,
                    clip_denoised=False
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    loss_terms["vb"] *= self.diffusion.num_timesteps / 1000.0
                print("vb loss:", loss_terms["vb"].mean().item())

            target = {
                ModelMeanType.PREVIOUS_X: self.diffusion.q_posterior_mean_variance(x_start=x_start_perm, 
                                                                                   x_t=x_t_perm, t=t)[0],
                ModelMeanType.START_X: x_start_perm,
                ModelMeanType.EPSILON: noise,
            }[self.diffusion.model_mean_type]

            # Ensure shapes match
            assert model_output_perm.shape == target.shape == x_start_perm.shape, "Target shape mismatch"

            # Convert mask: cond['target_mask'] (B, T, 1, keypoints, 3) -> (B, K, 3, T)
            mask = cond['target_mask'].squeeze(2).permute(0, 2, 3, 1)

            if self.config.trainer.use_loss_mse:
                loss_terms['loss_data'] = self.diffusion.masked_l2(target, model_output_perm, mask)

            if self.config.trainer.use_loss_vel:
                target_vel = target[..., 1:] - target[..., :-1]  # (B, K, 3, T-1)
                model_output_vel = model_output_perm[..., 1:] - model_output_perm[..., :-1]
                mask_vel = mask[..., 1:]  # (B, K, 3, T-1)

                loss_terms['loss_data_vel'] = self.diffusion.masked_l2(target_vel, model_output_vel, mask_vel)

            if self.config.trainer.use_loss_3d:
                target_rot = target.permute(0, 3, 1, 2)  # (B, T, K, 3)
                pred_rot = model_output_perm.permute(0, 3, 1, 2)  # (B, T, K, 3)
                mask_rot = cond['target_mask'].squeeze(2)  # (B, T, K, 3)

                loss_terms["loss_geo_xyz"] = self.diffusion.masked_l2(target_rot, pred_rot, mask_rot)

                if self.config.trainer.use_loss_vel:
                    target_xyz_vel = target_rot[:, 1:] - target_rot[:, :-1]
                    pred_xyz_vel = pred_rot[:, 1:] - pred_rot[:, :-1]
                    mask_xyz_vel = mask_rot[:, 1:]
                    loss_terms["loss_geo_xyz_vel"] = self.diffusion.masked_l2(target_xyz_vel, pred_xyz_vel, mask_xyz_vel)

            total_loss = (
                loss_terms.get('vb', 0.) +
                loss_terms.get('loss_data', 0.) +
                loss_terms.get('loss_data_vel', 0.) +
                loss_terms.get('loss_geo_xyz', 0.) +
                loss_terms.get('loss_geo_xyz_vel', 0.)
            )
            loss_terms["loss"] = total_loss
            print("Total loss:", total_loss.mean().item())

            model_output_final = model_output_perm.permute(0, 3, 1, 2)  # (B, T, K, 3)
            return model_output_final, loss_terms

        else:
            with torch.no_grad():
                model_output = self.model.interface(fluent_clip, t, cond)

            return model_output.unsqueeze(2)  # (B, T, 1, K, 3)

    def evaluate_sampling(self, dataloader: DataLoader, save_folder_name: str = 'init_samples'):
        """
        Perform inference and save generated samples from the model.
        Args:
            dataloader (Dataloader): A DataLoader instance containing the evaluation dataset.
            save_folder_name (str): Folder to save generated pose files.
        """
        self.model.eval()
        common.mkdir(f'{self.save_dir}/{save_folder_name}')

        patched_dataloader = DataLoader(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=self.config.trainer.workers,
            collate_fn=zero_pad_collator,
            pin_memory=True
        )

        datas = next(iter(patched_dataloader))
        fluent_clip = datas['data'].to(self.device)  # fluent_clip shape: (B, T, 1, keypoints, 3)
        cond = {key: (val.to(self.device) if torch.is_tensor(val) else val)
                for key, val in datas['conditions'].items()}
        t, _ = self.schedule_sampler.sample(patched_dataloader.batch_size, self.device)
        with torch.no_grad():
            model_output = self.diffuse(fluent_clip, t, cond, noise=None, return_loss=False)
        self.export_samples(fluent_clip, f'{self.save_dir}/{save_folder_name}', 'gt')
        self.export_samples(model_output, f'{self.save_dir}/{save_folder_name}', 'pred')
        self.logger.info(f'Evaluate sampling {save_folder_name} at epoch {self.epoch}')

    def export_samples(self, pose_output: Tensor, save_path: str, prefix: str):
        """
        Export pose sequences to disk in .pose format.
        Args:
            pose_output: A tensor of shape (B, T, 1, K, 3) or (B, T, K, 3).
            save_path: Path where files will be saved.
            prefix: Prefix for file names, e.g., "gt" or "pred".
        """
        for i in range(pose_output.shape[0]):
            pose_array = pose_output[i].cpu().numpy()  # (T, 1, keypoints, 3)
            T, P, K, D = pose_array.shape

            ###### To-Do: How to generate confidence values?
            confidence = np.ones((T, P, K), dtype=np.float32)

            pose_body = NumPyPoseBody(fps=25, data=pose_array, confidence=confidence)
            pose_obj = Pose(self.pose_header, pose_body)

            file_path = f'{save_path}/pose_{i}.{prefix}.pose'
            with open(file_path, 'wb') as f:
                pose_obj.write(f)
            self.logger.info(f'Saved pose file: {file_path}')