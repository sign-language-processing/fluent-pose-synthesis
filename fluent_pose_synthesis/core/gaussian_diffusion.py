from typing import Optional, Dict, Any
import torch as th

from CAMDM.diffusion.gaussian_diffusion import (GaussianDiffusion, get_named_beta_schedule, LossType, ModelMeanType,
                                                ModelVarType)


class PoseGaussianDiffusion(GaussianDiffusion):

    def __init__(self, schedule_kwargs: Dict[str, Any], **kwargs: Any):
        """
        Initialize the PoseGaussianDiffusion class using a beta schedule and other diffusion parameters.
        Args:
            schedule_kwargs (Dict[str, Any]): Parameters passed to `get_named_beta_schedule`.
            **kwargs: Additional arguments passed to the parent GaussianDiffusion class.
        """
        # Generate the beta schedule from the provided parameters
        betas = get_named_beta_schedule(**schedule_kwargs)

        # Initialize the parent GaussianDiffusion class with appropriate parameters
        super().__init__(
            betas=betas,
            model_mean_type=kwargs.get("model_mean_type", ModelMeanType.START_X),
            model_var_type=kwargs.get("model_var_type", ModelVarType.FIXED_SMALL),
            loss_type=kwargs.get("loss_type", LossType.MSE),
            rescale_timesteps=kwargs.get("rescale_timesteps", False),
            lambda_3d=kwargs.get("lambda_3d", 1.0),
            lambda_vel=kwargs.get("lambda_vel", 1.0),
            lambda_r_vel=kwargs.get("lambda_r_vel", 1.0),
            lambda_pose=kwargs.get("lambda_pose", 1.0),
            lambda_orient=kwargs.get("lambda_orient", 1.0),
            lambda_loc=kwargs.get("lambda_loc", 1.0),
            lambda_root_vel=kwargs.get("lambda_root_vel", 0.0),
            lambda_vel_rcxyz=kwargs.get("lambda_vel_rcxyz", 0.0),
            lambda_fc=kwargs.get("lambda_fc", 0.0),
            data_rep=kwargs.get("data_rep", "rot6d"),
        )

    def training_losses_pose(
        self,
        model: th.nn.Module,
        pose_target: th.Tensor,
        t: th.Tensor,
        config: Any,
        model_kwargs: Optional[Dict[str, Any]] = None,
        noise: Optional[th.Tensor] = None,
    ) -> Dict[str, th.Tensor]:
        """
        Compute training losses for pose sequences using 3D coordinates.
        Args:
            model (torch.nn.Module): The neural network used for denoising.
            pose_target (torch.Tensor): Ground truth pose sequence. Shape: (B, T, K, 3)
            t (torch.Tensor): Timestep tensor for diffusion. Shape: (B,)
            config (Any): Configuration object, should contain flags like `use_loss_mse`, `use_loss_3d`, etc.
            model_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for the model.
            noise (Optional[torch.Tensor]): Optional noise tensor. If None, generated internally.
        """
        # Diffuse the target pose to obtain x_t
        pose_noisy = self.q_sample(pose_target, t, noise=noise)
        if noise is None:
            noise = th.randn_like(pose_target)

        # Retrieve the mask from model_kwargs (if provided)
        mask = None
        if model_kwargs is not None and "mask" in model_kwargs:
            mask = model_kwargs["mask"]  # shape: (B, T, 1, keypoints, 3)
            mask = mask.squeeze(2)  # (B, T, keypoints, 3)
            if mask.size(-1) == 3:
                mask = mask.mean(dim=-1)  # now (B, T, keypoints)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # now (B, 1, T, keypoints)

        loss_terms = {}

        if self.loss_type in (LossType.KL, LossType.RESCALED_KL):
            loss_terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=pose_target,
                x_t=pose_noisy,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                loss_terms["loss"] *= self.num_timesteps

        elif self.loss_type in (LossType.MSE, LossType.RESCALED_MSE):
            # Compute model output; expected shape: (B, T, keypoints, 3)
            model_output = model(pose_noisy, self._scale_timesteps(t), model_kwargs)
            if self.model_var_type in [
                    ModelVarType.LEARNED,
                    ModelVarType.LEARNED_RANGE,
            ]:
                batch_size, time = pose_noisy.shape[:2]
                assert model_output.shape == (
                    batch_size,
                    time * 2,
                    *pose_noisy.shape[2:],
                ), f"Unexpected model output shape: {model_output.shape}"
                model_output, model_var_values = th.split(model_output, time, dim=1)
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                loss_terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=pose_target,
                    x_t=pose_noisy,
                    t=t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    loss_terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=pose_target, x_t=pose_noisy, t=t)[0],
                ModelMeanType.START_X: pose_target,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            # Ensure shapes match: (B, T, keypoints, 3)
            assert model_output.shape == target.shape == pose_target.shape, (
                f"Shape mismatch: model_output {model_output.shape}, target {target.shape}, "
                f"pose_target {pose_target.shape}")

            loss_terms["output"] = model_output

            if config.trainer.use_loss_mse:
                loss_terms["loss_mse"] = self.masked_l2(model_output.unsqueeze(1), pose_target.unsqueeze(1), mask)

            if config.trainer.use_loss_3d:
                # Direct 3D loss between predicted and target poses
                loss_terms["loss_3d"] = self.masked_l2(model_output, pose_target, mask)

            if config.trainer.use_loss_vel:
                # Compute velocity (frame differences) and calculate velocity loss
                pred_velocity = model_output[:, 1:] - model_output[:, :-1]
                target_velocity = pose_target[:, 1:] - pose_target[:, :-1]
                # Adjust mask for velocity if provided (assume time dimension is at index 1)
                if mask is not None:
                    mask_vel = mask[:, :, 1:]
                else:
                    mask_vel = None
                loss_terms["loss_vel"] = self.masked_l2(pred_velocity, target_velocity, mask_vel)

            # Aggregate total loss using lambda factors
            loss_terms["loss"] = (loss_terms["loss_mse"] + loss_terms.get("vb", 0.0) +
                                  (self.lambda_3d * loss_terms.get("loss_3d", 0.0)) +
                                  (self.lambda_vel * loss_terms.get("loss_vel", 0.0)))
        else:
            raise NotImplementedError(self.loss_type)

        return loss_terms
