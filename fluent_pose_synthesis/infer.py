# Example usage:
# python -m fluent_pose_synthesis.infer \
#   -c save/debug_run/two_losses_sample_dataset/config.json \
#   -i assets/sample_dataset \
#   -r save/debug_run/two_losses_sample_dataset/best.pt \
#   -o save/debug_run/two_losses_sample_dataset/infer_result \
#   --batch_size 16


import argparse
import json
from pathlib import Path
from pathlib import PosixPath
import torch
from torch.utils.data import DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
import numpy as np
from types import SimpleNamespace
import torch.serialization
from types import SimpleNamespace
from numpy.core.multiarray import scalar
from numpy import dtype

from CAMDM.diffusion.create_diffusion import create_gaussian_diffusion
from CAMDM.utils.common import fixseed
from fluent_pose_synthesis.core.models import SignLanguagePoseDiffusion
from fluent_pose_synthesis.data.load_data import SignLanguagePoseDataset
from fluent_pose_synthesis.core.training import PoseTrainingPortal
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody


torch.serialization.add_safe_globals([
        SimpleNamespace,
        Path,
        PosixPath,
        scalar,
        dtype,
        np.int64,
        np.int32,
        np.float64,
        np.float32,
        np.bool_,
    ])


def convert_namespace_to_dict(namespace):
    if isinstance(namespace, SimpleNamespace):
        return {k: convert_namespace_to_dict(v) for k, v in vars(namespace).items()}
    elif isinstance(namespace, dict):
        return {k: convert_namespace_to_dict(v) for k, v in namespace.items()}
    elif isinstance(namespace, PosixPath):
        return str(namespace)
    elif isinstance(namespace, torch.device):
        return str(namespace)
    else:
        return namespace


def load_checkpoint_and_config(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded model checkpoint from {checkpoint_path}")
    return model, checkpoint.get("config", None)


def main():
    parser = argparse.ArgumentParser(description="Fluent Pose Synthesis Inference")
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="Path to input disfluent dataset"
    )
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Path to config file"
    )
    parser.add_argument(
        "-r", "--resume", required=True, type=str, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "-o", "--output", default="output/infer_results", type=str, help="Path to save generated outputs"
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size for inference"
    )

    args = parser.parse_args()

    # === Load model ===

    # Helper function to turn dict into Namespace recursively
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        else:
            return d

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
    except json.JSONDecodeError:
        with open(args.config, "r", encoding="utf-8") as f:
            namespace_content = f.read()
        config_namespace = eval(namespace_content,{"namespace": SimpleNamespace, "PosixPath": PosixPath, "device": torch.device})
        config_dict = convert_namespace_to_dict(config_namespace)

        # Save the fixed JSON config
        standard_json_path = Path(args.config).with_suffix('.fixed.json')
        with open(standard_json_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        print(f"[INFO] Saved standard JSON config to {standard_json_path}")

    config = dict_to_namespace(config_dict)

    fixseed(42)

    # Load dataset
    dataset = SignLanguagePoseDataset(
        data_dir=Path(args.input),
        split="test",  # Or "valid", as needed
        fluent_frames=config.arch.clip_len,
        dtype=np.float32,
        limited_num=-1,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=zero_pad_collator,
        num_workers=0,
    )

    # Load model
    input_feats = config.arch.keypoints * config.arch.dims
    model = SignLanguagePoseDiffusion(
        input_feats=input_feats,
        clip_len=config.arch.clip_len,
        keypoints=config.arch.keypoints,
        dims=config.arch.dims,
        latent_dim=config.arch.latent_dim,
        ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers,
        num_heads=config.arch.num_heads,
        dropout=getattr(config.arch, "dropout", 0.2),
        ablation=getattr(config.arch, "ablation", None),
        activation=getattr(config.arch, "activation", "gelu"),
        legacy=getattr(config.arch, "legacy", False),
        arch=config.arch.decoder,
        cond_mask_prob=config.trainer.cond_mask_prob,
        device=torch.device("cpu"),
    ).to(torch.device("cpu"))


    model, config = load_checkpoint_and_config(model, args.resume)

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(config.device)

    diffusion = create_gaussian_diffusion(config)

    # Initialize a PoseTrainingPortal only for evaluation
    infer_portal = PoseTrainingPortal(
        config=config,
        model=model,
        diffusion=diffusion,
        dataloader=dataloader,
        logger=None,
        tb_writer=None,
    )
    infer_portal.save_dir = Path(args.output)
    infer_portal.epoch = 0  # Optional: Set epoch info if needed
    infer_portal.pose_header = dataset.pose_header

    model.eval()
    model.training = False

    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    infer_portal.evaluate_sampling(dataloader)

#     # Load one batch for inference
#     datas = next(iter(dataloader))
#     batch_data = datas

#     fluent_clip = batch_data["data"].to(config.device)

#     cond = {
#     key: (val.permute(0, 3, 2, 1).to(config.device) if key == "input_sequence" else (val.to(config.device) if torch.is_tensor(val) else val))
#     for key, val in batch_data["conditions"].items()
# }

#     shape = (fluent_clip.shape[0], config.arch.keypoints, config.arch.dims, config.arch.clip_len)


#     class WrappedSignLanguageDiffusion(torch.nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self.model = model

#         def forward(self, x, t, **kwargs):
#             return self.model.interface(x, t, **kwargs)

#     model = WrappedSignLanguageDiffusion(model)

#     with torch.no_grad():
#         samples = diffusion.p_sample_loop(
#             model=model,
#             shape=shape,
#             clip_denoised=False,
#             model_kwargs={"y": cond},
#             skip_timesteps=0,
#             init_image=None,
#             progress=True,
#             dump_steps=None,
#             noise=None,
#             const_noise=False,
#         )

#     # Save the generated samples
#     pred_output_array = samples.permute(0, 3, 1, 2).cpu().numpy()

#     # Unnormalize the output
#     unormed_pred_output = pred_output_array * dataset.input_std + dataset.input_mean

#     np.save(save_dir / "pred_output_normed.npy", pred_output_array)
#     np.save(save_dir / "pred_output.npy", unormed_pred_output)

#     # Export the generated samples to .pose files
#     for i in range(unormed_pred_output.shape[0]):
#         pose_array = unormed_pred_output[i]  # (time, keypoints, dims)
#         time_steps, keypoints, dims = pose_array.shape
#         pose_array = pose_array.reshape(time_steps, 1, keypoints, dims)  # (time, 1, keypoints, dims)
#         confidence = np.ones((time_steps, 1, keypoints), dtype=np.float32)

#         pose_body = NumPyPoseBody(fps=25, data=pose_array, confidence=confidence)
#         pose_obj = Pose(infer_portal.pose_header, pose_body)

#         output_file = save_dir / f"pose_{i}.pred.pose"
#         with open(output_file, "wb") as f:
#             pose_obj.write(f)

#     # Save the disfluent input
#     disfluent_input_array = cond["input_sequence"].cpu().numpy()
#     disfluent_input_array = disfluent_input_array.transpose(0, 3, 2, 1)  # (B, T, K, D)
#     disfluent_input_array = disfluent_input_array.transpose(0, 2, 3, 1)  # (B, K, D, T)

#     # Unnormalize the disfluent input
#     input_std = dataset.input_std.reshape(1, 178, 3, 1)
#     input_mean = dataset.input_mean.reshape(1, 178, 3, 1)

#     unormed_disfluent_input = disfluent_input_array * input_std + input_mean
#     unormed_disfluent_input = unormed_disfluent_input.transpose(0, 3, 1, 2)

#     np.save(save_dir / "disfluent_input_normed.npy", disfluent_input_array)
#     np.save(save_dir / "disfluent_input.npy", unormed_disfluent_input)

#     # Export the disfluent input to .pose files
#     for i in range(unormed_disfluent_input.shape[0]):
#         disfluent_array = unormed_disfluent_input[i]  # (time, keypoints, dims)
#         time_steps, keypoints, dims = disfluent_array.shape
#         disfluent_array = disfluent_array.reshape(time_steps, 1, keypoints, dims)  # (time, 1, keypoints, dims)
#         confidence = np.ones((time_steps, 1, keypoints), dtype=np.float32)

#         disfluent_body = NumPyPoseBody(fps=25, data=disfluent_array, confidence=confidence)
#         disfluent_pose_obj = Pose(infer_portal.pose_header, disfluent_body)

#         disfluent_output_file = save_dir / f"pose_{i}.input.pose"
#         with open(disfluent_output_file, "wb") as f:
#             disfluent_pose_obj.write(f)

#     print(f"Saved inference results to {save_dir}")

if __name__ == "__main__":
    main()
