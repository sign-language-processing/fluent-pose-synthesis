# Example usage:
# python -m  fluent_pose_synthesis.infer_autoregressive \
#   -i assets/sample_dataset \
#   -c save/debug_run/weighted_auto_step32_with_200data_output/config.json \
#   -r save/debug_run/weighted_auto_step32_with_200data_output/best.pt \
#   -o save/debug_run/weighted_auto_step32_with_200data_output/infer_output \
#   --batch_size 4 \
#   --chunk_size 30 \
#   --max_len 120 \
#   --stop_threshold 1e-4 \
#   --seed 1234


import argparse
import json
import time
from pathlib import Path, PosixPath
import torch
from torch.utils.data import DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
import numpy as np
from types import SimpleNamespace
from numpy.core.multiarray import scalar
from numpy import dtype
import torch.serialization

# CAMDM and project imports
from CAMDM.diffusion.create_diffusion import create_gaussian_diffusion
from CAMDM.utils.common import fixseed
from fluent_pose_synthesis.core.models import SignLanguagePoseDiffusion
from fluent_pose_synthesis.data.load_data import SignLanguagePoseDataset
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_anonymization.data.normalization import unnormalize_mean_std
from pose_format.utils.generic import normalize_pose_size


torch.serialization.add_safe_globals([
        SimpleNamespace, Path, PosixPath, scalar, dtype,
        np.int64, np.int32, np.float64, np.float32, np.bool_,
    ])


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def convert_namespace_to_dict(namespace_obj):
    if isinstance(namespace_obj, SimpleNamespace):
        return {k: convert_namespace_to_dict(v) for k, v in vars(namespace_obj).items()}
    elif isinstance(namespace_obj, dict):
        return {k: convert_namespace_to_dict(v) for k, v in namespace_obj.items()}
    elif isinstance(namespace_obj, (Path, PosixPath)):
        return str(namespace_obj)
    elif isinstance(namespace_obj, torch.device):
        return str(namespace_obj)
    else:
        return namespace_obj


def load_checkpoint_and_config(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded model checkpoint from {checkpoint_path}")
    loaded_config_dict = checkpoint.get("config", None)
    if loaded_config_dict:
        if isinstance(loaded_config_dict, SimpleNamespace):
            loaded_config_dict = convert_namespace_to_dict(loaded_config_dict)
        print("[INFO] Config found in checkpoint.")
        return model, loaded_config_dict
    return model, None


def main():
    parser = argparse.ArgumentParser(description="Autoregressive inference for fluent pose synthesis")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to disfluent input data directory")
    parser.add_argument("-c", "--config", required=True, type=str, help="Path to model config JSON file")
    parser.add_argument("-r", "--resume", required=True, type=str, help="Path to model checkpoint (.pt)")
    parser.add_argument("-o", "--output", default="output/infer_autoregressive_results", type=str, help="Directory to save generated outputs")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for inference.")
    parser.add_argument("--chunk_size", default=10, type=int, help="Number of frames to attempt to generate per autoregressive step.")
    parser.add_argument("--max_len", default=100, type=int, help="Maximum total frames to generate.")
    parser.add_argument("--stop_threshold", default=1e-5, type=float, help="Threshold for mean absolute value of generated chunk to detect stop condition.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    args = parser.parse_args()

    fixseed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Configuration
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        print(f"[INFO] Successfully loaded config from {args.config} as JSON.")
    except json.JSONDecodeError:
        print(f"[WARNING] Could not parse {args.config} as JSON. Attempting to eval as SimpleNamespace string...")
        with open(args.config, "r", encoding="utf-8") as f:
            namespace_content = f.read()
        config_namespace = eval(namespace_content, {"SimpleNamespace": SimpleNamespace, "namespace": SimpleNamespace, "PosixPath": PosixPath, "device": torch.device, "Path": Path})
        config_dict = convert_namespace_to_dict(config_namespace)
        standard_json_path = Path(args.config).with_suffix('.fixed.json')
        with open(standard_json_path, "w", encoding="utf-8") as f_json:
            json.dump(config_dict, f_json, indent=4)
        print(f"[INFO] Saved successfully parsed config to {standard_json_path}")

    config = dict_to_namespace(config_dict)
    config.inference = SimpleNamespace(
        chunk_size=args.chunk_size, max_len=args.max_len, stop_threshold=args.stop_threshold
    )
    config.device = device

    print("Loading dataset...")
    dataset = SignLanguagePoseDataset(
        data_dir=Path(args.input),
        split="test",
        chunk_len=config.arch.chunk_len,
        dtype=np.float32,
        limited_num=-1
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=zero_pad_collator, num_workers=0
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    if hasattr(dataset, 'pose_header') and dataset.pose_header is not None:
        pose_header = dataset.pose_header
        print("[INFO] Pose header loaded from dataset.")

    print("Initializing model...")
    input_feats = config.arch.keypoints * config.arch.dims
    model = SignLanguagePoseDiffusion(
        input_feats=input_feats, chunk_len=config.arch.chunk_len,
        keypoints=config.arch.keypoints, dims=config.arch.dims,
        latent_dim=config.arch.latent_dim, ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers, num_heads=config.arch.num_heads,
        dropout=getattr(config.arch, "dropout", 0.2),
        activation=getattr(config.arch, "activation", "gelu"),
        arch=config.arch.decoder, cond_mask_prob=0, device=config.device
    ).to(config.device)

    model, _ = load_checkpoint_and_config(model, args.resume, device)
    model.eval()

    diffusion = create_gaussian_diffusion(config)

    class WrappedDiffusionModel(torch.nn.Module):
        def __init__(self, model_to_wrap):
            super().__init__()
            self.model_to_wrap = model_to_wrap
        def forward(self, x_noisy_chunk, t, **kwargs):
            return self.model_to_wrap.interface(x_noisy_chunk, t, y=kwargs["y"])
    wrapped_model = WrappedDiffusionModel(model)

    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {save_dir}")

    print(f"Starting autoregressive inference: target_chunk_size_per_step={args.chunk_size}, max_len={args.max_len}, stop_threshold={args.stop_threshold}")
    total_batches = len(dataloader)
    # for batch_idx, batch_data in enumerate(dataloader):
    #     print(f"\nProcessing batch {batch_idx + 1}/{total_batches}")
    for batch_idx, batch_data in enumerate(dataloader):
        print(f"\n--- Processing batch {batch_idx + 1}/{total_batches} ---")
        print(f"Batch data type: {type(batch_data)}")
        if isinstance(batch_data, dict):
            for key, value in batch_data.items():
                print(f"  Key: '{key}'")
                if isinstance(value, torch.Tensor):
                    print(f"    Tensor shape: {value.shape}, dtype: {value.dtype}")
                elif isinstance(value, dict): # 比如 'conditions'
                    print(f"    Value is a dict:")
                    for sub_key, sub_value in value.items():
                        print(f"      Sub-key: '{sub_key}'")
                        if isinstance(sub_value, torch.Tensor):
                            print(f"        Tensor shape: {sub_value.shape}, dtype: {sub_value.dtype}")
                        else:
                            print(f"        Value type: {type(sub_value)}")
                else:
                    print(f"    Value type: {type(value)}")

        disfluent_cond_seq_loader = batch_data["conditions"]["input_sequence"].to(device)
        disfluent_cond_seq = disfluent_cond_seq_loader.permute(0, 2, 3, 1) # (B, K, D, T_loader)

        current_batch_size, K, D, T_in = disfluent_cond_seq.shape
        print(f"  Initial disfluent condition shape: (B,K,D,T) = ({current_batch_size}, {K}, {D}, {T_in})")

        generated_fluent_seq = torch.empty(current_batch_size, K, D, 0, device=device)
        active_sequences = torch.ones(current_batch_size, dtype=torch.bool, device=device)
        num_autoregressive_steps = (args.max_len + args.chunk_size - 1) // args.chunk_size

        for step in range(num_autoregressive_steps):
            if not active_sequences.any():
                print(f"  Step {step + 1}: All sequences stopped. Ending batch.")
                break
            current_generated_len = generated_fluent_seq.shape[3]
            if current_generated_len >= args.max_len:
                print(f"  Step {step + 1}: Reached max_len ({args.max_len}). Ending batch.")
                break
            n_frames_to_generate_this_step = min(args.chunk_size, args.max_len - current_generated_len)
            target_chunk_shape = (current_batch_size, K, D, n_frames_to_generate_this_step)
            print(f"  Step {step + 1}: Gen {n_frames_to_generate_this_step} frames. Total gen: {current_generated_len}.")

            model_kwargs_y = {
                "input_sequence": disfluent_cond_seq,
                "previous_output": generated_fluent_seq
            }
            model_kwargs_for_sampler = {"y": model_kwargs_y}

            with torch.no_grad():
                generated_chunk = diffusion.p_sample_loop(
                    model=wrapped_model, shape=target_chunk_shape,
                    clip_denoised=False, model_kwargs=model_kwargs_for_sampler,
                    progress=False
                )

            mean_abs_for_chunk = torch.zeros(current_batch_size, device=device)
            if generated_chunk.numel() > 0:
                 mean_abs_for_chunk[active_sequences] = generated_chunk[active_sequences].abs().mean(dim=(1, 2, 3))
            newly_stopped_sequences = (mean_abs_for_chunk < args.stop_threshold) & active_sequences
            if newly_stopped_sequences.any():
                print(f"    Sequences at indices {newly_stopped_sequences.nonzero(as_tuple=True)[0].tolist()} stopped.")
            active_sequences = active_sequences & (~newly_stopped_sequences)
            generated_chunk[newly_stopped_sequences] = torch.zeros_like(generated_chunk[newly_stopped_sequences])
            generated_fluent_seq = torch.cat([generated_fluent_seq, generated_chunk], dim=3)

        print(f"  Finished generation for batch {batch_idx + 1}. Final length: {generated_fluent_seq.shape[3]}")

        pred_fluent_normed_np_btdk = generated_fluent_seq.permute(0, 3, 1, 2).cpu().numpy() # (B, T, D, K) -> (B, T, K, D) for NumPyPoseBody

        unnorm_pred_fluent_list_np = [] # To store unnormalized numpy arrays for combined .npy file

        for i in range(current_batch_size):
            sample_idx_global = batch_idx * args.batch_size + i
            normed_pose_data_tdk = pred_fluent_normed_np_btdk[i] # (T, K, D)
            T_final, K_sample, D_sample = normed_pose_data_tdk.shape

            if T_final == 0: # Skip if no frames were generated for this sample
                unnorm_pred_fluent_list_np.append(np.empty((0, K_sample, D_sample), dtype=normed_pose_data_tdk.dtype)) # Add empty for consistent list length
                continue

            # Create Pose object from normalized data
            normed_pose_data_tpkd = normed_pose_data_tdk.reshape(T_final, 1, K_sample, D_sample) # Add person dim
            normed_confidence = np.ones((T_final, 1, K_sample), dtype=np.float32)

            fps_to_use = 25
            normed_pose_body = NumPyPoseBody(fps=fps_to_use, data=normed_pose_data_tpkd, confidence=normed_confidence)
            normed_pose_obj = Pose(pose_header, normed_pose_body)

            # Unnormalize using the new method
            unnorm_pose_obj = unnormalize_mean_std(normed_pose_obj.copy()) # Use .copy() if unnormalize_mean_std modifies in-place
            normalize_pose_size(unnorm_pose_obj) # Scale for visualization consistency

            # Save unnormalized .pose file
            with open(save_dir / f"pose_pred_fluent_{sample_idx_global}.pose", "wb") as f:
                unnorm_pose_obj.write(f)

            # Get unnormalized numpy data for combined .npy file
            unnorm_data_np_tdk = np.array(unnorm_pose_obj.body.data.data.astype(normed_pose_data_tdk.dtype)).squeeze(1) # (T, K, D)
            unnorm_pred_fluent_list_np.append(unnorm_data_np_tdk)

        # Save combined .npy files for the batch (one normalized, one unnormalized)
        np.save(save_dir / f"pred_fluent_normed_batch{batch_idx}.npy", pred_fluent_normed_np_btdk)
        if unnorm_pred_fluent_list_np: # Only save if list is not empty
            for i_un, unnorm_arr in enumerate(unnorm_pred_fluent_list_np):
                sample_idx_global_un = batch_idx * args.batch_size + i_un
                if unnorm_arr.shape[0] > 0 : # only save if there is data
                    np.save(save_dir / f"pred_fluent_unnormed_sample{sample_idx_global_un}.npy", unnorm_arr)

        print(f"  Saved predicted fluent poses for batch {batch_idx + 1} to {save_dir}")

        # Save original disfluent input for reference (using similar unnormalization)
        disfluent_input_normed_np_btdk = disfluent_cond_seq_loader.cpu().numpy() # (B, T_in_loader, K, D)

        # Save normalized disfluent input
        np.save(save_dir / f"input_disfluent_normed_batch{batch_idx}.npy", disfluent_input_normed_np_btdk)

        # Save unnormalized disfluent input
        for i in range(current_batch_size):
                sample_idx_global = batch_idx * args.batch_size + i
                normed_disfluent_data_tdk = disfluent_input_normed_np_btdk[i]
                T_disfl, K_disfl, D_disfl = normed_disfluent_data_tdk.shape

                if T_disfl == 0: continue

                normed_disfluent_data_tpkd = normed_disfluent_data_tdk.reshape(T_disfl, 1, K_disfl, D_disfl)
                disfl_confidence = np.ones((T_disfl, 1, K_disfl), dtype=np.float32)

                fps_to_use = pose_header.fps if hasattr(pose_header, 'fps') and pose_header.fps > 0 else 25
                normed_disfl_body = NumPyPoseBody(fps=fps_to_use, data=normed_disfluent_data_tpkd, confidence=disfl_confidence)
                normed_disfl_obj = Pose(pose_header, normed_disfl_body)

                unnorm_disfl_obj = unnormalize_mean_std(normed_disfl_obj.copy())
                normalize_pose_size(unnorm_disfl_obj) # Optional

                # Save unnormalized disfluent .pose file
                with open(save_dir / f"pose_input_disfluent_{sample_idx_global}.pose", "wb") as f:
                    unnorm_disfl_obj.write(f)

                # Save individual unnormalized disfluent .npy
                unnorm_disfl_np_tdk = np.array(unnorm_disfl_obj.body.data.data.astype(normed_disfluent_data_tdk.dtype)).squeeze(1)
                np.save(save_dir / f"input_disfluent_unnormed_sample{sample_idx_global}.npy", unnorm_disfl_np_tdk)

        print(f"  Saved input disfluent poses for batch {batch_idx + 1} to {save_dir}")

    print("\nAutoregressive inference finished for all batches.")

if __name__ == "__main__":
    main()