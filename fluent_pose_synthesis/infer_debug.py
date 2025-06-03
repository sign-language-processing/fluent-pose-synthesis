# Example usage:
# python -m fluent_pose_synthesis.infer_debug \
# -i assets/sample_dataset \
# -c save/debug_run/4th_train_whole_dataset_output/config.json \
# -r save/debug_run/4th_train_whole_dataset_output/best_model_validation.pt \
# -o save/debug_run/4th_train_whole_dataset_output/validation_infer_output_steps \
# --batch_size 64 \
# --chunk_size 40 \
# --max_len 40 \
# --stop_threshold 1e-4 \
# --seed 1234

import argparse
import json
from pathlib import Path, PosixPath
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
import numpy as np
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
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to disfluent input data directory (should contain a 'test' split)")
    parser.add_argument("-c", "--config", required=True, type=str,
                        help="Path to model config JSON file (from training)")
    parser.add_argument("-r", "--resume", required=True, type=str, help="Path to model checkpoint (.pt)")
    parser.add_argument("-o", "--output", default="output/infer_results", type=str,
                        help="Directory to save generated outputs")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for inference.")
    parser.add_argument("--chunk_size", default=10, type=int,
                        help="Number of frames to attempt to generate per autoregressive step.")
    parser.add_argument("--max_len", default=100, type=int, help="Maximum total frames to generate.")
    parser.add_argument("--stop_threshold", default=1e-5, type=float,
                        help="Threshold for mean absolute value of generated chunk to detect stop condition.")
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
            config_namespace = eval(
                f.read(), {
                    "SimpleNamespace": SimpleNamespace, "namespace": SimpleNamespace, "PosixPath": PosixPath, "device":
                    torch.device, "Path": Path
                })  # type: ignore
        config_dict = convert_namespace_to_dict(config_namespace)
        standard_json_path = Path(args.config).with_suffix('.fixed.json')
        with open(standard_json_path, "w", encoding="utf-8") as f_json:
            json.dump(config_dict, f_json, indent=4)
        print(f"[INFO] Saved successfully parsed config to {standard_json_path}")

    config = dict_to_namespace(config_dict)
    config.inference = SimpleNamespace(chunk_size=args.chunk_size, max_len=args.max_len,
                                       stop_threshold=args.stop_threshold)
    config.device = device

    print("Loading dataset...")
    dataset = SignLanguagePoseDataset(
        data_dir=Path(args.input),
        split="validation",
        # split="train",  # use trainset for now during overfitting stage
        chunk_len=config.arch.chunk_len,  # From training config
        history_len=getattr(config.arch, "history_len", 5),  # Use default from load_data.py or from config if added
        dtype=np.float32,
        limited_num=-1  # Load all samples from the test set
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=zero_pad_collator,
        num_workers=0  # num_workers=0 for easier debugging
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    print(
        f"[DEBUG infer.py] Raw dataset.input_mean shape: {dataset.input_mean.shape if hasattr(dataset.input_mean, 'shape') else 'N/A'}"
    )
    print(
        f"[DEBUG infer.py] Raw dataset.input_std shape: {dataset.input_std.shape if hasattr(dataset.input_std, 'shape') else 'N/A'}"
    )
    print(f"[DEBUG infer.py] Raw dataset.input_mean value (first few elements):\n{dataset.input_mean.ravel()[:5]}")

    data_input_mean_np = np.array(dataset.input_mean).squeeze()
    data_input_std_np = np.array(dataset.input_std).squeeze()
    data_cond_mean_np = np.array(dataset.condition_mean).squeeze()
    data_cond_std_np = np.array(dataset.condition_std).squeeze()

    print(f"[DEBUG infer.py] Squeezed data_input_mean_np shape: {data_input_mean_np.shape}")
    print(f"[DEBUG infer.py] Squeezed data_input_std_np shape: {data_input_std_np.shape}")
    print(f"[DEBUG infer.py] Squeezed data_input_mean_np value (first few elements):\n{data_input_mean_np.ravel()[:5]}")

    expected_stat_shape_suffix = (config.arch.keypoints, config.arch.dims)
    print(f"[DEBUG infer.py] Expected stat shape suffix: {expected_stat_shape_suffix}")

    assert data_input_mean_np.shape[-len(expected_stat_shape_suffix):] == expected_stat_shape_suffix, \
        f"Mean shape mismatch. Actual: {data_input_mean_np.shape}, Expected suffix: {expected_stat_shape_suffix}"
    assert data_input_std_np.shape[-len(expected_stat_shape_suffix):] == expected_stat_shape_suffix, \
        f"Std shape mismatch. Actual: {data_input_std_np.shape}, Expected suffix: {expected_stat_shape_suffix}"

    pose_header = dataset.pose_header
    print("[INFO] Pose header loaded from dataset.")

    print("Initializing model...")
    input_feats = config.arch.keypoints * config.arch.dims
    model = SignLanguagePoseDiffusion(input_feats=input_feats, chunk_len=config.arch.chunk_len,
                                      keypoints=config.arch.keypoints, dims=config.arch.dims,
                                      latent_dim=config.arch.latent_dim, ff_size=config.arch.ff_size,
                                      num_layers=config.arch.num_layers, num_heads=config.arch.num_heads,
                                      dropout=getattr(config.arch, "dropout",
                                                      0.2), activation=getattr(config.arch, "activation", "gelu"),
                                      arch=config.arch.decoder, cond_mask_prob=0, device=config.device,
                                      batch_first=getattr(config.arch, "batch_first", True)).to(config.device)

    model, _ = load_checkpoint_and_config(model, args.resume, device)
    model.eval()

    diffusion = create_gaussian_diffusion(config)
    print(f"ACTUAL diffusion.num_timesteps = {diffusion.num_timesteps}")
    print(f"Betas: {diffusion.betas}")
    print(f"Alphas Cumprod: {diffusion.alphas_cumprod}")
    print(f"Sqrt Alphas Cumprod: {diffusion.sqrt_alphas_cumprod}")
    print(f"Sqrt One Minus Alphas Cumprod: {diffusion.sqrt_one_minus_alphas_cumprod}")

    class WrappedDiffusionModel(torch.nn.Module):

        def __init__(self, model_to_wrap):
            super().__init__()
            self.model_to_wrap = model_to_wrap

        def forward(self, x_noisy_chunk, t, **kwargs):
            # The model's interface expects y=conditions_dict
            return self.model_to_wrap.interface(x_noisy_chunk, t, y=kwargs["y"])

    wrapped_model = WrappedDiffusionModel(model)

    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {save_dir}")

    processed_originals = {}
    unique_outputs_generated = 0  # Track how many unique fluent sequences are actually generated

    print(
        f"Starting autoregressive inference: target_chunk_size_per_step={args.chunk_size}, max_len={args.max_len}, stop_threshold={args.stop_threshold}"
    )
    total_batches = len(dataloader)

    history_len_for_inference = getattr(config.arch, "history_len", 5)
    print(f"[INFO] Using history_len_for_inference: {history_len_for_inference}")

    for batch_idx, batch_data in enumerate(dataloader):
        print(f"\n--- Processing batch {batch_idx + 1}/{total_batches} ---")

        disfluent_cond_seq_batch_loader = batch_data["conditions"]["input_sequence"].to(device)
        metadata_from_collate = batch_data.get("metadata")

        tasks_for_this_batch = []

        current_actual_batch_size = disfluent_cond_seq_batch_loader.shape[0]

        for i in range(current_actual_batch_size):  # Iterate over each logical sample in the batch
            current_original_id = None

            # Extract metadata of the i-th sample from the already collated metadata_from_collate
            # Prefer using file path as ID
            if "original_disfluent_filepath" in metadata_from_collate and i < len(
                    metadata_from_collate["original_disfluent_filepath"]):
                current_original_id = metadata_from_collate["original_disfluent_filepath"][i]
            # If file path is missing or index is out of range, try using example_index
            elif "original_example_index" in metadata_from_collate and i < len(
                    metadata_from_collate["original_example_index"]):
                current_original_id = f"example_idx_{metadata_from_collate['original_example_index'][i]}"

            if current_original_id not in processed_originals:
                individual_disfluent_seq_tensor = disfluent_cond_seq_batch_loader[i:i + 1].permute(0, 2, 3, 1)
                tasks_for_this_batch.append((current_original_id, i, individual_disfluent_seq_tensor))
                processed_originals[current_original_id] = "PROCESSING"
            else:
                print(f"  Skipping already processed or queued original sequence ID: {current_original_id}")

        if not tasks_for_this_batch:
            print("  No new sequences to process in this batch.")
            continue  # Skip to the next batch

        # Combine all unique tasks collected in this batch into a new "mini-batch" for generation
        # The shape of disfluent_cond_seq_for_generation will be (num_unique_tasks, K, D, T_in)
        disfluent_cond_seq_for_generation = torch.cat([task[2] for task in tasks_for_this_batch], dim=0)

        current_batch_size_for_generation = disfluent_cond_seq_for_generation.shape[0]
        K, D, T_in = disfluent_cond_seq_for_generation.shape[1:]  # Get dimensions from the merged tensor

        print(f"  Combined {current_batch_size_for_generation} unique sequences for generation from this batch.")
        print(
            f"  Effective disfluent condition shape for model: (B_eff, K, D, T) = ({current_batch_size_for_generation}, {K}, {D}, {T_in})"
        )

        # Start autoregressive generation for this mini-batch
        generated_fluent_seq = torch.empty(current_batch_size_for_generation, K, D, 0,
                                           device=device)  # (B_eff, K, D, T_gen)
        active_sequences = torch.ones(current_batch_size_for_generation, dtype=torch.bool, device=device)
        num_autoregressive_steps = (args.max_len + args.chunk_size - 1) // args.chunk_size

        for step in range(num_autoregressive_steps):
            if not active_sequences.any():
                print(f"  Step {step + 1}: All sequences stopped for this effective batch.")
                break
            current_generated_len = generated_fluent_seq.shape[3]
            if current_generated_len >= args.max_len:
                print(f"  Step {step + 1}: Reached max_len ({args.max_len}).")
                break

            n_frames_to_generate_this_step = min(args.chunk_size, args.max_len - current_generated_len)
            target_chunk_shape = (current_batch_size_for_generation, K, D, n_frames_to_generate_this_step)
            print(
                f"  Step {step + 1}: Gen {n_frames_to_generate_this_step} frames. Total gen: {current_generated_len}.")

            effective_previous_output = torch.empty(current_batch_size_for_generation, K, D, 0, device=device)
            if current_generated_len > 0:
                if current_generated_len < history_len_for_inference:
                    start_index = max(0, current_generated_len - history_len_for_inference)
                    effective_previous_output = generated_fluent_seq[:, :, :, start_index:]
                else:
                    effective_previous_output = generated_fluent_seq[:, :, :, -history_len_for_inference:]

            print(f"    Effective previous_output shape for model: {effective_previous_output.shape}")

            model_kwargs_y = {
                "input_sequence": disfluent_cond_seq_for_generation, "previous_output": effective_previous_output
            }
            model_kwargs_for_sampler = {"y": model_kwargs_y}

            # --- Progressive sampling and per-step saving ---
            with torch.no_grad():
                # Prepare to collect all pred_xstart for each step in the progressive sampling
                all_steps_pred_xstart = []
                for prog_step, sample in enumerate(
                        diffusion.p_sample_loop_progressive(
                            # diffusion.ddim_sample_loop_progressive(
                            model=wrapped_model,
                            shape=target_chunk_shape,
                            clip_denoised=False,
                            model_kwargs=model_kwargs_for_sampler,
                            progress=False,
                            # eta=0.0,
                        )):
                    pred_xstart = sample["pred_xstart"].cpu().numpy()  # shape (B_eff, K, D, chunk)
                    for task_idx, (original_id, _, _) in enumerate(tasks_for_this_batch):
                        if "/" in original_id or "\\" in original_id:
                            filename_base_from_id = Path(original_id).stem
                        else:
                            filename_base_from_id = original_id
                        np.save(save_dir / f"pose_pred_fluent_{filename_base_from_id}_step{prog_step}.npy",
                                pred_xstart[task_idx])
                        single_pred = pred_xstart[task_idx]  # (K, D, chunk)
                        # transpose to (chunk, K, D)
                        single_pred = np.transpose(single_pred, (2, 0, 1))  # (chunk, K, D)
                        # reshape to (chunk, 1, K, D)
                        single_pred = single_pred.reshape(single_pred.shape[0], 1, single_pred.shape[1],
                                                          single_pred.shape[2])  # (chunk, 1, K, D)
                        # unnormalize
                        unnorm_pred = single_pred * data_input_std_np + data_input_mean_np
                        # confidence
                        confidence = np.ones((unnorm_pred.shape[0], 1, unnorm_pred.shape[2]), dtype=np.float32)
                        fps_to_use = pose_header.fps if hasattr(pose_header, 'fps') and pose_header.fps > 0 else 25.0
                        pose_body = NumPyPoseBody(fps=fps_to_use, data=unnorm_pred, confidence=confidence)
                        pose_obj = Pose(pose_header, pose_body)
                        with open(save_dir / f"pose_pred_fluent_{filename_base_from_id}_step{prog_step}.pose",
                                  "wb") as f:
                            pose_obj.write(f)
                    all_steps_pred_xstart.append(pred_xstart)
                # Use the last step as the generated chunk (convert to tensor, move to device)
                generated_chunk = torch.tensor(all_steps_pred_xstart[-1], device=device)

            mean_abs_for_chunk = torch.zeros(current_batch_size_for_generation, device=device)
            if generated_chunk.numel() > 0:
                mean_abs_for_chunk[active_sequences] = generated_chunk[active_sequences].abs().mean(dim=(1, 2, 3))

            newly_stopped_sequences = (mean_abs_for_chunk < args.stop_threshold) & active_sequences
            if newly_stopped_sequences.any():
                print(
                    f"    Sequences (effective batch indices) {newly_stopped_sequences.nonzero(as_tuple=True)[0].tolist()} stopped this step."
                )

            active_sequences = active_sequences & (~newly_stopped_sequences)
            generated_chunk[newly_stopped_sequences] = torch.zeros_like(generated_chunk[newly_stopped_sequences])
            generated_fluent_seq = torch.cat([generated_fluent_seq, generated_chunk], dim=3)

        print(
            f"  Finished generation for {current_batch_size_for_generation} unique sequences. Final length: {generated_fluent_seq.shape[3]}"
        )

        # --- Save results ---
        pred_fluent_normed_np_bEff_T_K_D = generated_fluent_seq.permute(0, 3, 1,
                                                                        2).cpu().numpy()  # (B_eff, T_final, K, D)

        input_disfluent_normed_np_bEff_T_K_D = disfluent_cond_seq_for_generation.permute(0, 3, 1, 2).cpu().numpy()

        for task_idx, (original_id, original_batch_idx, _) in enumerate(tasks_for_this_batch):
            # Extract current task data from the merged batch
            current_pred_fluent_normed_np = pred_fluent_normed_np_bEff_T_K_D[task_idx]  # (T_final, K, D)
            current_input_disfluent_normed_np = input_disfluent_normed_np_bEff_T_K_D[task_idx]  # (T_in, K, D)

            # --- Unnormalize and save logic ---
            # Unnormalize predicted results using data_input_mean_np, data_input_std_np (from dataset)
            unnorm_pred_fluent_np_tdk = current_pred_fluent_normed_np * data_input_std_np + data_input_mean_np

            # Unnormalize input using data_cond_mean_np, data_cond_std_np (from dataset)
            # (Assume these stats have already been obtained from the dataset earlier in infer.py)
            unnorm_input_disfluent_np_tdk = current_input_disfluent_normed_np * data_cond_std_np + data_cond_mean_np

            # Use original_id as basis for filename to ensure uniqueness
            # original_id may be a file path, extract filename part
            if "/" in original_id or "\\" in original_id:  # If it's a file path
                filename_base_from_id = Path(original_id).stem
            else:  # If it's "example_idx_X" or "dataloader_item_X"
                filename_base_from_id = original_id

            save_prefix_pred = f"pose_pred_fluent_{filename_base_from_id}"
            save_prefix_input = f"pose_input_disfluent_{filename_base_from_id}"

            # Save predicted fluent pose (.pose and .npy)
            T_final_sample, K_sample, D_sample = unnorm_pred_fluent_np_tdk.shape
            if T_final_sample > 0 and pose_header is not None:
                unnorm_pose_data_tpkd = unnorm_pred_fluent_np_tdk.reshape(T_final_sample, 1, K_sample, D_sample)
                unnorm_confidence = np.ones((T_final_sample, 1, K_sample), dtype=np.float32)
                fps_to_use = pose_header.fps if hasattr(pose_header, 'fps') and pose_header.fps > 0 else 25.0
                unnorm_pose_body = NumPyPoseBody(fps=fps_to_use, data=unnorm_pose_data_tpkd,
                                                 confidence=unnorm_confidence)
                unnorm_pose_obj = Pose(pose_header, unnorm_pose_body)

                with open(save_dir / f"{save_prefix_pred}.pose", "wb") as f:
                    unnorm_pose_obj.write(f)
                np.save(save_dir / f"{save_prefix_pred}_unnormed.npy", unnorm_pred_fluent_np_tdk)
                np.save(save_dir / f"{save_prefix_pred}_normed.npy", current_pred_fluent_normed_np)

            # Save original disfluent pose (.pose and .npy)
            T_in_sample, K_in_sample, D_in_sample = unnorm_input_disfluent_np_tdk.shape
            if T_in_sample > 0 and pose_header is not None:
                unnorm_input_data_tpkd = unnorm_input_disfluent_np_tdk.reshape(T_in_sample, 1, K_in_sample, D_in_sample)
                input_confidence = np.ones((T_in_sample, 1, K_in_sample), dtype=np.float32)
                fps_to_use = pose_header.fps if hasattr(pose_header, 'fps') and pose_header.fps > 0 else 25.0
                unnorm_input_body = NumPyPoseBody(fps=fps_to_use, data=unnorm_input_data_tpkd,
                                                  confidence=input_confidence)
                unnorm_input_obj = Pose(pose_header, unnorm_input_body)

                with open(save_dir / f"{save_prefix_input}.pose", "wb") as f:
                    unnorm_input_obj.write(f)
                np.save(save_dir / f"{save_prefix_input}_unnormed.npy", unnorm_input_disfluent_np_tdk)
                np.save(save_dir / f"{save_prefix_input}_normed.npy", current_input_disfluent_normed_np)

            # Mark this original_id as successfully processed and saved
            processed_originals[original_id] = f"{save_prefix_pred}.pose"
            unique_outputs_generated += 1

    print(
        f"\nAutoregressive inference finished. Generated and saved {unique_outputs_generated} unique fluent sequences.")


if __name__ == "__main__":
    main()
