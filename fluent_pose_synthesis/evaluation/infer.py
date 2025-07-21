import argparse
import json
from pathlib import Path, PosixPath
from types import SimpleNamespace
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm
import torch
import torch.serialization
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
from pose_evaluation.metrics.pose_processors import NormalizePosesProcessor
from CAMDM.diffusion.create_diffusion import create_gaussian_diffusion
from CAMDM.utils.common import fixseed

from fluent_pose_synthesis.core.models import SignLanguagePoseDiffusion
from fluent_pose_synthesis.data.load_data import SignLanguagePoseDataset
from fluent_pose_synthesis.core.training import _ConditionalWrapper

# Example usage:
# python -m fluent_pose_synthesis.evaluation.infer \
# -i /scratch/ronli/fluent-pose-synthesis/pose_data/output \
# -c /scratch/ronli/fluent-pose-synthesis/save/final_train_velocity_output/config.json \
# -r /scratch/ronli/fluent-pose-synthesis/save/final_train_velocity_output/best.pt \
# -o /scratch/ronli/fluent-pose-synthesis/save/final_train_velocity_output/best_training_ckpt_on_test \
# --split test

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.serialization.add_safe_globals([
    SimpleNamespace,
    Path,
    PosixPath,
    np.int64,
    np.int32,
    np.float64,
    np.float32,
    np.bool_,
])


def dict_to_namespace(d: Any) -> Any:
    """
    Recursively convert nested dictionaries and lists into SimpleNamespace objects for attribute-style access.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def convert_namespace_to_dict(ns: Any) -> Any:
    """
    Recursively convert SimpleNamespace objects (and Path or device types) back into primitive dicts and strings for JSON serialization.
    """
    if isinstance(ns, SimpleNamespace):
        return {k: convert_namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, dict):
        return {k: convert_namespace_to_dict(v) for k, v in ns.items()}
    elif isinstance(ns, (Path, PosixPath, torch.device)):
        return str(ns)
    else:
        return ns


def load_checkpoint_and_config(model: torch.nn.Module, ckpt_path: str,
                               device: torch.device) -> Tuple[torch.nn.Module, Union[Dict[str, Any], None]]:
    """
    Load model weights from a checkpoint and extract its config if present.
    Returns the model and the loaded config dict (or None).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded model checkpoint from {ckpt_path}")
    cfg = ckpt.get("config", None)
    if cfg is not None:
        if isinstance(cfg, SimpleNamespace):
            cfg = convert_namespace_to_dict(cfg)
        print("[INFO] Config found in checkpoint.")
        return model, cfg
    return model, None


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for inference.
    """
    p = argparse.ArgumentParser(
        description="Autoregressive inference for fluent pose synthesis with dynamic length prediction.")
    p.add_argument("-i", "--input", required=True, help="Path to input data dir (with split subdir)")
    p.add_argument("-c", "--config", required=True, help="Path to model config JSON")
    p.add_argument("-r", "--resume", required=True, help="Path to model checkpoint .pt")
    p.add_argument("-o", "--output", default="output/infer_results", help="Output dir")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size; recommended 1")
    p.add_argument("--chunk_size", type=int, default=40, help="Frames per infer step")
    p.add_argument("--stop_threshold", type=float, default=1e-4, help="Mean abs threshold for early stop")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--regression_slope", type=float, default=0.32, help="Slope a for length regression")
    p.add_argument("--regression_intercept", type=float, default=16.37, help="Intercept b for length regression")
    p.add_argument("--split", default="validation", help="Which split to run on")
    return p.parse_args()


def load_config(config_path: str, device: torch.device) -> Any:
    """
    Load model configuration from a JSON file and attach the target device.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = dict_to_namespace(cfg)
    cfg.device = device
    return cfg


def load_dataset(args: argparse.Namespace, cfg: Any) -> Tuple[Any, DataLoader, Any, List[str], np.ndarray, np.ndarray]:
    """
    Initialize the dataset and DataLoader.
    Collect valid sample IDs based on metadata length filters.
    """
    data_dir = Path(args.input)
    dataset = SignLanguagePoseDataset(data_dir=data_dir, split=args.split, chunk_len=cfg.arch.chunk_len,
                                      history_len=getattr(cfg.arch, "history_len", 10),
                                      fixed_condition_length=getattr(cfg.arch, "fixed_condition_length", 200),
                                      min_condition_length=getattr(cfg.arch, "min_condition_length",
                                                                   25), dtype=np.float32, limited_num=-1)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=zero_pad_collator, num_workers=0)
    pose_header = dataset.pose_header
    split_dir = data_dir / args.split
    fluent_files = sorted(split_dir.glob(f"{args.split}_*_original.pose"))
    valid_ids = []
    min_len = getattr(cfg.arch, "min_condition_length", 25)
    for f in fluent_files:
        m = f.with_name(f.name.replace("_original.pose", "_metadata.json"))
        if m.exists():
            md = json.load(open(m, "r", encoding="utf-8"))
            if md.get("disfluent_pose_length", 0) >= min_len:
                valid_ids.append(f.stem.replace("_original", ""))
        else:
            logger.warning(f"Metadata not found for {f}")
    mean = np.array(dataset.input_mean).squeeze()
    std = np.array(dataset.input_std).squeeze()
    return dataset, dl, pose_header, valid_ids, mean, std


def load_model(cfg: Any, resume_path: str, device: torch.device) -> torch.nn.Module:
    """
    Build and load the diffusion model with checkpoint weights,
    switch it to evaluation mode, and return it.
    """
    inp_feats = cfg.arch.keypoints * cfg.arch.dims
    model = SignLanguagePoseDiffusion(input_feats=inp_feats, chunk_len=cfg.arch.chunk_len, keypoints=cfg.arch.keypoints,
                                      dims=cfg.arch.dims, latent_dim=cfg.arch.latent_dim, ff_size=cfg.arch.ff_size,
                                      num_layers=cfg.arch.num_layers, num_heads=cfg.arch.num_heads,
                                      dropout=getattr(cfg.arch, "dropout",
                                                      0.2), activation=getattr(cfg.arch, "activation", "gelu"),
                                      arch=cfg.arch.decoder, cond_mask_prob=0, device=cfg.device,
                                      batch_first=getattr(cfg.arch, "batch_first", True)).to(device)
    model, _ = load_checkpoint_and_config(model, resume_path, device)
    model.eval()
    return model


def run_generation_and_save(model: torch.nn.Module, dl: DataLoader, pose_header: Any, valid_ids: List[str],
                            mean: np.ndarray, std: np.ndarray, save_dir: Path, args: argparse.Namespace, cfg: Any,
                            diffusion: Any, device: torch.device) -> List[Dict[str, Path]]:
    """
    Perform autoregressive generation of fluent pose sequences:
    1. Iterate over batches
    2. Apply dynamic length prediction
    3. Generate chunks until stop condition or max length
    4. Save each generated sequence to a .pose file
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    eval_pairs = []
    history_len = getattr(cfg.arch, "history_len", 10)

    for batch in tqdm(dl, desc="Generating Sequences"):
        cond_loader = batch["conditions"]["input_sequence"].to(device)
        cond = cond_loader.permute(0, 2, 3, 1)  # (B,C,F,T)
        B, C, F, T_in = cond.shape
        md = batch.get("metadata", {})
        dlen = md.get("disfluent_pose_length", [200])[0]
        use_len = min(dlen, 200)
        max_len = max(1, int(args.regression_slope * use_len + args.regression_intercept))
        logger.info(
            f"Processing sample. Original disfluent length: {dlen} (used for calc: {use_len}), dynamic target length: {max_len}"
        )

        gen = torch.empty(B, C, F, 0, device=device)
        active = torch.ones(B, device=device, dtype=torch.bool)

        # Start autoregressive sampling loop for this sample
        while gen.shape[3] < max_len and active.any():
            cur = gen.shape[3]
            n = min(args.chunk_size, max_len - cur)
            if cur < history_len:
                p = torch.zeros(B, C, F, history_len - cur, device=device)
                prev = torch.cat([p, gen], dim=3)
            else:
                prev = gen[..., -history_len:]
            cond_dict = {"input_sequence": cond, "previous_output": prev}
            wrapped = _ConditionalWrapper(model, cond_dict)
            with torch.no_grad():
                chunk = diffusion.p_sample_loop(model=wrapped, shape=(B, C, F, n), clip_denoised=False,
                                                model_kwargs={"y": cond_dict}, progress=False)
            mag = chunk.abs().mean(dim=(1, 2, 3))
            if (mag < args.stop_threshold).any():
                active.fill_(False)
            if active.any():
                gen = torch.cat([gen, chunk], dim=3)
        logger.info(f"Finished generation for sample. Final length: {gen.shape[3]}")

        # save outputs
        if gen.shape[3] == 0:
            continue
        arr = gen.squeeze(0).permute(2, 0, 1).cpu().numpy()  # (T,K,D)
        arr = arr * std + mean
        idx = md.get("original_example_index", [0])[0]
        bid = valid_ids[idx] if idx < len(valid_ids) else f"idx{idx}"
        prefix = f"pred_fluent_{bid}"
        T_out, K, D_out = arr.shape
        if T_out > 0:
            data = arr.reshape(T_out, 1, K, D_out)
            conf = np.ones((T_out, 1, K), dtype=np.float32)
            fps = getattr(pose_header, "fps", 25)
            body = NumPyPoseBody(fps=fps, data=data, confidence=conf)
            obj = Pose(pose_header, body)
            out = save_dir / f"{prefix}.pose"
            with open(out, "wb") as f:
                obj.write(f)
            gt = Path(args.input) / args.split / f"{bid}_original.pose"
            eval_pairs.append({"generated": out, "ground_truth": gt})
    return eval_pairs


def final_dtw_evaluation(eval_pairs: List[Dict[str, Path]], split: str) -> float:
    """
    Compute and print the mean DTW score over generated-ground truth pairs.
    """
    if not eval_pairs:
        print("No sequences were generated, skipping evaluation.")
        return 0.0
    metric = DistanceMetric(
        name="model_performance_DTW",
        distance_measure=DTWDTAIImplementationDistanceMeasure(name="dtaiDTW", use_fast=True, default_distance=0.0),
        pose_preprocessors=[NormalizePosesProcessor()],
    )
    scores = []
    for pair in tqdm(eval_pairs, desc="Calculating DTW"):
        if not pair["ground_truth"].exists():
            print(f"Warning: Ground truth not found: {pair['ground_truth']}")
            continue
        gen = Pose.read(open(pair["generated"], "rb").read())
        gt = Pose.read(open(pair["ground_truth"], "rb").read())
        gen.body.data = gen.body.data.astype(np.double)
        gt.body.data = gt.body.data.astype(np.double)
        scores.append(metric(gen, gt))
    if not scores:
        print("No valid DTW scores calculated.")
        return 0.0
    m = np.mean(scores)
    print(f"\n=== Final Model Performance on '{split}' split ===")
    print(f"Total sequences evaluated: {len(scores)}")
    print(f"Mean DTW Score: {m:.4f}")
    return m


def main():
    args = parse_args()
    if args.batch_size != 1:
        logger.warning(f"Batch size {args.batch_size} may affect dynamic length prediction.")
    fixseed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = load_config(args.config, device)
    ds, dl, header, vids, mean, std = load_dataset(args, cfg)
    model = load_model(cfg, args.resume, device)
    diffusion = create_gaussian_diffusion(cfg)
    save_dir = Path(args.output)
    logger.info(f"Output will be saved to: {save_dir}")

    eval_pairs = run_generation_and_save(model, dl, header, vids, mean, std, save_dir, args, cfg, diffusion, device)
    m = final_dtw_evaluation(eval_pairs, args.split)
    logger.info(f"Mean DTW: {m:.4f}")


if __name__ == "__main__":
    main()
