import sys
import os

import time
import shutil
import argparse
import json
from pathlib import Path, PosixPath
from types import SimpleNamespace

import numpy as np
import torch
import torch.serialization
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pose_format.torch.masked.collator import zero_pad_collator
from CAMDM.diffusion.create_diffusion import create_gaussian_diffusion
from CAMDM.utils.common import fixseed, select_platform
from CAMDM.utils.logger import Logger
from fluent_pose_synthesis.core.models import SignLanguagePoseDiffusion
from fluent_pose_synthesis.core.training import PoseTrainingPortal
from fluent_pose_synthesis.data.load_data import SignLanguagePoseDataset

from fluent_pose_synthesis.config.option import (
    add_model_args,
    add_train_args,
    add_diffusion_args,
    config_parse,
)

# Add custom globals to torch.serialization
torch.serialization.add_safe_globals([
    SimpleNamespace,
    PosixPath,
    np.int64,
    np.int32,
    np.float64,
    np.float32,
    np.bool_,
])
# Patch torch.load to avoid loading weights only
# This is a workaround for the issue where torch.load tries to load weights only
_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load


def train(
    config: argparse.Namespace,
    resume_path: Path,
    logger: Logger,
    tb_writer: SummaryWriter,
):
    """Main training loop for sign language pose post-editing."""
    fixseed(1024)
    np_dtype = select_platform(32)

    # Training Dataset and Dataloader
    logger.info("Loading training dataset...")
    train_dataset = SignLanguagePoseDataset(
        data_dir=config.data,
        split="train",
        chunk_len=config.arch.chunk_len,
        history_len=getattr(config.arch, "history_len", 5),
        dtype=np_dtype,
        limited_num=config.trainer.load_num,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=zero_pad_collator,
    )
    logger.info(f"Training Dataset includes {len(train_dataset)} samples, "
                f"with {config.arch.chunk_len} fluent frames per sample.")

    # Validation Dataset and Dataloader
    logger.info("Loading validation dataset...")
    validation_dataset = SignLanguagePoseDataset(data_dir=config.data, split="validation",
                                                 chunk_len=config.arch.chunk_len,
                                                 history_len=getattr(config.arch, "history_len", 5), dtype=np_dtype,
                                                 limited_num=config.trainer.load_num)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=config.trainer.workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=zero_pad_collator,
    )
    logger.info(f"Validation Dataset includes {len(validation_dataset)} samples.")

    # Model and Diffusion Initialization
    diffusion = create_gaussian_diffusion(config)
    input_feats = config.arch.keypoints * config.arch.dims

    model = SignLanguagePoseDiffusion(
        input_feats=input_feats,
        chunk_len=config.arch.chunk_len,
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
        device=config.device,
    ).to(config.device)

    logger.info(f"Model: {model}")

    # Training Portal Initialization
    trainer = PoseTrainingPortal(config, model, diffusion, train_dataloader, logger, tb_writer,
                                 validation_dataloader=validation_dataloader)

    if resume_path is not None:
        try:
            trainer.load_checkpoint(str(resume_path))
        except FileNotFoundError:
            print(f"No checkpoint found at {resume_path}")
            sys.exit(1)

    custom_profiler_directory = config.save / "profiler_logs"
    custom_profiler_directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Profiler output will be directed to: {custom_profiler_directory}")

    trainer.run_loop(enable_profiler=True, profiler_directory=str(custom_profiler_directory))
    # trainer.run_loop()


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="### Fluent Sign Language Pose Synthesis Training ###")
    parser.add_argument("-n", "--name", default="debug", type=str, help="The name of this training run")
    parser.add_argument(
        "-c",
        "--config",
        default="./fluent_pose_synthesis/config/default.json",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "-i",
        "--data",
        default="/pose_data/output",
        type=str,
        help="Path to dataset folder",
    )
    parser.add_argument("-r", "--resume", default=None, type=str, help="Path to latest checkpoint")
    parser.add_argument(
        "-s",
        "--save",
        default="save/debug_run",
        type=str,
        help="Directory to save model and logs",
    )
    parser.add_argument("--cluster", action="store_true", help="Enable cluster mode")

    add_model_args(parser)
    add_diffusion_args(parser)
    add_train_args(parser)

    args = parser.parse_args()
    config = config_parse(args)

    # Convert key paths to Path objects
    config.data = Path(config.data)
    config.save = Path(config.save)

    if args.cluster:
        config.data = Path("/scratch/ronli/pose_data/output") / args.data
        config.save = Path("/scratch/ronli/save") / args.name

    # Debug mode settings
    if "debug" in args.name:
        config.trainer.workers = 1
        config.trainer.load_num = -1
        config.trainer.batch_size = 32
        config.trainer.epoch = 2000

    # Handle existing folder
    if (not args.cluster and config.save.exists() and "debug" not in args.name and args.resume is None):
        allow_cover = input("Model folder exists. Overwrite? (Y/N): ").lower()
        if allow_cover == "n":
            sys.exit(0)
        shutil.rmtree(config.save, ignore_errors=True)

    # Auto-resume for cluster
    resume_path = None
    if config.save.exists() and args.resume is None:
        best_ckpt = config.save / "best.pt"
        if best_ckpt.exists():
            resume_path = best_ckpt
        else:
            ckpts = list(config.save.glob("weights_*.pt"))
            if ckpts:
                latest = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
                resume_path = latest

    config.save.mkdir(parents=True, exist_ok=True)

    logger = Logger(config.save / "log.txt")
    tb_writer = SummaryWriter(log_dir=config.save / "runtime")

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save config
    with open(config.save / "config.json", "w", encoding="utf-8") as f:
        # Convert SimpleNamespace to dict for JSON serialization
        json.dump(config_to_dict(config), f, indent=4)
        logger.info(f"Saved final configuration to {config.save / 'config.json'}")

    logger.info(f"\nLaunching training with config:\n{config}")
    train(config, resume_path, logger, tb_writer)
    logger.info(f"\nTotal training time: {(time.time() - start_time) / 60:.2f} mins")


def config_to_dict(config_namespace):
    """Helper to convert SimpleNamespace (recursively) to dict for JSON."""
    if isinstance(config_namespace, SimpleNamespace):
        return {k: config_to_dict(v) for k, v in vars(config_namespace).items()}
    elif isinstance(config_namespace, Path):
        return str(config_namespace)
    elif isinstance(config_namespace, (list, tuple)):
        return [config_to_dict(i) for i in config_namespace]
    elif isinstance(config_namespace, torch.device):
        return str(config_namespace)
    else:
        return config_namespace


if __name__ == "__main__":
    main()
