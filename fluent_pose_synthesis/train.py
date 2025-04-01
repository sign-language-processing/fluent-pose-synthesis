import sys
import time
import shutil
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pose_format.torch.masked.collator import zero_pad_collator
from CAMDM.PyTorch.diffusion.create_diffusion import create_gaussian_diffusion
from CAMDM.PyTorch.utils.common import fixseed, select_platform
from CAMDM.PyTorch.utils.logger import Logger
from fluent_pose_synthesis.core.models import SignLanguagePoseDiffusion
from fluent_pose_synthesis.core.training import PoseTrainingPortal
from fluent_pose_synthesis.data.load_data import SignLanguagePoseDataset
from fluent_pose_synthesis.config.option import (
    add_model_args,
    add_train_args,
    add_diffusion_args,
    config_parse,
)


def train(
    config: argparse.Namespace,
    resume_path: Path,
    logger: Logger,
    tb_writer: SummaryWriter,
):
    """Main training loop for sign language pose post-editing."""
    fixseed(1024)
    np_dtype = select_platform(32)

    logger.info("Loading dataset...")
    train_dataset = SignLanguagePoseDataset(
        data_dir=config.data,
        split="train",
        fluent_frames=config.arch.clip_len,
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

    logger.info(
        f"Training Dataset includes {len(train_dataset)} samples, "
        f"with {config.arch.clip_len} fluent frames per sample."
    )

    diffusion = create_gaussian_diffusion(config)
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
        device=config.device,
    ).to(config.device)

    trainer = PoseTrainingPortal(
        config, model, diffusion, train_dataloader, logger, tb_writer
    )

    if resume_path is not None:
        try:
            trainer.load_checkpoint(str(resume_path))
        except FileNotFoundError:
            print(f"No checkpoint found at {resume_path}")
            sys.exit(1)

    trainer.run_loop()


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="### Fluent Sign Language Pose Synthesis Training ###"
    )
    parser.add_argument(
        "-n", "--name", default="debug", type=str, help="The name of this training run"
    )
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
        default="/scratch/ronli/output",
        type=str,
        help="Path to dataset folder",
    )
    parser.add_argument(
        "-r", "--resume", default=None, type=str, help="Path to latest checkpoint"
    )
    parser.add_argument(
        "-s",
        "--save",
        default="./save",
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
        config.data = Path("/scratch/ronli/output") / args.data
        config.save = Path("/scratch/ronli/save") / args.name

    # Debug mode settings
    if "debug" in args.name:
        config.trainer.workers = 1
        config.trainer.load_num = 4
        config.trainer.batch_size = 4

    # Handle existing folder
    if (
        not args.cluster
        and config.save.exists()
        and "debug" not in args.name
        and args.resume is None
    ):
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
            ckpts = [f for f in config.save.glob("weights_*.pt")]
            if ckpts:
                latest = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
                resume_path = latest

    config.save.mkdir(parents=True, exist_ok=True)

    logger = Logger(config.save / "log.txt")
    tb_writer = SummaryWriter(log_dir=config.save / "runtime")

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save config
    with open(config.save / "config.json", "w", encoding="utf-8") as f:
        f.write(str(config))

    logger.info(f"\nLaunching training with config:\n{config}")
    train(config, resume_path, logger, tb_writer)
    logger.info(f"\nTotal training time: {(time.time() - start_time) / 60:.2f} mins")


if __name__ == "__main__":
    main()
