import sys
import os
import time
import shutil
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fluent_pose_synthesis.core.models import SignLanguagePoseDiffusion
from fluent_pose_synthesis.core.training import PoseTrainingPortal
from fluent_pose_synthesis.data.load_data import SignLanguagePoseDataset
from fluent_pose_synthesis.config.option import add_model_args, add_train_args, add_diffusion_args, config_parse ##to do


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMDM_PATH = os.path.join(BASE_DIR, "CAMDM", "PyTorch")
sys.path.append(CAMDM_PATH)

import utils.common as common
from utils.logger import Logger
from diffusion.create_diffusion import create_gaussian_diffusion

from pose_format.torch.masked.collator import zero_pad_collator


def train(config: argparse.Namespace, resume: Optional[str], logger: Logger, tb_writer: SummaryWriter):
    """
    Main training loop for sign language pose post-editing.
    Args:
        config (argparse.Namespace): Parsed configuration object.
        resume (Optional[str]): Path to checkpoint file to resume from.
        logger (Logger): Logger instance for saving logs.
        tb_writer (SummaryWriter): TensorBoard writer for visualization.
    """
    common.fixseed(1024)
    np_dtype = common.select_platform(32)

    print("Loading dataset...")
    train_dataset = SignLanguagePoseDataset(
        data_dir=Path(config.data),
        split="train",
        fluent_frames=config.arch.clip_len,
        dtype=np_dtype,
        limited_num=config.trainer.load_num
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=zero_pad_collator
    )

    logger.info(
        f"\nTraining Dataset includes {len(train_dataset)} samples, "
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
        device=config.device
    ).to(config.device)

    trainer = PoseTrainingPortal(config, model, diffusion, train_dataloader, logger, tb_writer)

    if resume is not None:
        try:
            trainer.load_checkpoint(resume)
        except FileNotFoundError:
            print(f"No checkpoint found at {resume}")
            exit()

    trainer.run_loop()


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='### Fluent Sign Language Pose Synthesis Training ###')

    parser.add_argument('-n', '--name', default='debug', type=str, help='The name of this training run')
    parser.add_argument('-c', '--config', default='./fluent_pose_synthesis/config/default.json', 
                        type=str, help='Path to config file')
    parser.add_argument('-i', '--data', default='/scratch/ronli/output', type=str, help='Path to dataset folder')
    parser.add_argument('-r', '--resume', default=None, type=str, help='Path to latest checkpoint')
    parser.add_argument('-s', '--save', default='./save', type=str, help='Directory to save model and logs')
    parser.add_argument('--cluster', action='store_true', help='Enable cluster mode')

    add_model_args(parser)
    add_diffusion_args(parser)
    add_train_args(parser)

    args = parser.parse_args()

    if args.cluster:
        # Override paths when running on a GPU cluster
        args.data = '/scratch/ronli/output' + os.path.basename(args.data)
        args.save = '/scratch/ronli/save' + args.name

    if args.config:
        config = config_parse(args)
    else:
        raise AssertionError("You must provide a config file using -c <config_path>")

    if 'debug' in args.name:
        config.trainer.workers = 1
        config.trainer.load_num = 5
        config.trainer.batch_size = 5

    # If not resuming from checkpoint, ask before overwriting old save folder
    if not args.cluster and os.path.exists(config.save) and 'debug' not in args.name and args.resume is None:
        allow_cover = input('Model folder exists. Overwrite? (Y/N)').lower()
        if allow_cover == 'n':
            exit()
        else:
            shutil.rmtree(config.save, ignore_errors=True)
    else:
        # Auto-resume for cluster
        if os.path.exists(config.save):
            best_ckpt = os.path.join(config.save, 'best.pt')
            if os.path.exists(best_ckpt):
                args.resume = best_ckpt
            else:
                existing = [f for f in os.listdir(config.save) if 'weights_' in f]
                if existing:
                    epoch_nums = [int(f.split('_')[1].split('.')[0]) for f in existing]
                    latest = max(epoch_nums)
                    args.resume = os.path.join(config.save, f'weights_{latest}.pt')

    os.makedirs(config.save, exist_ok=True)

    logger = Logger(os.path.join(config.save, 'log.txt'))
    tb_writer = SummaryWriter(log_dir=os.path.join(config.save, 'runtime'))

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save the config for reference
    with open('%s/config.json' % config.save, 'w') as f:
        f.write(str(config))
        f.close()

    logger.info('\nLaunching training with config:\n%s' % config)
    train(config, args.resume, logger, tb_writer)
    logger.info('\nTotal training time: %.2f mins' % ((time.time() - start_time) / 60))
