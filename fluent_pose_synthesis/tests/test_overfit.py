import os
import sys
import random
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from fluent_pose_synthesis.core.models import SignLanguagePoseDiffusion
from fluent_pose_synthesis.core.training import PoseTrainingPortal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMDM_PATH = os.path.join(BASE_DIR, "CAMDM", "PyTorch")
sys.path.append(CAMDM_PATH)

from diffusion.create_diffusion import create_gaussian_diffusion

# Ensure root path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


class DummyDataset(Dataset):
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return {"data": torch.tensor(0)}  # minimal stub


def get_toy_batch(batch_size=2, seq_len=20, keypoints=178):
    pose_data = torch.linspace(0, 1, seq_len * keypoints * 3).reshape(1, seq_len, 1, keypoints, 3)
    pose_data = pose_data.expand(batch_size, -1, -1, -1, -1).contiguous()

    batch = {
        "data": pose_data.clone(),
        "conditions": {
            "target_mask": torch.ones(batch_size, seq_len, 1, keypoints, 3),
            "input_sequence": torch.zeros(batch_size, 1, 1, keypoints, 3),
        }
    }
    return batch


def create_minimal_config(device="cpu"):
    return Namespace(
        device=torch.device(device),
        save="./test_overfit_output",
        data="./dummy",
        trainer=Namespace(
            use_loss_mse=True,
            use_loss_vel=True,
            use_loss_3d=True,
            workers=0,
            batch_size=2,
            cond_mask_prob=0.15,
            load_num=1,
            lr=1e-3,
            epoch=50,
            lr_anneal_steps=0,
            weight_decay=0,
            ema=False,
            save_freq=5
        ),
        arch=Namespace(
            keypoints=178,
            dims=3,
            clip_len=20,
            latent_dim=32,
            ff_size=64,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            decoder="trans_enc",
            ablation=None,
            activation="gelu",
            legacy=False
        ),
        diff=Namespace(
            noise_schedule="cosine",
            diffusion_steps=4,
            sigma_small=True
        )
    )


def test_overfit_toy_batch():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = create_minimal_config()
    dummy_dataloader = DataLoader(DummyDataset())

    batch = get_toy_batch(batch_size=config.trainer.batch_size,
                          seq_len=config.arch.clip_len,
                          keypoints=config.arch.keypoints)

    batch = {k: (v.to(config.device) if isinstance(v, torch.Tensor) else {
        kk: vv.to(config.device) for kk, vv in v.items()
    }) for k, v in batch.items()}

    diffusion = create_gaussian_diffusion(config)

    model = SignLanguagePoseDiffusion(
        input_feats=config.arch.keypoints * config.arch.dims,
        clip_len=config.arch.clip_len,
        keypoints=config.arch.keypoints,
        dims=config.arch.dims,
        latent_dim=config.arch.latent_dim,
        ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers,
        num_heads=config.arch.num_heads,
        dropout=config.arch.dropout,
        arch=config.arch.decoder,
        cond_mask_prob=config.trainer.cond_mask_prob,
        device=config.device
    ).to(config.device)

    trainer = PoseTrainingPortal(config, model, diffusion, dataloader=dummy_dataloader, logger=None, tb_writer=None)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.trainer.lr)

    print("Start overfitting...")
    losses = []
    for i in range(config.trainer.epoch):
        t, weights = trainer.schedule_sampler.sample(config.trainer.batch_size, config.device)
        output, loss_dict = trainer.diffuse(batch["data"], t, batch["conditions"], return_loss=True)
        loss = (loss_dict["loss"] * weights).mean()
        losses.append(loss.item())
        print(f"[Step {i}] Loss: {loss.item():.6f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert losses[-1] < 1e-3, "Final loss is too high. Model failed to overfit the toy batch."
    plot_loss_curve(losses, save_path="overfit_loss_curve.png")
    print("Overfitting test passed.")


def plot_loss_curve(losses, save_path="loss_curve.png"):
    plt.figure()
    plt.plot(losses, label='Loss')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Overfitting Loss Curve")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved loss plot to {save_path}")


if __name__ == "__main__":
    test_overfit_toy_batch()
