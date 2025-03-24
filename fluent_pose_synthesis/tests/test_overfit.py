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
        return {"data": torch.tensor(0)}  # minimal dummy data


def get_toy_batch(batch_size=2, seq_len=40, keypoints=178):
    base_linear = torch.linspace(0, 1, seq_len * keypoints * 3).reshape(seq_len, 1, keypoints, 3)
    base_sine = torch.sin(torch.linspace(0, 4 * np.pi, seq_len)).unsqueeze(1).unsqueeze(2)  # [T, 1, 1]
    base_sine = base_sine.expand(seq_len, 1, keypoints).unsqueeze(-1)  # [T, 1, K, 1]
    base_sine = base_sine.repeat(1, 1, 1, 3)  # [T, 1, K, 3] -> same sin pattern for x/y/z

    pose_data = []
    target_mask = []
    input_sequence = []

    for i in range(batch_size):
        if i == 0:
            sample = base_linear + torch.randn_like(base_linear) * 0.01
        elif i == 1:
            sample = 0.5 + 0.2 * base_sine + torch.randn_like(base_sine) * 0.01  # Sine pattern

        pose_data.append(sample)

        # Create distinct masks
        mask = torch.ones_like(sample)
        mask[i::2] = 0
        target_mask.append(mask)

        # Input sequence: one is 0, one is 1
        input_seq = torch.ones(1, keypoints, 3) * i
        input_sequence.append(input_seq)

    pose_data = torch.stack(pose_data, dim=0)
    target_mask = torch.stack(target_mask, dim=0)
    input_sequence = torch.stack(input_sequence, dim=0)

    batch = {
        "data": pose_data.clone(),
        "conditions": {
            "target_mask": target_mask,
            "input_sequence": input_sequence,
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
            epoch=100,
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

    batch = get_toy_batch(
        batch_size=config.trainer.batch_size,
        seq_len=config.arch.clip_len,
        keypoints=config.arch.keypoints
    )

    # Move batch to device
    batch = {
        k: (v.to(config.device) if isinstance(v, torch.Tensor) else {
            kk: vv.to(config.device) for kk, vv in v.items()
        }) for k, v in batch.items()
    }

    # Create diffusion and model
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

    trainer = PoseTrainingPortal(
        config, model, diffusion,
        dataloader=dummy_dataloader,
        logger=None, tb_writer=None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.trainer.lr)

    print("Start overfitting...")
    losses = []

    for step in range(config.trainer.epoch):
        t, weights = trainer.schedule_sampler.sample(config.trainer.batch_size, config.device)
        output, loss_dict = trainer.diffuse(
            batch["data"], t, batch["conditions"], return_loss=True
        )
        loss = (loss_dict["loss"] * weights).mean()
        losses.append(loss.item())
        print(f"[Step {step}] Loss: {loss.item():.6f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert losses[-1] < 1e-3, "Final loss is too high. Model failed to overfit the toy batch."
    plot_loss_curve(losses, save_path="overfit_loss_curve.png")

    # Check model output differences
    model.eval()
    with torch.no_grad():
        t, _ = trainer.schedule_sampler.sample(1, config.device)

        out1 = model(
            fluent_clip=batch["data"][0:1],
            disfluent_seq=batch["conditions"]["input_sequence"][0:1],
            t=t
        )

        out2 = model(
            fluent_clip=batch["data"][1:2],
            disfluent_seq=batch["conditions"]["input_sequence"][1:2],
            t=t
        )

        print(f"out1 shape: {out1.shape}, out2 shape: {out2.shape}")
        expected_shape = (1, config.arch.clip_len, config.arch.keypoints, config.arch.dims)
        assert out1.shape == out2.shape == expected_shape, f"Unexpected output shape, expected {expected_shape}"

        # Compute multiple metrics to assess output difference
        l2_diff = torch.norm(out1 - out2).item()
        avg_kpt_error = compute_average_keypoint_error(out1, out2)
        cosine_dist = compute_cosine_distance(out1, out2)

        print(f"Output L2 norm diff: {l2_diff:.6f}")
        print(f"Average keypoint error: {avg_kpt_error:.6f}")
        print(f"Cosine distance: {cosine_dist:.6f}")

        # Assert based on multiple metrics
        assert avg_kpt_error > 0.01 or cosine_dist > 0.01, "Outputs are too similar. Possible collapse."

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


def compute_average_keypoint_error(pose1, pose2):
    """
    Computes average L2 distance per keypoint per frame.
    """
    assert pose1.shape == pose2.shape, "Shape mismatch"
    diff = torch.norm(pose1 - pose2, dim=-1)    # [B, T, K]
    return diff.mean().item()                   # scalar


def compute_cosine_distance(pose1, pose2):
    """
    Computes cosine distance between flattened pose vectors.
    Returns a value in [0, 2], where 0 = identical, 1 = orthogonal.
    """
    v1 = pose1.flatten()
    v2 = pose2.flatten()
    cos = torch.nn.functional.cosine_similarity(v1, v2, dim=0)
    return 1 - cos.item()


if __name__ == "__main__":
    test_overfit_toy_batch()