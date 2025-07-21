import os
import sys
import torch
import pytest
from fluent_pose_synthesis.core.models import SignLanguagePoseDiffusion

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def get_dummy_batch(batch_size=2, seq_len=50, keypoints=178, dims=3):
    """
    Returns a minimal dummy batch dict like the example:
      - data: (B, seq_len, 1, keypoints, dims)
      - conditions.input_sequence: same shape
      - t: zeros (B,)
    """
    data = torch.ones(batch_size, seq_len, 1, keypoints, dims)
    conditions = {"input_sequence": torch.ones(batch_size, seq_len, 1, keypoints, dims)}
    t = torch.zeros(batch_size, dtype=torch.long)

    return data, conditions, t


@pytest.mark.parametrize("arch", ["trans_enc", "trans_dec", "gru"])
@pytest.mark.parametrize("seq_len", [50, 100])
@pytest.mark.parametrize("batch_size", [2, 4, 16])
def test_base_output_shape(arch, seq_len, batch_size):
    device = torch.device("cpu")

    model = SignLanguagePoseDiffusion(input_feats=534, chunk_len=seq_len, keypoints=178, dims=3, latent_dim=256,
                                      ff_size=256, num_layers=2, num_heads=4, dropout=0.2, arch=arch,
                                      cond_mask_prob=0.1, device=device).to(device)

    fluent_clip, conditions, t = get_dummy_batch(batch_size=batch_size, seq_len=seq_len)

    # Move tensors to device
    fluent_clip = fluent_clip.to(device)
    disfluent_seq = conditions["input_sequence"].to(device)
    t = t.to(device)

    output = model(fluent_clip, disfluent_seq, t)

    assert output.shape == (batch_size, seq_len, 178, 3), f"Arch {arch} output shape mismatch: {output.shape}"
    assert torch.isfinite(output).all(), f"Arch {arch} output contains NaN or Inf values"
    assert output.sum() != 0, f"Arch {arch} output is entirely zero"


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
