import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from CAMDM.PyTorch.network.models import PositionalEncoding, TimestepEmbedder, OutputProcess, MotionProcess


class BatchFirstPositionalEncoding(nn.Module):
    """
    Wrapper for CAMDM's PositionalEncoding to support batch-first input.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.pe = self.pos_encoder.pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x.transpose(0, 1)           # Convert to (T, B, d_model)
        x = self.pos_encoder(x)         # Apply CAMDM's positional encoding
        x = x.transpose(0, 1)           # Convert back to (B, T, d_model)
        return x


class BatchFirstTimestepEmbedder(nn.Module):
    """
    Wrapper for CAMDM's TimestepEmbedder to support batch-first output.
    """
    def __init__(self, latent_dim: int, sequence_pos_encoder: nn.Module) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedder(latent_dim, sequence_pos_encoder)
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: (B,)
        t_emb = self.timestep_embedder(timesteps)  # Original shape: (1, B, latent_dim)
        t_emb = t_emb.permute(1, 0, 2)               # Convert to (B, 1, latent_dim)
        return t_emb


class BatchFirstOutputProcess(nn.Module):
    """
    Wrapper for CAMDM's OutputProcess to support batch-first input.
    """
    def __init__(self, input_feats: int, latent_dim: int, keypoints: int, dims: int) -> None:
        super().__init__()
        self.output_process = OutputProcess(input_feats, latent_dim, keypoints, dims)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, latent_dim)
        x = x.transpose(0, 1)           # Convert to (T, B, latent_dim)
        out = self.output_process(x)    # Output shape: (B, keypoints, dims, T)
        out = out.permute(0, 3, 1, 2)     # Convert to (B, T, keypoints, dims)
        return out


class BatchFirstMotionProcess(nn.Module):
    """
    Wrapper for CAMDM's MotionProcess to support batch-first input."""
    def __init__(self, input_feats: int, latent_dim: int) -> None:
        super().__init__()
        self.motion_process = MotionProcess(input_feats, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, keypoints, dims)
        B, T, keypoints, dims = x.shape
        # Convert to (B, keypoints, dims, T)
        x = x.permute(0, 2, 3, 1)
        # Call CAMDM's MotionProcess which returns shape (T, B, latent_dim)
        out = self.motion_process(x)
        out = out.transpose(0, 1)     # Convert to (B, T, latent_dim)
        return out


class DisfluentContextEncoder(nn.Module):
    """
    Encodes the disfluent sequence to a global context vector using a Transformer and mean pooling.
    Input shape: (B, T, input_feats)
    """
    def __init__(self, input_feats: int, latent_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.pose_encoder = BatchFirstMotionProcess(input_feats, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_feats)
        x_emb = self.pose_encoder(x)  # (B, T, latent_dim)
        x_enc = self.encoder(x_emb)     # (B, T, latent_dim)
        # Mean pooling over time dimension
        disfluent_context = x_enc.mean(dim=1)  # (B, latent_dim)
        return disfluent_context


class SignLanguagePoseDiffusion(nn.Module):
    """
    Sign Language Pose Diffusion model.
    """
    def __init__(self,
                 input_feats: int,
                 clip_len: int,
                 keypoints: int,
                 dims: int,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 ablation: Optional[str] = None,
                 activation: str = "gelu",
                 legacy: bool = False,
                 arch: str = "trans_enc",
                 cond_mask_prob: float = 0,
                 device: Optional[torch.device] = None) -> None:
        """
        Args:
            input_feats (int): Number of input features (keypoints * dimensions).
            clip_len (int): Length of the target fluent clip.
            keypoints (int): Number of keypoints.
            dims (int): Number of dimensions per keypoint.
            latent_dim (int): Dimension of the latent space.
            ff_size (int): Feed-forward network size.
            num_layers (int): Number of Transformer layers.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            ablation (Optional[str]): Ablation study parameter.
            activation (str): Activation function.
            legacy (bool): Legacy flag.
            arch (str): Architecture type: "trans_enc", "trans_dec", or "gru".
            cond_mask_prob (float): Condition mask probability for CFG.
            device (Optional[torch.device]): Device to run the model.
        """
        super().__init__()
        self.input_feats = input_feats
        self.clip_len = clip_len  # L_target
        self.keypoints = keypoints
        self.dims = dims
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.legacy = legacy
        self.arch = arch
        self.cond_mask_prob = cond_mask_prob
        self.device = device

        # Batch-first positional encoding using the CAMDM module wrapper
        self.sequence_pos_encoder = BatchFirstPositionalEncoding(latent_dim, dropout=dropout)
        # Batch-first timestep embedding
        self.embed_timestep = BatchFirstTimestepEmbedder(latent_dim, self.sequence_pos_encoder)
        # Process fluent clip using the MotionProcess wrapper
        self.fluent_encoder = BatchFirstMotionProcess(input_feats, latent_dim)
        # Encode the entire disfluent sequence using a Transformer encoder
        self.disfluent_encoder = DisfluentContextEncoder(input_feats, latent_dim)

        # Define the sequence encoder based on the chosen architecture (batch-first)
        if self.arch == "trans_enc":
            print("Initializing Transformer Encoder")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=True
            )
            self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif self.arch == "trans_dec":
            print("Initializing Transformer Decoder")
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=True
            )
            self.sequence_encoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        elif self.arch == "gru":
            print("Initializing GRU Encoder")
            self.sequence_encoder = nn.GRU(latent_dim, latent_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("Please choose correct architecture [trans_enc, trans_dec, gru]")

        # Project latent representation back to pose space using the OutputProcess wrapper (as PoseProjection)
        self.pose_projection = BatchFirstOutputProcess(input_feats, latent_dim, keypoints, dims)

    def forward(self, fluent_clip: torch.Tensor, disfluent_seq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fluent_clip (torch.Tensor): (B, L_target, people, keypoints, dims) target fluent clip.
            disfluent_seq (torch.Tensor): (B, L_condition, people, keypoints, dims) disfluent sequence.
            t (torch.Tensor): (B,) diffusion time steps.
        """
        # Process fluent clip: remove the people dimension -> (B, L_target, keypoints, dims)
        B, L_target, P, K, D = fluent_clip.shape
        fluent_clip_proc = fluent_clip.squeeze(2)
        # Obtain fluent clip embedding using the MotionProcess wrapper (batch-first)
        fluent_clip_emb = self.fluent_encoder(fluent_clip_proc)  # (B, L_target, latent_dim)

        # Process disfluent sequence: remove people dimension and flatten keypoints and dims
        B2, L_condition, P2, K2, D2 = disfluent_seq.shape
        disfluent_proc = disfluent_seq.squeeze(2)  # (B, L_condition, keypoints, dims)
        disfluent_context = self.disfluent_encoder(disfluent_proc)  # (B, latent_dim)
        disfluent_context = disfluent_context.unsqueeze(1)  # (B, 1, latent_dim)

        # Obtain time step embedding (B, 1, latent_dim)
        t_emb = self.embed_timestep(t)  # (B, 1, latent_dim)

        # Handle different architectures
        if self.arch == "trans_enc":
            # Concatenate time embedding, disfluent context, and fluent clip embedding along time dimension
            sequence_input = torch.cat((t_emb, disfluent_context, fluent_clip_emb), dim=1)  # (B, 1+1+L_target, latent_dim)
            # Apply positional encoding to the entire sequence
            sequence_input = self.sequence_pos_encoder(sequence_input)
            # Process sequence using the chosen encoder
            x_encoded = self.sequence_encoder(sequence_input)  # (B, T, latent_dim)
            # Select the last L_target frames corresponding to the fluent clip
            x_out = x_encoded[:, -L_target:, :]  # (B, L_target, latent_dim)
        elif self.arch == "trans_dec":
            # Use fluent_clip_emb as target (tgt) and combine t_emb and disfluent_context as memory.
            tgt = fluent_clip_emb  # (B, L_target, latent_dim)
            memory = torch.cat((t_emb, disfluent_context), dim=1)  # (B, 2, latent_dim)
            # Apply positional encoding
            tgt = self.sequence_pos_encoder(tgt)
            memory = self.sequence_pos_encoder(memory)
            x_encoded = self.sequence_encoder(tgt, memory=memory)  # (B, L_target, latent_dim)
            x_out = x_encoded
        else:   # "gru"
            # Concatenate time embedding, disfluent context, and fluent clip embedding
            sequence_input = torch.cat((t_emb, disfluent_context, fluent_clip_emb), dim=1)  # (B, 1+1+L_target, latent_dim)
            sequence_input = self.sequence_pos_encoder(sequence_input)
            # GRU returns (output, hidden); we only need output
            x_encoded, _ = self.sequence_encoder(sequence_input)
            x_out = x_encoded[:, -L_target:, :]  # (B, L_target, latent_dim)
   
        # Project latent representation back to pose space
        output = self.pose_projection(x_out)  # (B, L_target, keypoints, dims)
        
        return output

    def interface(self, fluent_clip: torch.Tensor, t: torch.Tensor, y: dict) -> torch.Tensor:
        """
        Interface for Classifier-Free Guidance (CFG).
        Args:
            fluent_clip (torch.Tensor): (B, L_target, people, keypoints, dims) target fluent clip.
            t (torch.Tensor): (B,) diffusion time steps.
            y (Dict): Dictionary containing condition information.
        """
        B: int = fluent_clip.size(0)
        disfluent_seq = y["input_sequence"]
        # Apply CFG: randomly drop the condition with probability cond_mask_prob
        keep = (torch.rand(B, device=disfluent_seq.device) < (1 - self.cond_mask_prob)).float()
        disfluent_seq = disfluent_seq * keep.view(B, 1, 1, 1, 1)
        return self.forward(fluent_clip, disfluent_seq, t)


#############################
# Testing
#############################

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from load_data import SignLanguagePoseDataset
from pose_format.torch.masked.collator import zero_pad_collator

def test_model_with_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    data_dir = Path("/scratch/ronli/output")
    split = "train"
    fluent_frames = 50
    limited_num = 100
    
    # Instantiate dataset and DataLoader
    dataset = SignLanguagePoseDataset(data_dir=data_dir, split=split, fluent_frames=fluent_frames, limited_num=limited_num)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, collate_fn=zero_pad_collator)
    
    # Get one batch
    batch = next(iter(dataloader))
    fluent_clip = batch["data"]          # fluent_clip: (B, fluent_frames, people, keypoints, dimensions)
    conditions = batch["conditions"]
    disfluent_seq = conditions["input_sequence"]  # disfluent_seq: (B, L_condition, people, keypoints, dimensions)

    print("Fluent clip shape:", fluent_clip.shape)
    print("Disfluent sequence shape:", disfluent_seq.shape)
    
    # Generate random diffusion time steps for the batch
    B = fluent_clip.shape[0]
    t = torch.randint(0, 1000, (B,), device=device).long()
    
    model = SignLanguagePoseDiffusion(
        input_feats=534,  # 178 keypoints * 3 dimensions
        clip_len=fluent_clip.shape[1],
        keypoints=178,
        dims=3,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.2,
        arch="trans_enc",
        cond_mask_prob=0.1,
        device=device
    ).to(device)
    
    # Run the model's forward pass
    output = model(fluent_clip.to(device), disfluent_seq.to(device), t)
    print("Model output shape (via forward):", output.shape)
    
    # Also test the interface (CFG) function
    y = {"input_sequence": disfluent_seq.to(device)}
    output_cfg = model.interface(fluent_clip.to(device), t, y)
    print("Model output shape (via interface):", output_cfg.shape)


if __name__ == "__main__":
    test_model_with_dataset()