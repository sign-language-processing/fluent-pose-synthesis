import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SignLanguagePoseDiffusion(nn.Module):
    """Sign Language Pose Diffusion model."""
    def __init__(self,
                 input_feats: int,
                 clip_len: int,
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
                 device: Optional[torch.device] = None
                    ):
        """
        Args:
            input_feats (int): Number of input features (keypoints * dimensions).
            clip_len (int): Length of the target (fluent) clip.
            latent_dim (int): Dimension of the latent space.
            ff_size (int): Feed-forward network dimension.
            num_layers (int): Number of Transformer layers.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            ablation (Optional[str]): Ablation study parameter.
            activation (str): Activation function.
            legacy (bool): Legacy mode flag.
            arch (str): Architecture type: "trans_enc", "trans_dec", or "gru".
            cond_mask_prob (float): Probability to mask the condition (CFG).
            device (Optional[torch.device]): Device to run the model.
        """
        super().__init__()
        self.input_feats = input_feats
        self.clip_len = clip_len  # L_target
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

        # For time step embedding, use a fixed positional encoder
        self.time_pos_encoder = PositionalEncoding(latent_dim, dropout=dropout)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.time_pos_encoder)

        # Process fluent clip using a PoseEncoder module (batch-first)
        self.fluent_encoder = PoseEncoder(input_feats, latent_dim)
        # Encode the entire disfluent sequence (global condition) using a Transformer encoder (batch-first)
        self.disfluent_encoder = DisfluentContextEncoder(input_feats, latent_dim)

        # Sequence-level positional encoding for the concatenated sequence (batch-first)
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout=dropout)

        # Project latent representation back to pose space
        self.pose_projection = PoseProjection(input_feats, latent_dim)

        # Define the sequence encoder based on the chosen architecture (set batch_first=True)
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

    def forward(self, fluent_clip: torch.Tensor, disfluent_seq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fluent_clip (Tensor): (B, L_target, people, keypoints, dimensions) target fluent clip.
            disfluent_seq (Tensor): (B, L_condition, people, keypoints, dimensions) entire disfluent sequence.
            t (Tensor): (B,) diffusion time steps.
        """
        # Process fluent clip via fluent_encoder
        B, L_target, P, K, D = fluent_clip.shape
        fluent_clip_proc = fluent_clip.squeeze(2)  # (B, L_target, K, D)
        # Flatten K, D into K*D, keep batch-first (B, L_target, input_feats)
        fluent_clip_proc = fluent_clip_proc.reshape(B, L_target, -1)
        fluent_clip_emb = self.fluent_encoder(fluent_clip_proc)  # (B, L_target, latent_dim)

        # Process entire disfluent sequence via disfluent_encoder
        B2, L_condition, P2, K2, D2 = disfluent_seq.shape
        disfluent_proc = disfluent_seq.squeeze(2)  # (B, L_condition, K, D)
        disfluent_proc = disfluent_proc.reshape(B2, L_condition, -1)  # (B, L_condition, input_feats)
        disfluent_context = self.disfluent_encoder(disfluent_proc)  # (B, latent_dim)
        disfluent_context = disfluent_context.unsqueeze(1)  # (B, 1, latent_dim)

        # Obtain time step embedding
        t_emb = self.embed_timestep(t)  # (B, latent_dim)
        t_emb = t_emb.unsqueeze(1)  # (B, 1, latent_dim)

        # Concatenate the branches along the time dimension: time embedding, disfluent context, fluent clip
        # New shape: (B, 1 + 1 + L_target, latent_dim)
        sequence_input = torch.cat((t_emb, disfluent_context, fluent_clip_emb), dim=1)

        # Apply sequence positional encoding (batch-first)
        sequence_input = self.sequence_pos_encoder(sequence_input)
        # Process through sequence encoder (batch-first)
        x_encoded = self.sequence_encoder(sequence_input)  # (B, T, latent_dim)
        # Select the last L_target frames (corresponding to the fluent clip) for output
        x_out = x_encoded[:, -L_target:, :]  # (B, L_target, latent_dim)
        # Project latent representation back to pose space; recover (B, L_target, keypoints, dims)
        output = self.pose_projection(x_out, K, D)
        
        return output

    def interface(self, fluent_clip: torch.Tensor, t: torch.Tensor, y: dict) -> torch.Tensor:
        """
        Interface for Classifier-Free Guidance (CFG).
        It extracts the disfluent condition from y, applies random dropout on it, and calls forward to generate the final motion sequence.
        Args:
            fluent_clip (Tensor): (B, L_target, people, keypoints, dimensions) target fluent clip.
            t (Tensor): (B,) diffusion time steps.
            y (dict): Dictionary containing condition information.
        """
        B = fluent_clip.size(0)
        disfluent_seq = y["input_sequence"]
        # Apply CFG: randomly drop the condition with probability cond_mask_prob.
        keep = (torch.rand(B, device=disfluent_seq.device) < (1 - self.cond_mask_prob)).float()
        disfluent_seq = disfluent_seq * keep.view(B, 1, 1, 1, 1)

        return self.forward(fluent_clip, disfluent_seq, t)    


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for sequence (batch-first version)."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model (int): Dimensionality of the model.
            dropout (float): Dropout probability.
            max_len (int): Maximum length of the input sequence.
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)  (batch-first)
        B, T, d_model = x.shape
        # Use the first T positions from pe (shape: (1, T, d_model))
        x = x + self.pe[:, :T, :]

        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    """Embeds time steps to latent space."""
    def __init__(self, latent_dim: int, sequence_pos_encoder: nn.Module) -> None:
        """
        Args:
            latent_dim (int): Dimension of the latent space.
            sequence_pos_encoder (nn.Module): Positional encoding module with buffer 'pe' of shape (1, max_len, latent_dim).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder
        
        self.time_embedding = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: (B,)
        # Index into the positional encoding buffer; result shape: (1, B, latent_dim)
        emb = self.sequence_pos_encoder.pe[:, timesteps, :]  # (1, B, latent_dim)
        emb = emb.squeeze(0)  # (B, latent_dim)
        t_emb = self.time_embedding(emb)
        
        return t_emb


class PoseEncoder(nn.Module):
    """Maps input pose sequence to latent space (batch-first)."""
    def __init__(self, input_feats: int, latent_dim: int) -> None:
        super().__init__()
        self.pose_embedding = nn.Linear(input_feats, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, input_feats)
        out = self.pose_embedding(x)  # (B, L, latent_dim)

        return out


class DisfluentContextEncoder(nn.Module):
    """Encodes the entire disfluent sequence via a Transformer and pools to a global vector (batch-first)."""
    def __init__(self, input_feats: int, latent_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.pose_encoder = PoseEncoder(input_feats, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True  # Modified to use batch-first
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, input_feats)
        x_emb = self.pose_encoder(x)  # (B, L, latent_dim)
        x_enc = self.encoder(x_emb)     # (B, L, latent_dim)
        # Using mean pooling as global summary along time dimension
        disfluent_context = x_enc.mean(dim=1)  # (B, latent_dim)
        return disfluent_context


class PoseProjection(nn.Module):
    """Projects latent representation back to pose space."""
    def __init__(self, input_feats: int, latent_dim: int) -> None:
        super().__init__()
        self.input_feats = input_feats
        self.pose_projector = nn.Linear(latent_dim, self.input_feats)

    def forward(self, x: torch.Tensor, keypoints: int, dims: int) -> torch.Tensor:        
        # x: (B, L, latent_dim)
        out = self.pose_projector(x)  # (B, L, input_feats)
        B, L, _ = out.shape
        out = out.reshape(B, L, keypoints, dims)

        return out    


#############################
# Testing
#############################

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from load_data import SignLanguagePoseDataset
from pose_format.torch.masked.collator import zero_pad_collator


def test_model_with_dataset():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = Path("/scratch/ronli/output")
    split = "train"
    fluent_frames = 20
    limited_num = 100
    
    # Instantiate dataset and DataLoader
    dataset = SignLanguagePoseDataset(data_dir=data_dir, split=split, fluent_frames=fluent_frames, limited_num=limited_num)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, collate_fn=zero_pad_collator)
    
    # Get one batch
    batch = next(iter(dataloader))
    fluent_clip = batch["data"]          # fluent_clip: (B, fluent_frames, people, keypoints, dimensions)
    conditions = batch["conditions"]
    disfluent_seq = conditions["input_sequence"]  # disfluent_seq: (B, L_condition, people, keypoints, dimensions)
    
    # Generate random diffusion time steps for the batch
    B = fluent_clip.shape[0]
    t = torch.randint(0, 1000, (B,), device=device).long()
    
    # Initialize the model with parameters matching your dataset.
    model = SignLanguagePoseDiffusion(
        input_feats=534,  # 178 keypoints * 3 dimensions
        clip_len=fluent_clip.shape[1],
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
