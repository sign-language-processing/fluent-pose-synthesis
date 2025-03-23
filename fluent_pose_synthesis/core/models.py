import sys
import os
import torch
import torch.nn as nn
from typing import Optional, Callable

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMDM_PATH = os.path.join(BASE_DIR, "CAMDM", "PyTorch")
# Add CAMDM directory to Python path
# sys.path.append(CAMDM_PATH)
sys.path.insert(0, CAMDM_PATH)

from network.models import PositionalEncoding, TimestepEmbedder, OutputProcess, MotionProcess


class BatchFirstWrapper(nn.Module):
    """
    A wrapper class to convert a module to support batch-first input and output.
    It allows custom pre_transform and post_transform functions.
    """
    def __init__(self,
                 module: nn.Module,
                 pre_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 post_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.module = module
        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_transform is not None:
            x = self.pre_transform(x)
        else:
            if x.dim() >= 3:
                x = x.transpose(0, 1)  # Convert (B, T, ...) to (T, B, ...)
        x = self.module(x)
        if self.post_transform is not None:
            x = self.post_transform(x)
        else:
            if x.dim() >= 3:
                x = x.transpose(0, 1)  # Convert back from (T, B, ...) to (B, T, ...)
        return x

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class DisfluentContextEncoder(nn.Module):
    """
    Encodes the disfluent sequence into a global context vector using a Transformer encoder and mean pooling.
    Expected input shape: (B, T, keypoints, dims)
    """
    def __init__(self, input_feats: int, latent_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Use MotionProcess with custom dimension transforms:
        self.pose_encoder = BatchFirstWrapper(
            MotionProcess(input_feats, latent_dim),
            pre_transform=lambda x: x.permute(0, 2, 3, 1),  # (B, T, keypoints, dims) -> (B, keypoints, dims, T)
            post_transform=lambda x: x.transpose(0, 1)        # (T, B, latent_dim) -> (B, T, latent_dim)
        )
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
        # x: (B, T, keypoints, dims)
        x_emb = self.pose_encoder(x)       # (B, T, latent_dim)
        x_enc = self.encoder(x_emb)         # (B, T, latent_dim)
        disfluent_context = x_enc.mean(dim=1)  # Mean pooling over time: (B, latent_dim)
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
                 device: Optional[torch.device] = None):
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
        self.clip_len = clip_len
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

        # Positional encoding
        self.sequence_pos_encoder = BatchFirstWrapper(PositionalEncoding(d_model=latent_dim, dropout=dropout))   
        # Timestep embedder: expects input (B,) and returns (1, B, latent_dim); post_transform converts it to (B, 1, latent_dim)
        self.embed_timestep = BatchFirstWrapper(
            TimestepEmbedder(latent_dim, self.sequence_pos_encoder),
            pre_transform=lambda x: x,  # Identity transform for input (B,)
            post_transform=lambda x: x.permute(1, 0, 2)  # (1, B, latent_dim) -> (B, 1, latent_dim)
        )
        # Fluent encoder: processes fluent clip using MotionProcess.
        # Fluent clip shape: (B, T, keypoints, dims) -> pre_transform converts to (B, keypoints, dims, T)
        # and post_transform converts output from (T, B, latent_dim) to (B, T, latent_dim)
        self.fluent_encoder = BatchFirstWrapper(
            MotionProcess(input_feats, latent_dim),
            pre_transform=lambda x: x.permute(0, 2, 3, 1),
            post_transform=lambda x: x.transpose(0, 1)
        )    
        # Disfluent encoder: encodes the disfluent sequence into a global context vector.
        self.disfluent_encoder = DisfluentContextEncoder(input_feats, latent_dim)

        # Define sequence encoder based on chosen architecture
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

        # Pose projection: projects latent representation back to pose space.
        # The OutputProcess returns (B, keypoints, dims, T); apply a post_transform to get (B, T, keypoints, dims)
        self.pose_projection = BatchFirstWrapper(
            OutputProcess(input_feats, latent_dim, keypoints, dims),
            post_transform=lambda x: x.permute(0, 3, 1, 2)
        )

    def forward(self, fluent_clip: torch.Tensor, disfluent_seq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fluent_clip (torch.Tensor): (B, L_target, people, keypoints, dims) target fluent clip.
            disfluent_seq (torch.Tensor): (B, L_condition, people, keypoints, dims) disfluent sequence.
            t (torch.Tensor): (B,) diffusion time steps.
        """
        # Process fluent clip: remove the people dimension -> (B, L_target, keypoints, dims)
        fluent_clip_proc = fluent_clip.squeeze(2)
        # Get fluent clip embedding using the MotionProcess wrapper
        fluent_clip_emb = self.fluent_encoder(fluent_clip_proc)  # (B, L_target, latent_dim)

        # Process disfluent sequence: remove the people dimension -> (B, L_condition, keypoints, dims)
        disfluent_proc = disfluent_seq.squeeze(2)
        disfluent_context = self.disfluent_encoder(disfluent_proc)  # (B, latent_dim)
        disfluent_context = disfluent_context.unsqueeze(1)  # (B, 1, latent_dim)

        # Obtain time step embedding (B, 1, latent_dim)
        t_emb = self.embed_timestep(t)

        # Handle different architectures
        if self.arch == "trans_enc":
            # Concatenate time embedding, disfluent context, and fluent clip embedding along the time dimension
            sequence_input = torch.cat((t_emb, disfluent_context, fluent_clip_emb), dim=1)  # (B, 2 + L_target, latent_dim)
            # Apply positional encoding
            sequence_input = self.sequence_pos_encoder(sequence_input)
            # Process sequence using the Transformer encoder
            x_encoded = self.sequence_encoder(sequence_input)  # (B, T, latent_dim)
            # Select the last L_target frames corresponding to the fluent clip
            x_out = x_encoded[:, -fluent_clip_emb.size(1):, :]
        elif self.arch == "trans_dec":
            # For Transformer Decoder, use fluent_clip_emb as target and combine t_emb and disfluent_context as memory.
            tgt = self.sequence_pos_encoder(fluent_clip_emb)  # (B, L_target, latent_dim)
            memory = self.sequence_pos_encoder(torch.cat((t_emb, disfluent_context), dim=1))  # (B, 2, latent_dim)
            x_encoded = self.sequence_encoder(tgt, memory=memory)  # (B, L_target, latent_dim)
            x_out = x_encoded
        else:  # "gru"
            sequence_input = torch.cat((t_emb, disfluent_context, fluent_clip_emb), dim=1)  # (B, 2 + L_target, latent_dim)
            sequence_input = self.sequence_pos_encoder(sequence_input)
            x_encoded, _ = self.sequence_encoder(sequence_input)
            x_out = x_encoded[:, -fluent_clip_emb.size(1):, :]

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
        B = fluent_clip.size(0)
        disfluent_seq = y["input_sequence"]
        # Apply CFG: randomly drop the condition with probability cond_mask_prob
        keep = (torch.rand(B, device=disfluent_seq.device) < (1 - self.cond_mask_prob)).float()
        disfluent_seq = disfluent_seq * keep.view(B, 1, 1, 1, 1)
        return self.forward(fluent_clip, disfluent_seq, t)