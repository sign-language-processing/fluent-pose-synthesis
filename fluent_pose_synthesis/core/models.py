from typing import Optional
import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess


class OutputProcessMLP(nn.Module):
    """
    Output process for the Sign Language Pose Diffusion model.
    """

    def __init__(self, input_feats, latent_dim, njoints, nfeats, hidden_dim=512):  # add hidden_dim as parameter
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.hidden_dim = hidden_dim  # store hidden dimension

        # MLP layers
        self.mlp = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim), nn.SiLU(),
                                 nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.SiLU(),
                                 nn.Linear(self.hidden_dim // 2, self.input_feats))

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.mlp(output)  # use MLP instead of single linear layer
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)
        return output


class SignLanguagePoseDiffusion(nn.Module):
    """
    Sign Language Pose Diffusion model.
    """

    def __init__(self, input_feats: int, chunk_len: int, keypoints: int, dims: int, latent_dim: int = 256,
                 ff_size: int = 1024, num_layers: int = 8, num_heads: int = 4, dropout: float = 0.2,
                 ablation: Optional[str] = None, activation: str = "gelu", legacy: bool = False,
                 arch: str = "trans_enc", cond_mask_prob: float = 0, device: Optional[torch.device] = None,
                 batch_first: bool = True):
        """
        Args:
            input_feats (int): Number of input features (keypoints * dimensions).
            chunk_len (int): Length of the target fluent clip.
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
        self.chunk_len = chunk_len
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
        self.batch_first = batch_first

        # Positional encoding
        self.sequence_pos_encoder = PositionalEncoding(d_model=latent_dim, dropout=dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        # Fluent encoder: processes fluent clip using MotionProcess.
        self.fluent_encoder = MotionProcess(input_feats, latent_dim)
        # Disfluent encoder: encodes the disfluent sequence into a global context vector.
        self.disfluent_encoder = MotionProcess(input_feats, latent_dim)

        # Define sequence encoder based on chosen architecture
        if self.arch == "trans_enc":
            print(f"Initializing Transformer Encoder (batch_first={self.batch_first})")
            encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
                                                       dropout=dropout, activation=activation,
                                                       batch_first=self.batch_first)
            self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif self.arch == "trans_dec":
            print(f"Initializing Transformer Decoder (batch_first={self.batch_first})")
            decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
                                                       dropout=dropout, activation=activation,
                                                       batch_first=self.batch_first)
            self.sequence_encoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        elif self.arch == "gru":
            print("Initializing GRU Encoder (batch_first=True)")
            self.sequence_encoder = nn.GRU(latent_dim, latent_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("Please choose correct architecture [trans_enc, trans_dec, gru]")

        # Pose projection: projects latent representation back to pose space.
        # The OutputProcess returns (B, keypoints, dims, T); apply a post_transform to get (B, T, keypoints, dims)
        self.pose_projection = OutputProcessMLP(input_feats, latent_dim, keypoints, dims, hidden_dim=1024)
        self.to(self.device)

    def forward(
            self,
            fluent_clip: torch.Tensor,  # (B, K, D, T_chunk)
            disfluent_seq: torch.Tensor,  # (B, K, D, T_disfl)
            t: torch.Tensor,  # (B,)
            previous_output: Optional[torch.Tensor] = None  # (B, K, D, T_hist)
    ) -> torch.Tensor:

        # Ensure inputs are on the correct device
        fluent_clip = fluent_clip.to(self.device)
        disfluent_seq = disfluent_seq.to(self.device)
        t = t.to(self.device)
        if previous_output is not None:
            previous_output = previous_output.to(self.device)

        B = fluent_clip.shape[0]
        T_chunk = fluent_clip.shape[-1]

        # 1. Embed Timestep
        _t_emb_raw = self.embed_timestep(t)  # Expected (B, D)
        t_emb = _t_emb_raw.permute(1, 0, 2).contiguous()

        # 2. Embed Disfluent Sequence (Condition)
        _disfluent_emb_raw = self.disfluent_encoder(disfluent_seq)  # Expected (T_disfl, B, D)
        disfluent_emb = _disfluent_emb_raw.permute(1, 0, 2).contiguous()  # Expected (B, T_disfl, D)

        # 3. Embed Previous Output (History), if available
        embeddings_to_concat = [t_emb, disfluent_emb]
        if previous_output is not None and previous_output.shape[-1] > 0:
            _prev_out_emb_raw = self.fluent_encoder(previous_output)  # Expected (T_hist, B, D)
            prev_out_emb = _prev_out_emb_raw.permute(1, 0, 2).contiguous()  # Expected (B, T_hist, D)
            embeddings_to_concat.append(prev_out_emb)
        else:
            pass

        # 4. Embed Current Fluent Clip (Noisy Target 'x')
        _fluent_emb_raw = self.fluent_encoder(fluent_clip)  # Expected (T_chunk, B, D)
        fluent_emb = _fluent_emb_raw.permute(1, 0, 2).contiguous()  # Expected (B, T_chunk, D)
        embeddings_to_concat.append(fluent_emb)

        # 5. Concatenate all embeddings along the sequence dimension (T)
        xseq = torch.cat(embeddings_to_concat, dim=1)

        # 6. Apply Positional Encoding
        # Adapt based on PositionalEncoding expectation (T, B, D) vs batch_first
        if self.batch_first:
            xseq_permuted = xseq.permute(1, 0, 2).contiguous()  # (T_total, B, D)
            xseq_encoded = self.sequence_pos_encoder(xseq_permuted)
            xseq = xseq_encoded.permute(1, 0, 2)  # Back to (B, T_total, D)
        else:
            # If not batch_first, assume xseq should be (T, B, D) already
            # Need to adjust concatenation and permutations above if batch_first=False
            xseq = xseq.permute(1, 0, 2)  # Assume needs (T, B, D)
            xseq = self.sequence_pos_encoder(xseq)
        # Keep as (T, B, D) if encoder needs it

        # 7. Process through sequence encoder
        if self.arch == "trans_enc":
            x_encoded = self.sequence_encoder(xseq)
        elif self.arch == "gru":
            x_encoded, _ = self.sequence_encoder(xseq)
        elif self.arch == "trans_dec":
            memory = xseq
            tgt = xseq
            x_encoded = self.sequence_encoder(tgt=tgt, memory=memory)
        else:
            raise ValueError("Unsupported architecture")

        # 8. Extract the output corresponding to the target fluent_clip
        if self.batch_first:
            # x_encoded is (B, T_total, D), take last T_chunk frames
            x_out = x_encoded[:, -T_chunk:, :]
            # Permute to (T_chunk, B, D) for pose_projection
            x_out = x_out.permute(1, 0, 2)
        else:
            # x_encoded is (T_total, B, D), take last T_chunk frames
            x_out = x_encoded[-T_chunk:, :, :]
            # No permute needed if pose_projection expects (T, B, D)

        # 9. Project back to pose space
        output = self.pose_projection(x_out)

        return output

    def interface(
            self,
            fluent_clip: torch.Tensor,  # (B, K, D, T_chunk)
            t: torch.Tensor,  # (B,)
            y: dict[str, torch.Tensor]  # Conditions dict
    ) -> torch.Tensor:
        """
        Interface for Classifier-Free Guidance (CFG). Handles previous_output.
        """
        batch_size = fluent_clip.size(0)
        # Extract conditions
        disfluent_seq = y["input_sequence"]
        previous_output = y.get("previous_output", None)

        # Apply CFG: randomly drop the condition with probability cond_mask_prob
        keep_batch_idx = torch.rand(batch_size, device=disfluent_seq.device) < (1 - self.cond_mask_prob)
        disfluent_seq = disfluent_seq * keep_batch_idx.view((batch_size, 1, 1, 1))

        # Call the forward function
        return self.forward(fluent_clip=fluent_clip, disfluent_seq=disfluent_seq, t=t, previous_output=previous_output)
