from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(
        self,
        mlp_dim: int,
        dtype: torch.dtype = torch.float32,
        out_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dtype = dtype
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

        # First linear layer
        self.dense1 = nn.Linear(
            mlp_dim if out_dim is None else out_dim, mlp_dim, dtype=self.dtype
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second linear layer
        self.dense2 = nn.Linear(
            mlp_dim, mlp_dim if out_dim is None else out_dim, dtype=self.dtype
        )
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dense1(inputs)
        x = F.gelu(x, approximate="tanh")
        x = self.dropout1(x)

        output = self.dense2(x)
        output = self.dropout2(output)

        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer."""

    def __init__(
        self,
        mlp_dim: int,
        num_heads: int,
        d_model: int = 768,
        dtype: torch.dtype = torch.float32,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.d_model = d_model
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        # Layer normalization
        self.norm1 = nn.LayerNorm(self.d_model, eps=1e-6, dtype=self.dtype)
        self.norm2 = nn.LayerNorm(self.d_model, eps=1e-6, dtype=self.dtype)

        # MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=num_heads,
            dropout=attention_dropout_rate,
            bias=True,
            batch_first=True,
        )

        # MLP block
        self.mlp = MLPBlock(
            mlp_dim=mlp_dim,
            dtype=dtype,
            out_dim=self.d_model,
            dropout_rate=dropout_rate,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert inputs.dim() == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"

        # Attention block
        x = self.norm1(inputs)

        # # TODO (lilkm): I need to check this
        # # Process attention mask
        # if attention_mask is not None:
        #     # Attention_mask comes in as (batch, 1, seq, seq)
        #     # PyTorch MultiheadAttention expects (batch * num_heads, seq, seq) or (seq, seq)
        #     # We'll use the simpler (seq, seq) format by taking the first batch
        #     # if attention_mask.dim() == 4:
        #     #     # Take the first batch and squeeze out the head dimension
        #     #     attention_mask = attention_mask[0, 0]  # (seq, seq)

        # Convert boolean mask to additive mask (True -> 0, False -> -inf)
        if attention_mask.dtype == torch.bool:
            attention_mask = (
                attention_mask.float()
                .masked_fill(~attention_mask, float("-inf"))
                .masked_fill(attention_mask, 0.0)
            )

            # batch_size, seq_len = attention_mask.shape[0], attention_mask.shape[2]

            # # attention_mask = attention_mask.unsqueeze(1)
            # attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1)
            # attention_mask = attention_mask.view(batch_size * self.num_heads, seq_len, seq_len)

        # Apply attention
        x, _ = self.attention(x, x, x, attn_mask=attention_mask, need_weights=False)
        x = self.dropout(x)
        x = x + inputs

        # MLP block
        y = self.norm2(x)
        y = self.mlp(y)

        return x + y


class Transformer(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        num_layers: int,
        mlp_dim: int,
        num_attention_heads: int,
        d_model: int = 768,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        add_position_embedding: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.num_attention_heads = num_attention_heads
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.add_position_embedding = add_position_embedding

        # Encoder blocks - initialize with d_model
        self.encoder_blocks = nn.ModuleList(
            [
                Encoder1DBlock(
                    mlp_dim=mlp_dim,
                    num_heads=num_attention_heads,
                    d_model=d_model,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.encoder_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x.dim() == 3, f"Expected (batch, len, emb) got {x.shape}"

        # Apply encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, attention_mask)

        # Final layer norm
        encoded = self.encoder_norm(x)

        return encoded
