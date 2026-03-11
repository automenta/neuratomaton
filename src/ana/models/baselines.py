import torch
import torch.nn as nn
import math
from .config import ANAConfig

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerBaseline(nn.Module):
    """
    Standard Transformer (GPT-style causal decoder) baseline.
    Uses nn.TransformerEncoder with causal masking.
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, dropout=config.dropout, max_len=config.max_position)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=4, # Heuristic: d_model // 16 or fixed 4
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config.num_layers)

        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def generate_square_subsequent_mask(self, sz: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, input_ids, return_info=False, force_prob=0.0):
        # input_ids: [Batch, Seq]
        x = self.embedding(input_ids) # [Batch, Seq, D]
        x = x.permute(1, 0, 2) # [Seq, Batch, D] for PosEnc
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2) # Back to [Batch, Seq, D] for Transformer (batch_first=True)

        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        output = self.transformer_encoder(x, mask=mask, is_causal=True)

        output = self.norm(output)
        logits = self.output_head(output)

        # Transformer doesn't have "internal states" to visualize in the same way as ANA
        return logits, {}

class LSTMBaseline(nn.Module):
    """
    Standard LSTM baseline.
    """
    def __init__(self, config: ANAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )

        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids, return_info=False, force_prob=0.0):
        # input_ids: [Batch, Seq]
        x = self.embedding(input_ids)

        # LSTM
        output, (hn, cn) = self.lstm(x)

        output = self.norm(output)
        logits = self.output_head(output)

        return logits, {}
