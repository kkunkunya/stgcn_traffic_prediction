import torch
import torch.nn as nn
from typing import Optional

from stgcn_traffic_prediction.models.transformer import make_model as make_transformer_model


class _TemporalBase(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.input_proj = nn.Linear(src_vocab, d_model)

    def _shape_output(self, features: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # features: (bs, N, T, d_model) or (bs, N, d_model)
        # tgt: (bs, N, tgt_T, tgt_vocab)
        bs, N = tgt.shape[0], tgt.shape[1]
        tgt_T, tgt_vocab = tgt.shape[2], tgt.shape[3]

        if features.dim() == 4:
            # (bs, N, T, d_model)
            T = features.shape[2]
            if tgt_T == T and tgt_vocab == 1:
                # map per-timestep to 1-dim
                out = self.pred_head_time(features)  # (bs, N, T, 1)
                return out
            elif tgt_T == 1:
                # pool over time and map to vocab
                pooled = features[:, :, -1, :]  # (bs, N, d_model)
                out = self.pred_head_vocab(pooled).unsqueeze(2)  # (bs, N, 1, tgt_vocab)
                return out
            else:
                # fallback: repeat last
                pooled = features[:, :, -1, :]
                out = self.pred_head_vocab(pooled).unsqueeze(2)
                return out
        else:
            # (bs, N, d_model)
            out = self.pred_head_vocab(features).unsqueeze(2)
            return out


class LSTMTemporal(_TemporalBase):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int, num_layers: int, dropout: float = 0.1):
        super().__init__(src_vocab, tgt_vocab, d_model, num_layers, dropout)
        self.rnn = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.pred_head_time = nn.Linear(d_model, 1)
        self.pred_head_vocab = nn.Linear(d_model, tgt_vocab)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # src: (bs, N, src_T, src_vocab)
        bs, N, T, _ = src.shape
        x = self.input_proj(src)  # (bs, N, T, d_model)
        x = self.dropout(x)
        x = x.reshape(bs * N, T, self.d_model)
        y, _ = self.rnn(x)  # (bs*N, T, d_model)
        y = y.reshape(bs, N, T, self.d_model)
        return self._shape_output(y, tgt)


class GRUTemporal(_TemporalBase):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int, num_layers: int, dropout: float = 0.1):
        super().__init__(src_vocab, tgt_vocab, d_model, num_layers, dropout)
        self.rnn = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.pred_head_time = nn.Linear(d_model, 1)
        self.pred_head_vocab = nn.Linear(d_model, tgt_vocab)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs, N, T, _ = src.shape
        x = self.input_proj(src)
        x = self.dropout(x)
        x = x.reshape(bs * N, T, self.d_model)
        y, _ = self.rnn(x)
        y = y.reshape(bs, N, T, self.d_model)
        return self._shape_output(y, tgt)


class DCNNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        y = self.dropout(y)
        # residual
        if y.shape == x.shape:
            y = y + x
        return y


class DCNNTemporal(_TemporalBase):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int, num_layers: int, dropout: float = 0.1):
        super().__init__(src_vocab, tgt_vocab, d_model, num_layers, dropout)
        dilations = [2 ** i for i in range(max(1, num_layers))]
        self.proj_in = nn.Linear(self.d_model, self.d_model)
        self.blocks = nn.ModuleList([DCNNBlock(self.d_model, d, dropout) for d in dilations])
        self.pred_head_time = nn.Linear(self.d_model, 1)
        self.pred_head_vocab = nn.Linear(self.d_model, tgt_vocab)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs, N, T, _ = src.shape
        x = self.input_proj(src)  # (bs, N, T, d_model)
        x = self.dropout(x)
        x = self.proj_in(x)
        x = x.reshape(bs * N, T, self.d_model).transpose(1, 2)  # (bs*N, d_model, T)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2).reshape(bs, N, T, self.d_model)
        return self._shape_output(x, tgt)


def make_temporal_model(arch: str, src_vocab: int, tgt_vocab: int, N: int, d_model: int,
                        d_ff: int = 64, h: int = 8, dropout: float = 0.1):
    arch = (arch or 'transformer').lower()
    if arch == 'transformer':
        return make_transformer_model(src_vocab, tgt_vocab, N, d_model, d_ff=d_ff, h=h, dropout=dropout, spatial=False)
    elif arch == 'lstm':
        return LSTMTemporal(src_vocab, tgt_vocab, d_model, N, dropout)
    elif arch == 'gru':
        return GRUTemporal(src_vocab, tgt_vocab, d_model, N, dropout)
    elif arch in ('dcnn', 'tcn'):
        return DCNNTemporal(src_vocab, tgt_vocab, d_model, N, dropout)
    else:
        raise ValueError(f'Unknown temporal arch: {arch}')


