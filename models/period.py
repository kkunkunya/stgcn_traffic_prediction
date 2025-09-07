import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math
from stgcn_traffic_prediction.models.transformer import make_model
from stgcn_traffic_prediction.models.temporal_modules import make_temporal_model


class period(nn.Module):
    def __init__(self, close_size, N, model_d, temporal_arch: str = 'transformer'):
        super(period, self).__init__()
        # 保持默认 transformer 行为
        self.p_temporal = make_temporal_model(temporal_arch, close_size, close_size, N, model_d)

    def forward(self, x_c, x_p, flow):
        '''initial data size
        x_c: bs*closeness*2*N
        x_p: bs*len_period*closeness*2*N
        '''
        '''temporal input
        tx_c: bs*2*N*1*closeness
        tx_p: bs*2*N*len_period*closeness
        '''
        '''temporal output
        sq_p: bs*2*N*1*closeness
        '''

        bs = len(x_c)
        N = x_c.shape[-1]
        len_closeness = x_c.shape[1]

        # 【修改处 1】使用 .to(x_c.device) 替代 .cuda()
        tgt = x_c.permute((0, 2, 3, 1))[:, flow].unsqueeze(dim=-2).to(x_c.device)

        # 【修改处 2】使用 .to(x_p.device) 替代 .cuda()
        tx_p = x_p.permute(0, 3, 4, 1, 2).float().to(x_p.device)

        sq_p = self.p_temporal(tx_p[:, flow], tgt).squeeze(dim=-2)

        return sq_p