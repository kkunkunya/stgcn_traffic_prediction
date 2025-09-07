# -*- coding: utf-8 -*-
"""
/*******************************************
**  license
********************************************/
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, scales=[4, 16], kernel_size=3):  # 移除空洞率1
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                dilation=scale,
                padding=scale * (kernel_size - 1) // 2
            ) for scale in scales
        ])

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        return torch.cat(features, dim=1)
class STMatrix(object):
    def __init__(self, data, timestamps, T=33, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps #list[(index,timestamp)]
        self.T = T
        self.pd_timestamps = timestamps
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i


    def get_matrix(self, timestamp, len_closeness, closeness):
        index = self.get_index[timestamp]
        if(closeness):
            return self.data[index]
        else:
            return self.data[np.arange(index,index+len_closeness)]


    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True


    def create_dataset(self, len_closeness=33, len_trend=3, TrendInterval=3, len_period=3, PeriodInterval=1):#PeriodInterval=1
        offset_frame = pd.DateOffset(days=11)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness+1),#1,2,3,...34
                   [PeriodInterval * self.T * j for j in range(1, len_period+1)],#33,66,99
                   [TrendInterval * self.T * j for j in range(1, len_trend+1)]]
        i = max(self.T * PeriodInterval * len_period, len_closeness)
        multiscale_conv = MultiScaleTemporalConv(scales=[4, 16])
        while i < len(self.pd_timestamps)-len_closeness:
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue

            # ------------------------- 生成新XP（含残差连接） --------------------------
            raw_data = self.data[i - len_closeness:i]

            # 1. 多尺度卷积特征提取
            x_tensor = torch.Tensor(raw_data).permute(1, 2, 0)  # [1, 2880, 33]
            sensor_features = []
            for sensor in range(2880):#数据点数目
                x_sensor = x_tensor[:, sensor:sensor + 1, :]  # [1, 1, 33]
                x_conv = multiscale_conv(x_sensor)  # [1, 2, 33]
                sensor_features.append(x_conv)
            x_conv_all = torch.stack(sensor_features, dim=3)  # [1, 2, 33, 2880]
            x_conv_all = x_conv_all.permute(1, 2, 0, 3)# [2, 33, 1, 2880]
            # 2. 原始数据作为残差分支（1通道）
            x_raw = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame, len_closeness, closeness=False) for j in   depends[1]]
            x_raw = torch.tensor(x_raw, dtype=torch.float32)
            # x_raw = torch.Tensor(raw_data).unsqueeze(0)  # [1, 33, 1, 2880]

            # 3. 拼接多尺度特征和原始特征（2+1=3通道）
            x_combined = torch.cat([x_conv_all, x_raw], dim=0)  # [3, 33, 1, 2880]
            # x_p = x_combined.squeeze(0).detach().numpy()  # [33, 3, 2880]
            # x_p = np.expand_dims(x_p, axis=3)
            x_p = x_combined.detach().numpy()  # [33, 3, 2880]
            x_p = x_p
            XP.append(x_p)
            # ------------------------------------------------------------------------

            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame,len_closeness,closeness=True) for j in depends[0]]
            # x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame,len_closeness,closeness =False) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame, len_closeness,closeness =False) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i],len_closeness,closeness=False)
            print(x_c[0].shape,x_p[0].shape) #(1,2880) (3,1,2880)
            if len_closeness > 0:
                XC.append(np.stack(x_c))
            if len_period > 0:
                XP.append(np.stack(x_p))
            if len_trend > 0:
                XT.append(np.stack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        print(timestamps_Y)
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)

        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)

        return XC, XP, XT, Y, timestamps_Y


    # def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
    #     offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
    #     XC = []
    #     XP = []
    #     XT = []
    #     Y = []
    #     timestamps_Y = []
    #     depends = [range(1, len_closeness+1),
    #                [PeriodInterval * self.T * j for j in range(1, len_period+1)],
    #                [TrendInterval * self.T * j for j in range(1, len_trend+1)]]

    #     i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
    #     while i < len(self.pd_timestamps):
    #         Flag = True
    #         for depend in depends:
    #             if Flag is False:
    #                 break
    #             Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

    #         if Flag is False:
    #             i += 1
    #             continue
    #         x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
    #         x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
    #         x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
    #         y = self.get_matrix(self.pd_timestamps[i])
    #         if len_closeness > 0:
    #             XC.append(np.vstack(x_c))
    #         if len_period > 0:
    #             XP.append(np.vstack(x_p))
    #         if len_trend > 0:
    #             XT.append(np.vstack(x_t))
    #         Y.append(y)
    #         timestamps_Y.append(self.timestamps[i])
    #         i += 1
    #     XC = np.asarray(XC)
    #     XP = np.asarray(XP)
    #     XT = np.asarray(XT)
    #     Y = np.asarray(Y)
    #     print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    #     return XC, XP, XT, Y, timestamps_Y