import os
import sys
import numpy as np
import torch
import h5py
from types import SimpleNamespace

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stgcn_traffic_prediction.models.model2 import T_STGCN
from stgcn_traffic_prediction.models.MinMaxNorm import MinMaxNorm01


def load_full_data(data_path: str):
    """加载完整的原始数据用于自回归预测"""
    with h5py.File(data_path, 'r') as f:
        data = f['data'][:]  # (T, H, W, C) 或其他格式
        timestamps = f['idx'][:].astype(str)
        
        print(f"加载的原始数据形状: {data.shape}")
        
        # 按照milano_crop2.py的逻辑处理数据
        if len(data.shape) == 4:
            # 4D数据: (T, height, width, channels)
            n, height, width, c = data.shape
            # 只取最后一个channel (对应call数据)
            data_2d = data[:, :, :, -1]  # (T, H, W)
            # reshape为 (T, 1, H*W)
            data = data_2d.reshape((n, 1, height * width))
            # 复制为2个flow
            data = np.tile(data, (1, 2, 1))  # (T, 2, H*W)
            
        elif len(data.shape) == 3:
            # 3D数据: (T, flows, N)
            T, flows, N = data.shape
            if flows == 1:
                # 复制flow维度: (T, 1, N) -> (T, 2, N)
                data = np.tile(data, (1, 2, 1))
            elif flows >= 2:
                # 只取前2个flow
                data = data[:, :2, :]
        else:
            raise ValueError(f"不支持的数据形状: {data.shape}")
    
    print(f"处理后数据形状: {data.shape}")
    return data, timestamps


def autoregressive_predict(model, initial_data, steps_to_predict=33, device='cpu'):
    """
    实现自回归预测
    
    Args:
        model: 训练好的模型
        initial_data: 初始历史数据，形状 (history_len, 2, N)
        steps_to_predict: 要预测的步数
        device: 设备
    
    Returns:
        predictions: 预测结果，形状 (N, steps_to_predict)
    """
    model.eval()
    
    # 使用最后289步作为初始历史
    history_len = 289
    if initial_data.shape[0] < history_len:
        raise ValueError(f"初始数据长度 {initial_data.shape[0]} 小于所需历史长度 {history_len}")
    
    # 取最后289步作为历史
    history = initial_data[-history_len:].copy()  # (289, 2, N)
    N = history.shape[-1]
    
    # 存储预测结果
    predictions = []
    
    with torch.no_grad():
        for step in range(steps_to_predict):
            # 准备输入数据 - 使用最近33步作为closeness输入
            closeness_len = 33
            x_c = history[-closeness_len:]  # (33, 2, N)
            x_c = torch.from_numpy(x_c).float().unsqueeze(0)  # (1, 33, 2, N)
            
            # 准备period输入（简化为zeros，或者可以根据需要设计）
            x_p = torch.zeros(1, 3, 33, 2, N).float()  # (1, 3, 33, 2, N)
            
            # 预测
            try:
                # 注意：训练与预测配置保持一致：s=False，避免形状/显存问题
                # 模型输出实际为 (bs, T, N)
                pred = model(x_c, 'cos', True, False, False, 'p', 'c', 0, x_p)
                # 取第一个时间步 (T=0) 的所有 N 个点
                if pred.dim() == 3 and pred.shape[1] >= 1:
                    next_pred = pred[0, 0, :].cpu().numpy()  # (N,)
                else:
                    raise ValueError(f"模型输出形状错误: {pred.shape}")
                    
            except Exception as e:
                print(f"预测第{step+1}步时出错: {e}")
                # 如果出错，用最后一步数据作为预测
                next_pred = history[-1, 0, :]  # 使用flow=0的数据
            
            predictions.append(next_pred)
            
            # 更新历史：移除最老的一步，添加新预测的一步
            new_step = np.zeros((1, 2, N))
            new_step[0, 0, :] = next_pred  # flow=0使用预测值
            new_step[0, 1, :] = history[-1, 1, :]  # flow=1保持最后一个值（或者也可以预测）
            
            # 滑动窗口：移除第一步，添加新步
            history = np.concatenate([history[1:], new_step], axis=0)
            
            if (step + 1) % 10 == 0:
                print(f"已预测 {step + 1}/{steps_to_predict} 步")
    
    # 转换为 (N, 33) 格式
    predictions = np.array(predictions).T  # (steps_to_predict, N) -> (N, steps_to_predict)
    print(f"自回归预测完成，结果形状: {predictions.shape}")
    
    return predictions


def generate_autoregressive_predictions(temporal_arch: str, model_path: str, data_path: str, 
                                      output_dir: str, device: torch.device):
    """
    生成某个模型的自回归预测结果
    """
    print(f"\n开始生成 {temporal_arch} 的自回归预测...")
    
    # 1. 加载完整数据和归一化器
    data, timestamps = load_full_data(data_path)
    
    # 2. 设置归一化器（使用前面的数据拟合）
    test_size = 66
    mmn = MinMaxNorm01()
    # 按照milano_crop2的逻辑：reshape后再拟合
    data_for_fit = data[:-test_size].reshape((-1, data.shape[-1]))  # (T_train*2, N)
    mmn.fit(data_for_fit)
    
    # 3. 归一化完整数据
    data_reshaped = data.reshape((-1, data.shape[-1]))  # (T*2, N)
    data_normalized_reshaped = mmn.transform(data_reshaped)
    data_normalized = data_normalized_reshaped.reshape(data.shape)  # (T, 2, N)
    
    # 4. 加载模型 - 使用与训练时一致的配置（与 final 保持一致）
    opt = SimpleNamespace(
        close_size=33, period_size=3, trend_size=1, nb_flow=2,
        k=10, model_N=2, c_model_d=128, p_model_d=128, s_model_d=64,
        spatial='transformer'
    )
    
    external_size = 6
    model = T_STGCN(opt.close_size, external_size, opt.model_N, opt.k, opt.spatial,
                    opt.s_model_d, opt.c_model_d, opt.p_model_d, t_model_d=64,
                    temporal_arch=temporal_arch).to(device)
    
    # 5. 加载训练好的权重
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint, strict=True)
        print(f"成功加载模型权重: {model_path}")
    except Exception as e:
        print(f"警告：无法加载预训练权重 {model_path}: {e}")
        print("使用随机初始化的模型（仅用于测试）")
    
    # 6. 执行自回归预测
    # 使用测试期开始前的数据作为初始历史
    test_start_idx = len(data) - test_size - 33  # 确保有足够的历史
    initial_data = data_normalized[:test_start_idx + 289]  # 取到测试开始前的289步
    
    predictions = autoregressive_predict(model, initial_data, steps_to_predict=33, device=device)
    
    # 7. 反归一化
    try:
        # predictions形状: (N, 33), 需要reshape为 (N*33,) 进行反归一化
        pred_flat = predictions.flatten()  # (N*33,)
        pred_denorm_flat = mmn.inverse_transform(pred_flat.reshape(1, -1)).flatten()  # (N*33,)
        predictions_denorm = pred_denorm_flat.reshape(predictions.shape)  # (N, 33)
        # 裁剪到合理范围
        predictions_denorm = np.clip(predictions_denorm, mmn.min, mmn.max)
    except Exception as e:
        print(f"反归一化失败: {e}, 使用原始预测值")
        predictions_denorm = predictions
    
    # 8. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{temporal_arch.upper()}_predictions.csv')
    np.savetxt(output_file, predictions_denorm, delimiter=',', fmt='%.6f')
    
    print(f"保存 {temporal_arch} 预测结果到: {output_file}")
    print(f"预测矩阵形状: {predictions_denorm.shape}")
    return predictions_denorm


if __name__ == "__main__":
    # 测试自回归预测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    data_path = os.path.join(PROJECT_ROOT, 'stgcn_traffic_prediction', 'all_data_sliced2.h5')
    output_dir = './output_autoregressive_test'
    
    # 生成一个模型的预测（即使没有预训练权重也可以测试流程）
    temporal_arch = 'transformer'
    model_path = f'./model_{temporal_arch}.pth'  # 这个文件可能不存在，仅测试流程
    
    try:
        predictions = generate_autoregressive_predictions(
            temporal_arch, model_path, data_path, output_dir, device
        )
        print("自回归预测测试完成！")
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
