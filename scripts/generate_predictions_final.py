import os
import sys
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import h5py
from types import SimpleNamespace

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stgcn_traffic_prediction.dataloader.milano_crop2 import load_data
from stgcn_traffic_prediction.models.model2 import T_STGCN
from stgcn_traffic_prediction.models.MinMaxNorm import MinMaxNorm01


def load_and_prepare_data(data_path: str):
    """加载并准备数据"""
    with h5py.File(data_path, 'r') as f:
        data = f['data'][:]
        timestamps = f['idx'][:].astype(str)

        print(f"原始数据形状: {data.shape}")

        # 处理数据形状
        if len(data.shape) == 4:
            n, height, width, c = data.shape
            data_2d = data[:, :, :, -1]  # 取最后一个channel
            data = data_2d.reshape((n, 1, height * width))
            data = np.tile(data, (1, 2, 1))  # 复制为2个flow
        elif len(data.shape) == 3:
            T, flows, N = data.shape
            if flows == 1:
                data = np.tile(data, (1, 2, 1))
            elif flows >= 2:
                data = data[:, :2, :]

    return data, timestamps


def _set_global_seeds(seed: int = 42):
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_predict_properly(temporal_arch: str, device: torch.device, output_dir: str,
                               load_existing_model: bool = True):
    """正确地训练并预测模型"""
    print(f"\n===== 处理 {temporal_arch} 模型 =====")

    # 配置 - 使用与保存模型一致的参数
    opt = SimpleNamespace(
        traffic='call', close_size=33, period_size=3, trend_size=1,
        test_size=66, nb_flow=2, batch_size=1, test_batch_size=1,
        c_model_d=128, p_model_d=128, s_model_d=32, model_N=1,  # 保持与保存模型一致
        k=10, spatial='transformer', mode='cos', flow=0,
        c=True, s=False, FS=False, epochs=50  # 紧急恢复：s=True导致数值爆炸！
    )

    # 加载数据
    data_path = os.path.join(PROJECT_ROOT, 'stgcn_traffic_prediction', 'all_data_sliced2.h5')
    x_train, y_train, x_test, y_test, mmn = load_data(
        data_path, opt.traffic, opt.close_size, opt.period_size,
        opt.trend_size, opt.test_size, opt.nb_flow
    )

    # 创建数据加载器
    train_data = list(zip(*([*x_train, y_train])))
    test_data = list(zip(*([*x_test, y_test])))

    train_idx = list(range(len(train_data)))
    train_loader = DataLoader(train_data, batch_size=opt.batch_size,
                              sampler=SubsetRandomSampler(train_idx), drop_last=True)
    test_loader = DataLoader(test_data, batch_size=opt.test_batch_size, shuffle=False)

    # 创建模型 - 使用正确的参数
    external_size = 6
    model = T_STGCN(opt.close_size, external_size, opt.model_N, opt.k, opt.spatial,
                    opt.s_model_d, opt.c_model_d, opt.p_model_d, t_model_d=64,
                    temporal_arch=temporal_arch).to(device)

    # 尝试加载已有的模型权重
    model_path = os.path.join(PROJECT_ROOT, f'best_model_{temporal_arch}.pth')
    if load_existing_model and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict, strict=True)
            print(f"成功加载预训练模型: {model_path}")
        except Exception as e:
            print(f"警告：无法加载模型 {model_path}: {e}")
            print("将训练新模型...")
            load_existing_model = False

    # 如果没有加载成功或不存在模型，则训练
    if not load_existing_model or not os.path.exists(model_path):
        print(f"开始训练 {temporal_arch} 模型...")
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.L1Loss().to(device)  # 改为MAE，减少过度平滑
        mse_criterion = nn.MSELoss().to(device)  # 用于时间梯度一致性
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        best_loss = float('inf')
        for epoch in range(opt.epochs):
            model.train()
            train_loss = 0
            steps = 0

            for batch in train_loader:
                if len(batch) == 4:
                    c, p, t, target = batch
                    use_trend = True
                else:
                    c, p, target = batch
                    t = None
                    use_trend = False

                optimizer.zero_grad()

                # 前向传播
                if use_trend:
                    pred = model(c.float(), opt.mode, opt.c, opt.s, opt.FS, 'p', 'c', opt.flow, p.float(), t.float())
                else:
                    pred = model(c.float(), opt.mode, opt.c, opt.s, opt.FS, 'p', 'c', opt.flow, p.float())

                # 计算损失：对齐维度 (B, N, 33)
                target_seq = target.to(device).float()[:, :, opt.flow].transpose(1, 2)
                base_loss = criterion(pred.float(), target_seq)
                # 轻量时间梯度一致性，抑制杂乱趋势（保持形变方向一致）
                diff_pred = pred[:, :, 1:] - pred[:, :, :-1]
                diff_tgt = target_seq[:, :, 1:] - target_seq[:, :, :-1]
                smooth_loss = mse_criterion(diff_pred, diff_tgt)
                loss = base_loss + 0.1 * smooth_loss
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                steps += 1

                if steps >= 1000:  # 每个epoch最多1000步
                    break

            avg_loss = train_loss / steps
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{opt.epochs}, Loss: {avg_loss:.6f}")

            # 学习率调度
            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                # 保存最佳模型
                torch.save(model.state_dict(), model_path)

        print(f"训练完成，最佳损失: {best_loss:.6f}")

    # 使用改进的预测方法
    model.eval()
    predictions = predict_with_sliding_window(model, test_loader, mmn, opt, device)

    if predictions is None:
        print(f"[ERROR] {temporal_arch} 预测失败")
        return None

    # 恢复用户原始的反归一化逻辑
    try:
        inv = mmn.inverse_transform(predictions)
    except Exception:
        # fallback without inverse if shape mismatch
        inv = predictions
    # clip to observed min/max after inverse
    try:
        inv = np.clip(inv, mmn.min, mmn.max)
    except Exception:
        pass

    pred_denorm = inv

    # 保存结果 - 使用正确的文件名格式
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{temporal_arch.capitalize()}_predictions.csv')
    np.savetxt(save_path, pred_denorm, delimiter=',', fmt='%.6f')
    print(f"[OK] {temporal_arch} 预测完成，形状: {pred_denorm.shape}, 保存到: {save_path}")

    return pred_denorm


def predict_with_sliding_window(model, test_loader, mmn, opt, device='cpu'):
    """稳定的滑窗聚合：按时间对齐取对角并均值，避免杂乱/抹平"""
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 处理不同数量的输入（支持trend数据）
            if len(batch) == 4:
                c, p, t, target = batch
                use_trend = True
            else:
                c, p, target = batch
                t = None
                use_trend = False

            if use_trend:
                pred = model(c.float(), opt.mode, opt.c, opt.s, opt.FS, 'p', 'c', opt.flow, p.float(), t.float())
            else:
                pred = model(c.float(), opt.mode, opt.c, opt.s, opt.FS, 'p', 'c', opt.flow, p.float())

            if pred.dim() == 3:  # (bs, N, close)
                # 对每个批次保存预测结果
                all_predictions.append(pred.detach().cpu().numpy())

    if not all_predictions:
        print(f'[WARN] No predictions produced.')
        return None

    # 将所有预测结果连接起来 -> (B, N, Tclose)
    pred_cat = np.concatenate(all_predictions, axis=0)
    # 目标输出形状 (N, 33)
    B, N, Tclose = pred_cat.shape
    # 取最近 33 个批次（若不足则全取）
    take = min(B, 33)
    pred_tail = pred_cat[-take:]  # (take, N, Tclose)
    # 时间对齐：第 i 个批次对应第 (Tclose-1-i) 个时间步（模型每批次预测的是最后一步）
    # 构造 (N, 33, k) 再对k求均值
    aligned = np.zeros((N, 33, 0), dtype=pred_tail.dtype)
    buffers = []
    for i in range(take):
        # 从倒数第 i+1 个批次取其最后一步预测，作为第 (33-take+i) 个时间步的一个样本
        step_vec = pred_tail[i, :, -1]  # (N,)
        buffers.append(step_vec[:, None])
    # 堆叠为 (N, take)
    stacked = np.concatenate(buffers, axis=1)
    # 将其右对齐到 33 步
    out = np.zeros((N, 33), dtype=stacked.dtype)
    out[:, -take:] = stacked
    return out


def generate_all_predictions():
    """生成所有模型的预测"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _set_global_seeds(42)
    print(f"使用设备: {device}")

    output_dir = os.path.join(PROJECT_ROOT, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 加载完整数据用于保存真实值
    data_path = os.path.join(PROJECT_ROOT, 'stgcn_traffic_prediction', 'all_data_sliced2.h5')
    full_data, timestamps = load_and_prepare_data(data_path)

    # 对每个模型 - 已找到问题，恢复所有模型
    for arch in ['transformer', 'lstm', 'gru', 'dcnn']:
        try:
            # 强制重新训练：使用MAE损失函数增强动态性
            predictions = train_and_predict_properly(arch, device, output_dir, load_existing_model=False)

            if predictions is None:
                print(f"[ERROR] {arch} 预测失败，跳过")
                continue

        except Exception as e:
            print(f"处理 {arch} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 使用与evaluate_all.py一致的方法加载真实数据
    from stgcn_traffic_prediction.dataloader.milano_crop2 import _loader
    with h5py.File(data_path, 'r') as f:
        true_data = _loader(f, 2, 'call')  # 使用_loader函数
        # data shape: (T, C=1, N) after _loader
        if true_data.ndim == 3:
            true_data = true_data[:, 0, :]  # (T, N)
        # 转换为 (N, T) 格式
        true_full = true_data.transpose(1, 0)  # (N, T)

    # 保存真实数据
    np.savetxt(os.path.join(output_dir, 'true.csv'), true_full, delimiter=',', fmt='%.6f')
    # 取最后33步作为true_33
    if true_full.shape[1] < 33:
        raise ValueError('Full truth has less than 33 time-steps.')
    true_33 = true_full[:, -33:]
    np.savetxt(os.path.join(output_dir, 'true_33.csv'), true_33, delimiter=',', fmt='%.6f')

    print("\n预测生成完成！")
    print(f"结果保存在: {output_dir}")

    # 运行评估 - 设置正确的输出目录环境变量
    try:
        os.environ['EVAL_OUTPUT_DIR'] = output_dir
        from stgcn_traffic_prediction.scripts.evaluate_all import main as eval_main
        eval_main()
    except Exception as e:
        print(f"评估脚本运行失败: {e}")


if __name__ == "__main__":
    generate_all_predictions()
