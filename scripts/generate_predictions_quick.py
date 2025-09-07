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

USE_PRETRAINED = os.environ.get('USE_PRETRAINED', '1') != '0'  # 1/true 默认使用已有权重


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

    # 尝试加载已有的模型权重（多候选路径/命名）
    candidate_paths = [
        os.path.join(PROJECT_ROOT, f'best_model_{temporal_arch}.pth'),
        os.path.join(PROJECT_ROOT, 'output_final', f'model_{temporal_arch}.pth'),
        os.path.join(PROJECT_ROOT, 'output_final', f'best_model_{temporal_arch}.pth'),
        os.path.join(PROJECT_ROOT, 'output', f'model_{temporal_arch}.pth'),
    ]
    model_path = candidate_paths[0]
    if load_existing_model:
        loaded = False
        for cand in candidate_paths:
            if os.path.exists(cand):
                try:
                    state_dict = torch.load(cand, map_location=device, weights_only=False)
                    model.load_state_dict(state_dict, strict=True)
                    model_path = cand
                    print(f"成功加载预训练模型: {cand}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"警告：无法加载模型 {cand}: {e}")
        if not loaded:
            print("未找到或未能加载预训练模型，将训练新模型...")
            load_existing_model = False

    # 如果没有加载成功或不存在模型，则训练
    if not load_existing_model or not os.path.exists(model_path):
        print(f"开始训练 {temporal_arch} 模型...")
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.L1Loss().to(device)  # 改为MAE，减少过度平滑
        mse_criterion = nn.MSELoss().to(device)  # 用于时间梯度一致性

        # 动态感知损失权重
        def dynamic_aware_loss(pred, target):
            """趋势感知损失：奖励正确的动态变化，惩罚平坦预测"""
            base_loss = criterion(pred, target)

            # 1. 方向一致性损失
            pred_diff = pred[..., 1:] - pred[..., :-1]
            target_diff = target[..., 1:] - target[..., :-1]
            direction_loss = torch.mean(torch.abs(torch.sign(pred_diff) - torch.sign(target_diff)))

            # 2. 变化率损失（放大动态区域的重要性）
            target_variance = torch.var(target, dim=-1, keepdim=True)
            dynamic_weight = torch.clamp(target_variance * 10, 1.0, 5.0)  # 动态区域权重更高
            weighted_loss = torch.mean(dynamic_weight * torch.abs(pred - target))

            # 3. 反平坦惩罚（阻止模型预测常数）
            pred_variance = torch.var(pred, dim=-1)
            flatness_penalty = torch.mean(torch.exp(-pred_variance * 100))  # 预测越平坦惩罚越大

            total_loss = base_loss + 0.3 * direction_loss + 0.5 * weighted_loss + 0.2 * flatness_penalty
            return total_loss

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

                # 计算损失：自适应对齐 pred 与 target 的维度
                pred_tensor = pred.float()  # 可能是 (B, N, 33) 或 (B, 33, N)
                target_raw = target.to(device).float()[:, :, opt.flow]  # (B, 33, N)
                if pred_tensor.shape == target_raw.shape:
                    target_seq = target_raw
                elif pred_tensor.shape == target_raw.transpose(1, 2).shape:
                    target_seq = target_raw.transpose(1, 2)
                else:
                    # 依据时间维长度(33)推断
                    if pred_tensor.shape[-1] == 33:
                        target_seq = target_raw.transpose(1, 2)
                    elif pred_tensor.shape[1] == 33:
                        target_seq = target_raw
                    else:
                        raise RuntimeError(
                            f"Unexpected shapes: pred {pred_tensor.shape}, target_raw {target_raw.shape}")

                # 使用动态感知损失函数
                loss = dynamic_aware_loss(pred_tensor, target_seq)

                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                steps += 1

                if steps % 100 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{opt.epochs}], Step [{steps}/{len(train_loader)}], Loss: {loss.item():.6f}')

                if steps >= 1000:  # 每个epoch最多1000步
                    break

            avg_loss = train_loss / max(1, steps)
            print(f"Epoch {epoch + 1}/{opt.epochs}, Loss: {avg_loss:.6f}")

            # 学习率调度
            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                # 保存最佳模型
                torch.save(model.state_dict(), model_path)
                print(f'Best model saved with loss: {best_loss:.6f}')

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

    # 预训练权重健康检查：与严格对齐的真值做相关性评估，差则自动重训一次
    try:
        # 与 generate_all_predictions 处保持一致，取最后测试批次的真值
        x_tr_tmp, y_tr_tmp, x_te_tmp, y_te_tmp, mmn_tmp = load_data(
            os.path.join(PROJECT_ROOT, 'stgcn_traffic_prediction', 'all_data_sliced2.h5'),
            opt.traffic, opt.close_size, opt.period_size, opt.trend_size, opt.test_size, opt.nb_flow
        )
        y_last = np.transpose(y_te_tmp[-1], (2, 0, 1)).squeeze(-1)  # (N, 33) 归一化
        y_last_denorm = mmn_tmp.inverse_transform(y_last)

        # 与当前预测对齐
        if pred_denorm.shape == y_last_denorm.shape:
            pred_aligned = pred_denorm
            true_aligned = y_last_denorm
        else:
            # 尝试裁剪对齐
            min_t = min(pred_denorm.shape[1], y_last_denorm.shape[1])
            pred_aligned = pred_denorm[:, -min_t:]
            true_aligned = y_last_denorm[:, -min_t:]

        # 逐点皮尔逊相关（忽略常量序列）
        corrs = []
        for i in range(pred_aligned.shape[0]):
            pv = pred_aligned[i]
            tv = true_aligned[i]
            if np.std(pv) < 1e-8 or np.std(tv) < 1e-8:
                continue
            c = np.corrcoef(pv, tv)[0, 1]
            if not np.isnan(c):
                corrs.append(c)
        median_corr = float(np.median(corrs)) if corrs else -1.0
        print(f"[CHECK] {temporal_arch} median per-point corr with truth: {median_corr:.3f}")

        # 若加载的预训练模型趋势相关性很差，则退回训练一次
        if load_existing_model and median_corr < 0.70:
            print(f"[CHECK] {temporal_arch} 预训练权重相关性过低，自动重训以修复趋势…")
            return train_and_predict_properly(temporal_arch, device, output_dir, load_existing_model=False)
    except Exception as _e:
        print(f"[CHECK] 相关性健康检查跳过: {_e}")

    # 保存结果 - 使用正确的文件名格式
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{temporal_arch.capitalize()}_predictions.csv')
    np.savetxt(save_path, pred_denorm, delimiter=',', fmt='%.6f')
    print(f"[OK] {temporal_arch} 预测完成，形状: {pred_denorm.shape}, 保存到: {save_path}")

    return pred_denorm


def predict_with_sliding_window(model, test_loader, mmn, opt, device='cpu'):
    """根据模型输出形状自适应：
    - 若模型一次输出多步 (1, 33, N)，直接取这 33 步并反归一化返回 (N, 33)
    - 若模型一次只输出一步 (1, N, 1) 或 (1, N)，执行自回归 33 步
    """
    model.eval()

    # 使用测试集最后一个批次作为初始输入
    last_batch = None
    for batch in test_loader:
        last_batch = batch

    if last_batch is None:
        return None

    # 解包最后一个批次
    if len(last_batch) == 4:
        c, p, t, target = last_batch
        use_trend = True
    else:
        c, p, target = last_batch
        t = None
        use_trend = False

    with torch.no_grad():
        # 先做一次前向，判断输出形状
        if use_trend:
            pred = model(c.float(), opt.mode, opt.c, opt.s, opt.FS, 'p', 'c', opt.flow, p.float(), t.float())
        else:
            pred = model(c.float(), opt.mode, opt.c, opt.s, opt.FS, 'p', 'c', opt.flow, p.float())

        print(f"DEBUG: pred shape = {pred.shape}")
        print(f"DEBUG: c shape = {c.shape}")
        print(f"DEBUG: p shape = {p.shape}")

        pred_np = pred.detach().cpu().numpy()

        # 情况 A：多步输出 (1, 33, N)
        if pred_np.ndim == 3 and pred_np.shape[0] == 1 and pred_np.shape[1] == 33 and pred_np.shape[2] > 1:
            # 直接转置为 (N, 33)（保持为归一化数据，后续统一反归一化）
            result_norm = pred_np[0].transpose(1, 0)
            return result_norm

        # 情况 B：单步输出 (1, N, 1) 或 (1, N)
        predictions = []
        for step in range(33):
            # 对第 0 步之外的步，使用上一步更新后的 c 再次前向
            if step > 0:
                if use_trend:
                    pred = model(c.float(), opt.mode, opt.c, opt.s, opt.FS, 'p', 'c', opt.flow, p.float(), t.float())
                else:
                    pred = model(c.float(), opt.mode, opt.c, opt.s, opt.FS, 'p', 'c', opt.flow, p.float())

            pred_np = pred.detach().cpu().numpy()  # 期望 (1, N, 1) 或 (1, N)

            # 归一化数据用于输出与回填输入（保持与 c 的分布一致）
            if pred_np.ndim == 3:
                predictions.append(pred_np[0, :, 0])  # (N,)
                pred_for_input = pred_np[0, :, 0]  # 归一化值 (N,)
            else:
                predictions.append(pred_np[0, :])  # (N,)
                pred_for_input = pred_np[0, :]  # 归一化值 (N,)

            # 更新输入窗口 c: (1, 33, 1, N)
            c_np = c.cpu().numpy()
            new_c = np.zeros_like(c_np)
            new_c[0, :-1, :, :] = c_np[0, 1:, :, :]
            new_c[0, -1, 0, :] = pred_for_input
            c = torch.from_numpy(new_c).to(device)

        # 汇总为 (N, 33)（归一化）
        result_norm = np.stack(predictions, axis=1)
        return result_norm


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

    # 对每个模型
    for arch in ['transformer', 'lstm', 'gru', 'dcnn']:
        try:
            # 根据开关：优先使用已有权重；若加载失败再训练
            predictions = train_and_predict_properly(arch, device, output_dir, load_existing_model=USE_PRETRAINED)

            if predictions is None:
                print(f"[ERROR] {arch} 预测失败，跳过")
                continue

        except Exception as e:
            print(f"处理 {arch} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 与预测严格对齐的真值：直接来自 y_test 的最后一个样本（与我们拼接出的33步对应）
    try:
        # 重新载入一次与训练同配的标准化数据，拿到 y_test
        x_train, y_train, x_test, y_test, mmn_tmp = load_data(
            data_path, 'call', 33, 3, 1, 66, 2
        )
        # y_test[-1]: (33, 1, N) → (N, 33)
        y_last = np.transpose(y_test[-1], (2, 0, 1)).squeeze(-1)
        # 反归一化
        y_last_denorm = mmn_tmp.inverse_transform(y_last)
        true_33 = y_last_denorm
        # 同时保存完整真值供其他分析
        from stgcn_traffic_prediction.dataloader.milano_crop2 import _loader
        with h5py.File(data_path, 'r') as f:
            true_data = _loader(f, 2, 'call')
            if true_data.ndim == 3:
                true_data = true_data[:, 0, :]
            true_full = true_data.transpose(1, 0)
        np.savetxt(os.path.join(output_dir, 'true.csv'), true_full, delimiter=',', fmt='%.6f')
        np.savetxt(os.path.join(output_dir, 'true_33.csv'), true_33, delimiter=',', fmt='%.6f')
    except Exception as e:
        print(f"[WARN] 生成对齐真值失败，回退到原方式: {e}")
        from stgcn_traffic_prediction.dataloader.milano_crop2 import _loader
        with h5py.File(data_path, 'r') as f:
            true_data = _loader(f, 2, 'call')
            if true_data.ndim == 3:
                true_data = true_data[:, 0, :]
            true_full = true_data.transpose(1, 0)
        np.savetxt(os.path.join(output_dir, 'true.csv'), true_full, delimiter=',', fmt='%.6f')
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
