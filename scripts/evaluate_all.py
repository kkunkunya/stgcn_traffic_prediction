import os
import sys
import json
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stgcn_traffic_prediction.dataloader.milano_crop2 import _loader
import h5py

# ---------- Config (edit as needed) ----------
_OVERRIDE_DIR = os.environ.get('EVAL_OUTPUT_DIR')
OUTPUT_DIR = _OVERRIDE_DIR if _OVERRIDE_DIR else os.path.join(PROJECT_ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

H5_PATH = os.path.join(PROJECT_ROOT, 'stgcn_traffic_prediction', 'all_data_sliced2.h5')
TRAFFIC_TYPE = 'call'  # keeps consistent with training
NB_FLOW = 2  # keeps consistent with training

MODEL_NAMES = ['Transformer', 'LSTM', 'GRU', 'DCNN']
PRED_FILENAMES = {
    'Transformer': 'Transformer_predictions.csv',
    'LSTM': 'LSTM_predictions.csv',
    'GRU': 'GRU_predictions.csv',
    'DCNN': 'DCNN_predictions.csv',
}

# 0-based time-step indices (length=33, 0..32). Confirmed list:
TIME_STEP_INDICES = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 32]

# Visualization style
FONT_FAMILY = 'Times New Roman'
POINT_STYLES = {
    'True': dict(color='black', marker='o', linestyle='-', linewidth=2, markersize=5),
    'Transformer': dict(color='red', marker='s', linestyle='-', linewidth=2, markersize=5),
    'LSTM': dict(color='blue', marker='^', linestyle='-', linewidth=2, markersize=5),
    'GRU': dict(color='green', marker='D', linestyle='-', linewidth=2, markersize=5),
    'DCNN': dict(color='orange', marker='*', linestyle='-', linewidth=2, markersize=7),
}


# ---------- Metrics ----------
def _safe_flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _safe_flat(y_true), _safe_flat(y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _safe_flat(y_true), _safe_flat(y_pred)
    return float(np.mean(np.abs(a - b)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    a, b = _safe_flat(y_true), _safe_flat(y_pred)
    return float(np.mean(np.abs((a - b) / (np.abs(a) + eps))))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _safe_flat(y_true), _safe_flat(y_pred)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _safe_flat(y_true), _safe_flat(y_pred)
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _safe_flat(y_true), _safe_flat(y_pred)
    var_a = np.var(a)
    if var_a == 0:
        return 0.0
    return float(1 - np.var(a - b) / (var_a + 1e-12))


def mtd(y_true: np.ndarray, y_pred: np.ndarray, p: float = 1.5) -> float:
    a, b = _safe_flat(y_true), _safe_flat(y_pred)
    return float(np.mean(np.abs(a - b) ** p) ** (1.0 / p))


def mpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _safe_flat(y_true), _safe_flat(y_pred)
    return float(np.mean(np.maximum(b - a, 0.0)))


def mgd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # mean gradient difference along time (assumes last axis is time)
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    if a.ndim == 1:
        da = np.diff(a)
        db = np.diff(b)
        return float(np.mean(np.abs(da - db)))
    # assume shape (points, time)
    da = np.diff(a, axis=-1)
    db = np.diff(b, axis=-1)
    return float(np.mean(np.abs(da - db)))


def ssim_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # simplified SSIM for 1D vectors
    a, b = _safe_flat(y_true), _safe_flat(y_pred)
    C1, C2 = 1e-4, 9e-4
    mu_a, mu_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a), np.var(b)
    cov = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2)
    return float(num / (den + 1e-12))


def psnr_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _safe_flat(y_true), _safe_flat(y_pred)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    max_val = np.max(np.abs(a)) + 1e-12
    return float(20 * math.log10(max_val / math.sqrt(mse)))


def compute_all_metrics(y_true_mat: np.ndarray, y_pred_mat: np.ndarray) -> Dict[str, float]:
    # inputs shape: (points, time)
    return {
        'RMSE': round(rmse(y_true_mat, y_pred_mat), 4),
        'MAE': round(mae(y_true_mat, y_pred_mat), 4),
        'MAPE': round(mape(y_true_mat, y_pred_mat), 4),
        'Pearson_r': round(pearson_r(y_true_mat, y_pred_mat), 4),
        'R2': round(r2_score(y_true_mat, y_pred_mat), 4),
        'EVar': round(explained_variance(y_true_mat, y_pred_mat), 4),
        'MTD_p1.5': round(mtd(y_true_mat, y_pred_mat, p=1.5), 4),
        'MPD': round(mpd(y_true_mat, y_pred_mat), 4),
        'MGD': round(mgd(y_true_mat, y_pred_mat), 4),
        'SSIM': round(ssim_1d(y_true_mat, y_pred_mat), 4),
        'PSNR': round(psnr_1d(y_true_mat, y_pred_mat), 4),
    }


# ---------- Data I/O ----------
def load_full_truth(h5_path: str, nb_flow: int, traffic_type: str) -> np.ndarray:
    with h5py.File(h5_path, 'r') as f:
        data = _loader(f, nb_flow, traffic_type)
        # data shape: (T, C=1, N) after _loader with nb_flow==2 returns (n,1,N)
        # Save as (N, T)
        if data.ndim == 3:
            data = data[:, 0, :]  # (T, N)
        elif data.ndim == 2:
            pass  # (T, N)
        else:
            raise ValueError('Unexpected data shape from _loader')
        return data.transpose(1, 0)


def save_csv(path: str, mat: np.ndarray):
    np.savetxt(path, mat, delimiter=',', fmt='%.6f')


def try_load_predictions(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        return None
    arr = np.loadtxt(path, delimiter=',')
    return np.asarray(arr)


# ---------- Visualization ----------
def _apply_axes_style(ax, xlabel: str, ylabel: str):
    ax.set_xlabel(xlabel, fontname=FONT_FAMILY, fontweight='bold', fontsize=14)
    ax.set_ylabel(ylabel, fontname=FONT_FAMILY, fontweight='bold', fontsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(FONT_FAMILY)
        label.set_fontweight('bold')
        label.set_fontsize(12)


def plot_point_overview(point_idx: int, dates: List[str], series: Dict[str, np.ndarray], out_path: str):
    plt.figure(figsize=(10, 5))
    x = np.arange(len(dates))
    # pick 4 evenly spaced ticks
    tick_idx = np.linspace(0, len(dates) - 1, num=4, dtype=int)

    for name, vec in series.items():
        style = POINT_STYLES[name]
        plt.plot(x, vec, label=name, **style)

    plt.xticks(tick_idx, [dates[i] for i in tick_idx])
    _apply_axes_style(plt.gca(), 'Time', 'InSAR time series deformation/mm')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_point_scatter(point_idx: int, series: Dict[str, np.ndarray], out_path: str):
    # X = true, Y = predictions of 4 models
    true_vec = series['True']
    plt.figure(figsize=(6, 6))
    for name in MODEL_NAMES:
        pred = series[name]
        style = POINT_STYLES[name]
        plt.scatter(true_vec, pred, label=name, c=style['color'], marker=style['marker'])
    _apply_axes_style(plt.gca(), 'Deformation Value/mm', 'Prediction Value/mm')
    # unify x/y limits to same range for fair comparison
    all_vals = [true_vec]
    for name in MODEL_NAMES:
        if name in series:
            all_vals.append(series[name])
    all_concat = np.concatenate([np.asarray(v).reshape(-1) for v in all_vals], axis=0)
    vmin, vmax = float(np.min(all_concat)), float(np.max(all_concat))
    pad = 0.05 * (vmax - vmin + 1e-6)
    lo, hi = vmin - pad, vmax + pad
    ax = plt.gca()
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_point_hist(point_idx: int, series: Dict[str, np.ndarray], vmin: float, vmax: float, out_path: str):
    plt.figure(figsize=(8, 5))
    bins = np.linspace(vmin, vmax, 30)
    for name, vec in series.items():
        style = POINT_STYLES[name]
        plt.hist(vec, bins=bins, alpha=0.5 if name != 'True' else 0.6, label=name, color=style['color'])
    _apply_axes_style(plt.gca(), 'Cumulative Deformation/mm', 'Number of pixels')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_timestep_hist(ts_idx: int, series_per_model: Dict[str, np.ndarray], vmin: float, vmax: float, out_path: str):
    # series_per_model[name]: (N,) at given ts
    plt.figure(figsize=(8, 5))
    bins = np.linspace(vmin, vmax, 30)
    order = ['True'] + [n for n in MODEL_NAMES if n in series_per_model]
    for name in order:
        arr = series_per_model[name]
        style = POINT_STYLES[name]
        plt.hist(arr, bins=bins, alpha=0.5 if name != 'True' else 0.6, label=name, color=style['color'])
    _apply_axes_style(plt.gca(), 'Cumulative Deformation/mm', 'Number of pixels')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------- Main pipeline ----------
def main():
    # 1) Load full truth and export true.csv and true_33.csv
    true_full = load_full_truth(H5_PATH, NB_FLOW, TRAFFIC_TYPE)  # (N, T_full)
    save_csv(os.path.join(OUTPUT_DIR, 'true.csv'), true_full)
    # take last 33 steps as true_33
    if true_full.shape[1] < 33:
        raise ValueError('Full truth has less than 33 time-steps.')
    true_33 = true_full[:, -33:]
    save_csv(os.path.join(OUTPUT_DIR, 'true_33.csv'), true_33)

    # 2) Load predictions (each should be (N, 33)) if they exist
    preds: Dict[str, np.ndarray] = {}
    for name in MODEL_NAMES:
        # try standard filename
        std_path = os.path.join(OUTPUT_DIR, PRED_FILENAMES[name])
        # also try upper-case variant for backward compatibility
        alt_path = os.path.join(OUTPUT_DIR, f'{name.upper()}_predictions.csv')
        arr = try_load_predictions(std_path)
        if arr is None:
            arr = try_load_predictions(alt_path)
        if arr is None:
            print(f'[WARN] Missing predictions CSV for {name}: {std_path} (alt: {alt_path})')
            continue
        if arr.shape != true_33.shape:
            print(
                f'[WARN] Shape mismatch for {name}: {arr.shape} vs true_33 {true_33.shape}; will skip this model in metrics/plots')
            # keep but mark as invalid by not inserting
            continue
        preds[name] = arr

    # 3) Points selection by variance on true_33
    var_per_point = np.var(true_33, axis=1)
    topk_idx = np.argsort(-var_per_point)[:4]
    np.savetxt(os.path.join(OUTPUT_DIR, 'points_to_analyze.csv'), topk_idx, fmt='%d', delimiter=',')

    # Build date labels for 33 steps
    # Using end date 2019-12-13 and 11-day interval backwards
    import datetime
    end_date = datetime.datetime(2019, 12, 13)
    dates = [(end_date - datetime.timedelta(days=11 * (32 - i))).strftime('%Y/%m/%d') for i in range(33)]

    # 4) Single-point figures
    vmin = float(np.min(true_33))
    vmax = float(np.max(true_33))
    pad = 0.1 * (vmax - vmin + 1e-6)
    vmin_p, vmax_p = vmin - pad, vmax + pad
    for p in topk_idx:
        series = {'True': true_33[p]}
        for name in MODEL_NAMES:
            if name in preds and preds[name].shape == true_33.shape:
                series[name] = preds[name][p]

        # Overview
        plot_point_overview(int(p), dates, series, os.path.join(OUTPUT_DIR, f'point_{int(p)}_overview.png'))
        # Scatter
        if all(k in preds for k in MODEL_NAMES):
            plot_point_scatter(int(p), series, os.path.join(OUTPUT_DIR, f'point_{int(p)}_scatter.png'))
        # Histogram
        plot_point_hist(int(p), series, vmin_p, vmax_p, os.path.join(OUTPUT_DIR, f'point_{int(p)}_hist.png'))

    # 5) Time-step histograms over selected indices
    for ts in TIME_STEP_INDICES:
        series_ts = {'True': true_33[:, ts]}
        for name in MODEL_NAMES:
            if name in preds and preds[name].shape == true_33.shape:
                series_ts[name] = preds[name][:, ts]
        plot_timestep_hist(ts, series_ts, vmin_p, vmax_p, os.path.join(OUTPUT_DIR, f'timestep_{ts}_hist.png'))

    # 6) Metrics (overall + per-point) and export CSVs
    import csv
    overall_rows = []
    per_point_rows = []
    for name in [n for n in MODEL_NAMES if n in preds and preds[n].shape == true_33.shape]:
        pred = preds[name]
        # overall
        m = compute_all_metrics(true_33, pred)
        overall_rows.append({'Model': name, **m})
        # per-point
        for p in range(true_33.shape[0]):
            mp = compute_all_metrics(true_33[p], pred[p])
            per_point_rows.append({'Point': p, 'Model': name, **mp})

    # write overall
    if overall_rows:
        keys = ['Model', 'RMSE', 'MAE', 'MAPE', 'Pearson_r', 'R2', 'EVar', 'MTD_p1.5', 'MPD', 'MGD', 'SSIM', 'PSNR']
        with open(os.path.join(OUTPUT_DIR, 'metrics_overall.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in overall_rows:
                w.writerow(r)
    # write per-point
    if per_point_rows:
        keys = ['Point', 'Model', 'RMSE', 'MAE', 'MAPE', 'Pearson_r', 'R2', 'EVar', 'MTD_p1.5', 'MPD', 'MGD', 'SSIM',
                'PSNR']
        with open(os.path.join(OUTPUT_DIR, 'metrics_per_point.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_point_rows:
                w.writerow(r)

    print('Evaluation finished. Outputs are saved to ./output/')


if __name__ == '__main__':
    main()


