import scipy.sparse as sp
from scipy.sparse.linalg import norm
import os, pickle, torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from tqdm import tqdm 
import bisect
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random as random

def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    #adj = 0.5 * (dir_adj + dir_adj.transpose())
    
    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id
    
    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso

def set_env(seed):
    # Set available CUDA devices
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def calc_chebynet_gso(gso):
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    # If you encounter a NotImplementedError, please update your scipy version to 1.10.1 or later.
    eigval_max = norm(gso, 2)

    # If the gso is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso

def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')

def evaluate_model(model, loss, data_iter, time_step=1, args=None):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            x = x.permute(0, 3, 1, 2).to(args.device) # [:,0,:,:].unsqueeze(1).to(args.device)# [batch_size, n_his, n_vertex]
            y = y.permute(0, 3, 1, 2)[:,0,time_step,:].to(args.device) # [batch_size, num_nodes, n_pred]
            y_pred = model(x).squeeze()[:,time_step,:]  # [batch_size, T, num_nodes]
            # print(y_pred.shape, y.shape) # [batch_size, num_nodes]
            # y = y[:,time_step,:]  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

class MaskedMAELoss(nn.Module):
    def __init__(self, null_val=0.0):
        super().__init__()
        self.null_val = null_val

    def forward(self, pred, target):
        mask = (target != self.null_val).float()
        mask /= (mask.mean() + 1e-6)
        return torch.abs(pred - target).mul(mask).mean()
    
class MaskedMSELoss(nn.Module):
    def __init__(self, null_val=0.0):
        super().__init__()
        self.null_val = null_val

    def forward(self, pred, target):
        mask = (target != self.null_val).float()
        mask /= (mask.mean() + 1e-6)
        return ((pred - target)**2).mul(mask).mean()


# def evaluate_metric(model, data_iter, scaler):
#     model.eval()
#     with torch.no_grad():
#         mae, sum_y, mape, mse = [], [], [], []
#         for x, y in data_iter:
#             y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
#             y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
#             # print(x.shape, y.shape, y_pred.shape) # B*T 
#             d = np.abs(y - y_pred)
#             mae += d.tolist()
#             sum_y += y.tolist()
#             mape += (d / (y+1e-5)).tolist()
#             mse += (d ** 2).tolist()
#         MAE = np.array(mae).mean()
#         #MAPE = np.array(mape).mean()
#         RMSE = np.sqrt(np.array(mse).mean())
#         WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

#         #return MAE, MAPE, RMSE
#         return MAE, RMSE, WMAPE
def evaluate_metric(model,
                    data_iter,
                    scaler,
                    null_val=0.0,        # 결측치로 간주할 값
                    use_mask=False,
                    time_step=1,
                    args=None):      # 마스킹 적용 여부
    """
    - null_val : target 속 결측치 값(대개 0.0). None 이면 마스킹 안 함.
    - use_mask : True일 때만 null_val 마스킹.
    """
    model.eval()
    with torch.no_grad():
        ae, se, sum_y = [], [], []

        for x, y in data_iter:
            y =y.permute(0, 3, 1, 2)[:,0,:,:].to(args.device)  # [B, N, T]
            x = x.permute(0, 3, 1, 2).to(args.device) #[:,0,:,:].unsqueeze(1).to(args.device)  # [B, 1, n_his, N]
            y_true  = scaler.inverse_transform(y[:,time_step,:]).cpu().numpy()           # (B, N)
            y_pred  = scaler.inverse_transform(
                          model(x).squeeze()[:,time_step,:].view(len(x), -1)).cpu().numpy()         # (B, N)

            # ──────────── 마스킹 적용 ────────────
            if use_mask and null_val is not None:
                mask = (y_true != null_val)
                # 결측치는 계산에서 제외
                diff = np.abs(y_true[mask] - y_pred[mask])
                y_vals = y_true[mask]
            else:
                diff  = np.abs(y_true - y_pred).reshape(-1)
                y_vals = y_true.reshape(-1)
            # ─────────────────────────────────────

            ae.extend(diff.reshape(-1))
            se.extend((diff ** 2).reshape(-1))
            sum_y.extend(y_vals.reshape(-1))

        ae   = np.array(ae)
        se   = np.array(se)
        sumy = np.array(sum_y) + 1e-8        # 분모 0 방지

        MAE   = ae.mean()
        RMSE  = np.sqrt(se.mean())
        WMAPE = ae.sum() / sumy.sum()

        return MAE, RMSE, WMAPE

def iterative_forecast(model, initial_x, n_pred_steps, device):
    """
    STGCN 등 단일 시점 예측 모델로 다중 시점(autoregressive) 예측을 수행합니다.

    :param model: 학습된 STGCN 모델
    :param initial_x: 예측을 시작할 초기 시퀀스 데이터 (shape: [B, 1, n_his, V] 또는 [B, n_his, V])
    :param n_pred_steps: 예측할 미래 스텝의 수 (예: 48)
    :param device: 연산을 수행할 장치 (e.g., 'cuda' or 'cpu')
    :return: 예측값 배열 (shape: [n_pred_steps, V])
    """
    # ── 0) 입력 차원 정리 ────────────────────────────────────────────
    if initial_x.dim() == 3:                   # [B, n_his, V] 형태인 경우
        initial_x = initial_x.unsqueeze(1)     # [B, 1, n_his, V] 형태로 변환

    model.eval().to(device)
    current_sequence = initial_x.clone().to(device)   # [B, 1, n_his, V]
    pred_list = []

    with torch.no_grad():
        for _ in range(n_pred_steps):
            # 1) 다음 한 스텝을 예측합니다.
            y_pred = model(current_sequence)
            
            # ── 1-1) 필요 없는 차원을 제거하여 [B, V] 형태로 통일합니다.
            while y_pred.dim() > 2:
                # 채널 또는 시간 차원의 크기가 1이면 해당 차원을 제거합니다.
                if y_pred.shape[1] == 1:
                    y_pred = y_pred.squeeze(1)
                else:
                    # 예외 처리: 예상치 못한 형태의 출력이 나오면 에러를 발생시킵니다.
                    raise ValueError(f"예상치 못한 출력 형태입니다. Shape: {y_pred.shape}")

            # 배치 차원을 제거하고 CPU로 이동하여 numpy 배열로 변환 후 저장합니다.
            pred_list.append(y_pred.squeeze(0).cpu().numpy())

            # 2) 다음 예측을 위한 입력 데이터를 준비합니다.
            # 예측 결과를 [B, 1, 1, V] 형태로 변환합니다.
            y_reshaped = y_pred.unsqueeze(1).unsqueeze(1)
            
            # 현재 시퀀스에서 가장 오래된 데이터를 버리고, 맨 뒤에 새로운 예측값을 이어 붙입니다.
            current_sequence = torch.cat(
                (current_sequence[:, :, 1:, :], y_reshaped), dim=2
            )

    # 예측값 리스트를 하나의 numpy 배열로 합칩니다.
    return np.stack(pred_list, axis=0)                 # 최종 shape: [n_pred_steps, V]


def visualize(model,
              test_loader,
              zscore_scaler,
              n_pred_steps=48,
              node_idx=0,
              time_label=None,
              save_dir='./visualizations',
              device='cuda:1',
              rotate_xticks=45,
              plot_ground_truth=True):
    """
    지정된 노드에 대해 24~48시간 후의 예측과 실제 값을 비교하여 시각화합니다.

    :param model: 학습된 모델
    :param test_loader: 테스트 데이터로더
    :param zscore_scaler: 데이터 정규화에 사용된 Z-score 스케일러 객체
    :param n_pred_steps: 총 예측할 스텝 수
    :param node_idx: 시각화할 노드의 인덱스
    :param time_labels: x축에 표시될 시간 레이블 리스트
    :param save_dir: 그래프를 저장할 디렉토리
    :param device: 연산을 수행할 장치
    :param rotate_xticks: x축 레이블의 회전 각도
    :param plot_ground_truth: 실제 값을 함께 표시할지 여부
    """
    model.to(device).eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # 예시: 노드 인덱스를 실제 이름으로 매핑하는 사전 (파일이 없는 경우를 대비한 예외 처리)
    try:
        with open('./script/label_to_station.pkl', 'rb') as f:
            label_to_station = pickle.load(f)
    except FileNotFoundError:
        label_to_station = {}
        print("Warning: label_to_station.pkl 파일을 찾을 수 없습니다. 노드 인덱스로 표시됩니다.")

    with torch.no_grad():
        # 테스트 데이터로더에서 첫 번째 배치만 가져와서 사용합니다.
        for x_test, y_test in test_loader:
            # ── 입력 텐서 차원 정리 ────────────────────────────────
            if x_test.dim() == 3:
                initial_x = x_test[:1].unsqueeze(1)
            else:
                initial_x = x_test[:1]
            initial_x = initial_x.to(device)

            # ── 48-step 예측 수행 및 역정규화 ───────────────────
            preds = iterative_forecast(model, initial_x, n_pred_steps, device)
            preds_denorm = zscore_scaler.inverse_transform(preds)
            preds_node = preds_denorm[:, node_idx]

            # ── 실제 값(Ground Truth) 준비 (옵션) ───────────────
            y_true_node = None
            # plot_ground_truth가 True이고, y_test가 다중 스텝 데이터를 포함할 때만 실행
            if plot_ground_truth:
                y_true = y_test[:24].cpu().numpy()
                # y_true_denorm = zscore_scaler.inverse_transform(y_true)
                # y_true_node = y_true_denorm[:, node_idx]
                y_true_node = y_true[node_idx, : ]
                print(y_true_node.shape) # 396
            else:
                plot_ground_truth = False # 다중 스텝 실제 값이 없으면 플래그를 비활성화

            # ── x축 구성 (24시간부터 47시간까지) ───────────────
            if time_label is not None:
                x_axis = time_label[24:n_pred_steps]
            else:
                x_axis = np.arange(24, n_pred_steps)

            # ── 그래프 그리기 ──────────────────────────────────
            plt.figure(figsize=(15, 5))
            plt.plot(x_axis, preds_node[24:], label='Prediction', lw=2, color='orangered')
            
            if plot_ground_truth:
                plt.plot(x_axis, y_true_node[24:], '--', label='Ground Truth', color='royalblue')

            station_name = label_to_station.get(node_idx, f'Node {node_idx}')
            plt.title(f'"{station_name}" - 24 to 48 Hours Ahead Forecast', fontsize=16)
            plt.xlabel('Hour Ahead' if time_label is None else 'Datetime', fontsize=12)
            plt.ylabel('Demand', fontsize=12)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            
            # x축 레이블이 겹치지 않도록 회전
            plt.xticks(rotation=rotate_xticks)
            plt.tight_layout()

            # 그래프 저장
            file_name = os.path.join(save_dir, f'forecast_node_{node_idx}.png')
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'✔ 시각화 그래프 저장 완료: {file_name}')
            break # 첫 번째 배치만 시각화하고 종료
        
class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
def evaluate_metric_multi(model,
                          data_iter,
                          scaler,
                          horizons=(3, 6, 12)):   # 필요하면 horizon 갯수 변경
    """
    horizons: tuple  ─ 예) (3, 6, 12)은 3-step, 6-step, 12-step
                       ※ n_pred보다 큰 값은 넣지 마세요.
    반환값: {horizon: (MAE, RMSE, WMAPE)}
    """
    model.eval()
    # horizon별 누적용 dict 초기화
    stats = {h: {"ae": [], "se": [], "sum_y": []} for h in horizons}

    with torch.no_grad():
        for x, y in data_iter:
            y_inv = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred_inv = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            print(f"y_inv.shape: {y_inv.shape}, y_pred_inv.shape: {y_pred_inv.shape}") # (B*N, T)
            for h in horizons:
                # 마지막 h-step 중 가장 마지막 시점 하나 (예: h=3 → t=3, h=6 → t=6)
                y_h     = y_inv[:, h-1]             # shape (B, N)
                y_pred_h = y_pred_inv[:, h-1]

                diff = np.abs(y_h - y_pred_h)
                stats[h]["ae"].extend(diff.reshape(-1))
                stats[h]["se"].extend((diff ** 2).reshape(-1))
                stats[h]["sum_y"].extend(y_h.reshape(-1))

    # 최종 metric 계산
    results = {}
    for h in horizons:
        ae   = np.array(stats[h]["ae"])
        se   = np.array(stats[h]["se"])
        sumy = np.array(stats[h]["sum_y"])
        mae  = ae.mean()
        rmse = np.sqrt(se.mean())
        wmape = ae.sum() / sumy.sum()
        results[h] = (mae, rmse, wmape)

    return results


class CustomDataset(Dataset):
    """
    PyTorch Dataset을 위한 커스텀 클래스.
    __len__과 __getitem__ 메소드를 구현해야 합니다.
    """
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        # 데이터셋의 전체 샘플 수를 반환합니다.
        return len(self.xs)

    def __getitem__(self, idx):
        # 주어진 인덱스(idx)에 해당하는 샘플을 반환합니다.
        return self.xs[idx], self.ys[idx]

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean