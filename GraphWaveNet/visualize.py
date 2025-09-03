import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from wp import *
import time
from scipy import sparse as sp
import util
import random
# from engine1 import trainer
# from stgcn_wp import STGCNWP
# from stgcn import STGCN

# ------------------------------ 설정 ----------------------------------------
num_timesteps_input  = 24*7 # 과거 24*7시간 (7일) 관측치로부터
num_timesteps_output = 48# horizon = 48 # 미래 48시간 (2일) 예측
#################### 모델 파일 경로 ##################################
base_best_ckpt_path       = 'garage/ours_base_exp801_best.pth' 
gat_best_ckpt_path       = 'garage/ours_gat_exp79_best.pth'
wp_best_ckpt_path       = "garage/ours_wp_exp51_best.pth"
# -------------------------------------------------------------------------
def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

##################학습 코드랑 동일하게 설정#############################
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda',help='')
parser.add_argument('--data',type=str,default='data/ours_doc_one_week',help='data path')
parser.add_argument('--adjdata',type=str,default='data/ours_doc_one_week/adj.npz',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=48,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=396,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=16,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/ours_base',help='save path')
parser.add_argument('--expid',type=int,default=100,help='experiment id')
parser.add_argument('--paitence', type=int, default=10, help='early stopping patience')
parser.add_argument('--num_heads', type=int, default=4, help='number of heads') # [2,4]로 튜닝 추천
parser.add_argument('--gat_dropout', type=int, default=0.1, help='number of heads') # [0.1, 0.3, 0.5]로 튜닝 추천
parser.add_argument('--target_feature_index', type=int, default=1, help='target feature index') # 0: occupancy, 1:demand, 2:has_fast_charger, 3:has_slow_charger
args = parser.parse_args()
# -------------------------------------------------------------------------

def extract_and_predict_from_loader(
    nets: dict,
    dataloader,
    device: str = 'cpu',
    max_batches: int = None,
    scaler=None
):
    """
    dataloader.get_iterator() 로부터 (x, y) 배치를 받아,
    각 모델에 inference 한 결과를 쌓아서 반환합니다.

    Args:
      nets        : {'Base': model_base, 'GAT': model_gat, ...}
      dataloader  : util.DataLoader (get_iterator() 지원)
      device      : 'cuda:0' 등
      max_batches : None or int, 최대 몇 배치만 처리할지

    Returns:
      yhat  : dict[name -> torch.Tensor(N, V, T_out)]
      realy : torch.Tensor(N, V, T_out)
    """
    # 결과 저장용
    yhat = {name: [] for name in nets.keys()}
    realy_list = []

    for batch_idx, (x_batch, y_batch) in enumerate(dataloader.get_iterator()):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # ────────────────────────────────────────────────────────────
        # 1) 배치 데이터 전처리
        # x_batch, y_batch: numpy or tensor, shape=(B,1,V,T)
        x = torch.as_tensor(x_batch).float()  # (B,1,V,T)
        y = torch.as_tensor(y_batch).float()  # (B,1,V,T)

        # (B,1,V,T) → (B,V,T,1)
        y = y.transpose(1, 3)[:,args.target_feature_index,:,:]#.transpose(1, 2)  # (B,V,T,1)
        x_model = x.transpose(1, 3).to(device)
        # GT: (B,V,T)
        realy_list.append(y.squeeze())  # keep on CPU for plotting later

        # ────────────────────────────────────────────────────────────
        # 2) 모델별 inference
        for name, net in nets.items():
            net = net.to(device).eval()
            with torch.no_grad():
                out = net(x_model).transpose(1,3).cpu() #pred = scaler.inverse_transform(yhat[:,0,:,i])
                out = out[:,args.target_feature_index,:,:]
                # out: (B,1,V,T) → (B,V,T)
                yhat[name].append(out)

    # ────────────────────────────────────────────────────────────
    # 3) 리스트 합치기
    realy = torch.cat(realy_list, dim=0) # (N, V, T_out)
    for name in yhat:
        yhat[name] = torch.cat(yhat[name], dim=0) # (N, V, T_out)

    return yhat, realy

def quick_plot_comparison(
    sample_idx: int,
    node_idx: int,
    yhat: dict,
    realy: torch.Tensor,
    scaler,
    save_path: str,
    tick_stride: int = 4
):
    """
    Fixed version with proper custom StandardScaler inverse transform
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    T_out = realy.size(2)
    horizon = np.arange(1, T_out + 1)

    # Extract time series for specific sample and node
    gt_vec = realy[sample_idx, node_idx, :].cpu().numpy()  # (T,)
    
    pred_vecs = {}
    for name in yhat:
        pred_vecs[name] = yhat[name][sample_idx, node_idx, :].cpu().numpy()  # (T,)

    # Debug: print raw values before inverse transform
    print(f"Sample {sample_idx}, Node {node_idx}")
    print(f"GT raw range: [{gt_vec.min():.4f}, {gt_vec.max():.4f}]")
    for name, vec in pred_vecs.items():
        print(f"{name} raw range: [{vec.min():.4f}, {vec.max():.4f}]")

    # Apply custom StandardScaler inverse transform
    # The custom scaler uses: normalized = (x - mean) / std
    # So inverse is: x = normalized * std + mean
    
    # Check if scaler has the expected attributes
    if hasattr(scaler, 'mean') and hasattr(scaler, 'std'):
        print(f"Scaler mean: {scaler.mean:.4f}, std: {scaler.std:.4f}")
        
        # GT seems to be already denormalized (range 0-3), so don't transform it
        # Only transform predictions which are still normalized (range ~0.94-0.97)
        if gt_vec.min() >= 0 and gt_vec.max() > 2:  # GT appears denormalized
            print("GT appears already denormalized, keeping as is")
            gt_den = gt_vec
        else:
            print("GT appears normalized, applying inverse transform")
            gt_den = gt_vec * scaler.std + scaler.mean
        
        # Apply inverse transform to predictions: x = normalized * std + mean
        pred_dens = {}
        for name, vec in pred_vecs.items():
            pred_dens[name] = vec * scaler.std.item() + scaler.mean.item()
            
    elif hasattr(scaler, 'inverse_transform'):
        # If it has inverse_transform method, use it
        print("Using scaler's inverse_transform method")
        
        # Check if GT needs transformation
        if gt_vec.min() >= 0 and gt_vec.max() > 2:
            gt_den = gt_vec  # Already denormalized
        else:
            gt_den = scaler.inverse_transform(gt_vec)
        
        pred_dens = {}
        for name, vec in pred_vecs.items():
            pred_dens[name] = scaler.inverse_transform(vec)
    else:
        # Fallback: use normalized values
        print("Warning: Could not find scaler parameters, using normalized values")
        gt_den = gt_vec
        pred_dens = pred_vecs

    # Debug: print values after inverse transform
    print(f"GT denorm range: [{gt_den.min():.4f}, {gt_den.max():.4f}]")
    for name, vec in pred_dens.items():
        print(f"{name} denorm range: [{vec.min():.4f}, {vec.max():.4f}]")

    # Plot
    plt.figure(figsize=(12, 6))
    ticks = horizon[::tick_stride]
    labels = [f'+{h}h' for h in ticks]
    plt.xticks(ticks=ticks, labels=labels, rotation=45, ha='right')

    colors = dict(GT='black', Base='tab:blue', GAT='tab:green', WP='tab:orange')
    
    # Plot ground truth
    plt.plot(horizon, gt_den, label='GT', c=colors['GT'], linewidth=2)
    
    # Plot predictions
    for name, vec in pred_dens.items():
        color = colors.get(name, 'tab:red')
        plt.plot(horizon, vec, label=name, c=color, linewidth=1)

    plt.title(f'Sample {sample_idx}, Node {node_idx} — {T_out}-step Forecast')
    plt.xlabel('Horizon (h)')
    plt.ylabel('Value')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✔ Saved → {save_path}')

# 사용 예시 (메인 코드에서 호출할 때)
if __name__ == "__main__":
    # torch.manual_seed(7)
    # np.random.seed(7)
    set_env(7)

    # ---- 데이터 로드 & 분할 --------------------------------------------------
    device = torch.device(args.device)
    adj_mx = sp.load_npz(args.adjdata)
    # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_and_preprocess_dataset(args.data, adj_mx, args.batch_size, args.target_feature_index, apply_preprocessing=False)
    a, b, adj_mx = util.load_adj(args.adjdata, args.adjtype, dataloader['adj_mx'])      
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None
    # print(args
    # ---- 그래프 & 모델 -------------------------------------------------------
    from engine import trainer
    net_base_engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)
    from engine1 import trainer
    net_gat_engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, num_heads=args.num_heads, gat_dropout=args.gat_dropout)
    
    # -----------------------------------------------------------------------
    #  ♛  베스트 모델 로딩 & 테스트 평가  ♛
    # -----------------------------------------------------------------------
    print("\n=== Test with best validation checkpoint ===========================")
    net_gat_engine.model.load_state_dict(torch.load(gat_best_ckpt_path, map_location=args.device),strict=False)
    print("Success")
    net_base_engine.model.load_state_dict(torch.load(base_best_ckpt_path, map_location=args.device), strict=False)
    print("Success")
  
    # infer_end = time.time()
    te_in = dataloader['test_loader']
    te_tg = dataloader['y_test']
    # ---- 예측 시각화 예시 ---------------------------------------------------
    nets = {
    'Base': net_base_engine.model,
    'GAT':  net_gat_engine.model,
}
    # 배치 단위로 예측값 추출
    yhat, realy = extract_and_predict_from_loader(
    nets=nets,
    dataloader=dataloader['test_loader'],
    device=args.device,
    max_batches=None,  # None 이면 전체 배치
    scaler=scaler
)

    sample_idx = 0 # sample index in [0, 샘플 총 개수]
    node_idx = 0 # node index in [0, 노드 총 개수]
    # 단일 샘플/노드 플롯
    quick_plot_comparison(
        sample_idx=sample_idx,
        node_idx=node_idx,  
        yhat=yhat,
        realy=realy,
        scaler=scaler,
        save_path=f'visuals/compare_sample{sample_idx}_node{node_idx}.png'
    )
