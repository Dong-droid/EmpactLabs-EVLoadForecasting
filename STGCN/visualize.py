import logging
import os
import gc
import argparse 
import numpy as np
import torch
import warnings
from script import dataloader, utility
from model import models
import matplotlib.pyplot as plt
import pickle # load pred.pkl and visualize as GraphWaveNet prediction

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN Visualization')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=7, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='ours_doc_bfs', choices=['metr-la', 'pems-bay', 'pemsd7-m', 'METR-LA', 'ours', 'ours_doc','ours_doc_one_week'], help='dataset name, default as METR-LA')
    parser.add_argument('--n_his', type=int, default=48) # the number of time interval for history, default as 48, i.e., 24*7 hours 로 변경할 수 있음
    parser.add_argument('--n_pred', type=int, default=48, help='the number of time i nterval for prediction, default as 3')
    parser.add_argument('--time_intvl', type=int, default=60)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100, help='epochs, default as 1000')
    parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'nadamw', 'lion'], help='optimizer, default as nadamw')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads for GAT')
    parser.add_argument('--gat_dropout', type=float, default=0.3, help='dropout for GAT')
    parser.add_argument('--target_feature_index', type=int, default=1, help='예측할 타겟 피쳐 인덱스')
   
    # Model checkpoint paths
    parser.add_argument('--base_ckpt', type=str, default='./STGCN_BASE.pt', help='base model checkpoint path')
    parser.add_argument('--gat_ckpt', type=str, default='./STGCN_GAT_BEST', help='gat model checkpoint path')
    
    # Visualization parameters
    parser.add_argument('--max_batches', type=int, default=None, help='maximum number of batches to process for visualization')
    parser.add_argument('--save_dir', type=str, default='visuals', help='directory to save visualization results')
    parser.add_argument('--save_predictions', type=bool, default=False, help='whether to save prediction results as pickle')
    
    args = parser.parse_args()
    print('Visualization configs: {}'.format(args))

    # For stable experiment results
    utility.set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        gc.collect()
    args.device = device
    
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer
    blocks = []
    blocks.append([4])
    for l in range(args.stblock_num):
        blocks.append([64, 32, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([args.n_pred])
    
    return args, device, blocks

def load_and_prepare_data(data_path, batch_size):
    """데이터를 로드하고, 스케일링하며, DataLoader를 생성하는 함수"""
    data = {}
    # train, val, test 데이터를 .npz 파일에서 로드
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_path, category + '.npz'))
        data['x_' + category] = torch.from_numpy(cat_data['x']).float()
        data['y_' + category] = torch.from_numpy(cat_data['y']).float()

    # StandardScaler를 사용하여 데이터 정규화 (train set 기준)
    print(data['x_train'][...,0].mean(), data['x_train'][...,1].mean())
    scaler = utility.StandardScaler(mean=data['x_train'][..., 1].mean(), std=data['x_train'][..., 1].std())
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 1] = scaler.transform(data['x_' + category][..., 1])
        data['y_' + category][..., 1] = scaler.transform(data['y_' + category][..., 1])

    # PyTorch Dataset 객체 생성
    train_dataset = utility.CustomDataset(data['x_train'], data['y_train'])
    val_dataset = utility.CustomDataset(data['x_val'], data['y_val'])
    test_dataset = utility.CustomDataset(data['x_test'], data['y_test'])

    # PyTorch DataLoader 생성
    data['train_loader'] = utility.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data['val_loader'] = utility.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    data['test_loader'] = utility.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data['scaler'] = scaler
    
    print("데이터 로드 및 준비 완료:")
    print(f"  - Train: {data['x_train'].shape}, {data['y_train'].shape}")
    print(f"  - Val:   {data['x_val'].shape}, {data['y_val'].shape}")
    print(f"  - Test:  {data['x_test'].shape}, {data['y_test'].shape}")

    return data

def data_preparate(args, device):    
    adj, n_vertex = dataloader.load_adj(args.dataset)
    gso = utility.calc_gso(adj, args.gso_type)
    args.wp_gso = torch.from_numpy(gso.toarray().astype(dtype=np.float32)).to(device)
    gso = utility.calc_chebynet_gso(gso)  # Always use cheb for consistency
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data = load_and_prepare_data(os.path.join(dataset_path), args.batch_size)
    test_iter = data['test_loader']
    zscore = data['scaler']
    n_vertex = data['x_train'].shape[2]  # 충전소(정점) 개수  
    
    return n_vertex, zscore, test_iter, data

def load_models(args, blocks, n_vertex, device):
    """Load all three model variants"""
    models_dict = {}
    
    # Base model (cheb_graph_conv)
    try:
        args.graph_conv_type = 'cheb_graph_conv'
        base_model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
        base_model.load_state_dict(torch.load(args.base_ckpt, map_location=device))
        base_model.eval()
        models_dict['Base'] = base_model
        print(f"✓ Base model loaded from {args.base_ckpt}")
    except Exception as e:
        print(f"✗ Failed to load base model: {e}")
    try:
        args.graph_conv_type = 'gat'
        gat_model = models.STGCNChebGraphConvGAT(args, blocks, n_vertex).to(device)
        gat_model.load_state_dict(torch.load(args.gat_ckpt, map_location=device))
        gat_model.eval()
        models_dict['GAT'] = gat_model
        print(f"✓ GAT model loaded from {args.gat_ckpt}")
    except Exception as e:
        print(f"✗ Failed to load GAT model: {e}")

    return models_dict
def evaluate_metric(model,
                    data_iter,
                    scaler,
                    args=None):
    model.eval()
    with torch.no_grad():
        pred, true = None, None
        
        for x, y in data_iter:
            y = y.permute(0, 3, 1, 2)[:,1,:,:].to(args.device)
            x = x.permute(0, 3, 1, 2).to(args.device)
            y_pred_tensor = model(x)
            if len(y_pred_tensor.shape) > len(y.shape):
                y_pred_tensor = y_pred_tensor.squeeze()
            y_true = scaler.inverse_transform(y).cpu().numpy()
            y_pred = scaler.inverse_transform(y_pred_tensor).cpu().numpy()
            # y_pred = y_pred_tensor.cpu().numpy()
            if pred is None:
                pred = y_pred
                true = y_true
            else:
                if len(y_pred.shape) == 2 : y_pred = np.expand_dims(y_pred, axis=0)
                pred = np.concatenate((pred, y_pred), axis=0)
                true = np.concatenate((true, y_true), axis=0)
        print(pred.shape, true.shape)
        return pred, true
def extract_and_predict_from_loader(
    nets: dict,
    dataloader,
    device: str = 'cpu',
    max_batches: int = None,
    args=None,
    scaler=None
):
    """
    DataLoader로부터 (x, y) 배치를 받아, 각 모델에 inference한 결과를 반환합니다.
    train.py와 동일한 방식으로 처리합니다.
    """
    # 결과 저장용 - train.py의 evaluate_metric과 동일한 방식
    yhat = {name: None for name in nets.keys()}
    realy = None
    for name, net in nets.items():
        net = net.to(device).eval()
        pred, real = evaluate_metric(net, dataloader, scaler, use_mask=False, args=args)
        yhat[name] = pred
    return yhat, real

def quick_plot_comparison(
    sample_idx: int,
    node_idx: int,
    yhat: dict,
    realy: np.array,
    scaler,
    save_path: str,
    tick_stride: int = 4
):
    """
    특정 샘플과 노드에 대한 예측 결과를 시각화합니다.
    이미 inverse transform이 적용된 데이터를 사용합니다.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(type(realy), realy.shape)
    T_out = realy.shape[1]  # Time dimension
    horizon = np.arange(1, T_out + 1)
    print("Completed")

    # Extract time series for specific sample and node
    # 이미 denormalized된 데이터이므로 바로 사용
    gt_vec = realy[sample_idx, :, node_idx]#.numpy()  # (T,)
    print("Completed")
    
    pred_vecs = {}
    for name in yhat:
        pred_vecs[name] = yhat[name][sample_idx, :, node_idx] #.numpy()  # (T,)
    with open('./pred.pkl', 'rb') as f:
        pred_gwn = pickle.load(f)
    print(pred_gwn.shape)
    pred_vecs['GraphWaveNet'] = pred_gwn[sample_idx, node_idx, :]
    print("Completed")
    # Plot
    plt.figure(figsize=(12, 6))
    ticks = horizon[::tick_stride]
    labels = [f'+{h}h' for h in ticks]
    plt.xticks(ticks=ticks, labels=labels, rotation=45, ha='right')

    colors = dict(GT='black', Base='tab:blue', GAT='tab:green', WP='tab:orange', GraphWaveNet='tab:purple')
    
    # Plot ground truth
    plt.plot(horizon, gt_vec, label='GT', c=colors['GT'], linewidth=2)
    
    # Plot predictions
    for name, vec in pred_vecs.items():
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

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # WMAPE calculation with division by zero protection
    sum_abs_true = np.sum(np.abs(y_true))
    if sum_abs_true > 0:
        wmape = np.sum(np.abs(y_true - y_pred)) / sum_abs_true
    else:
        wmape = np.nan
    
    return mae, rmse, wmape

def evaluate_models(yhat, realy, scaler, model_names, args):
    """
    Evaluate all models and print results
    train.py와 동일한 시간 스텝별 평가 방식 사용
    """
    print("\n=== Model Evaluation Results (Time Step-wise) ===")
    
    # 각 모델별 시간 스텝별 결과 저장
    all_results = {}
    
    for name in model_names:
        if name not in yhat:
            continue
        
        pred = yhat[name] # [batch, time, nodes]
        real = realy       # [batch, time, nodes]
        
        # 이미 denormalized된 데이터이므로 바로 사용
        step_results = []
        
        # train.py와 동일하게 시간 스텝별로 평가
        for i in range(args.n_pred):
            pred_step = pred[:, i, :]  # [batch, nodes]
            real_step = real[:, i, :]  # [batch, nodes]
            
            mae, rmse, wmape = calculate_metrics(real_step, pred_step)
            step_results.append({'MAE': mae, 'RMSE': rmse, 'WMAPE': wmape})
            
            print(f"Time Step {i+1:2d} | {name:>4} | MAE: {mae:.6f} | RMSE: {rmse:.6f} | WMAPE: {wmape:.8f}")
        
            # 평균 계산 (train.py와 동일)
        avg_mae = np.mean([result['MAE'] for result in step_results])
        avg_rmse = np.mean([result['RMSE'] for result in step_results])
        avg_wmape = np.mean([result['WMAPE'] for result in step_results])
        
        all_results[name] = {
            'avg_MAE': avg_mae,
            'avg_RMSE': avg_rmse, 
            'avg_WMAPE': avg_wmape
        }
        
        print(f"Average      | {name:>4} | MAE: {avg_mae:.6f} | RMSE: {avg_rmse:.6f} | WMAPE: {avg_wmape:.8f}")
        print("-" * 80)
    
    return all_results

if __name__ == "__main__":
    # Logging setup
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Get parameters
    args, device, blocks = get_parameters()
    
    # Prepare data
    print("=== Loading and preparing data ===")
    n_vertex, zscore, test_iter, data = data_preparate(args, device)
    
    # Load models
    print("\n=== Loading models ===")
    nets = load_models(args, blocks, n_vertex, device)
    
    if not nets:
        print("No models loaded successfully. Please check checkpoint paths.")
        exit(1)
    
    # Extract predictions
    print(f"\n=== Extracting predictions (max_batches: {args.max_batches}) ===")
    yhat, realy = extract_and_predict_from_loader(
        nets=nets,
        dataloader=test_iter,
        device=device,
        max_batches=args.max_batches,
        args=args,
        scaler=zscore
    )
    
    # Evaluate models (train.py와 동일한 방식)
    results = evaluate_models(yhat, realy, zscore, list(nets.keys()), args)
    
    
    # Create visualization directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Generate sample visualizations
    print(f"\n=== Generating visualizations ===")
    sample_indices = [0,48,95,142, 231, 327]  # Sample indices to visualize
    node_indices = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325,115, 363, 282, 197]  # Node indices to visualize
    # node_indices =np.random.randint(0, 396, 10)
    for sample_idx in sample_indices:
        for node_idx in node_indices:
                quick_plot_comparison(
                sample_idx=sample_idx,
                node_idx=node_idx,
                yhat=yhat,
                realy=realy,
                scaler=zscore,
                save_path=f'{args.save_dir}/compare_node{node_idx}_sample{sample_idx}.png'
            )
                # except Exception as e:
                #     print(f"✗ Failed to create visualization for sample {sample_idx}, node {node_idx}: {e}")
    
    print(f"\n=== Visualization complete! Results saved in '{args.save_dir}' directory ===")