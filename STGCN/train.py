import logging
import os
import gc
import argparse 
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import warnings
from script import dataloader, utility, earlystopping, opt
from model.wp import adjust_wp_hparams, collect_wp_modules, get_param_groups
from model import models
import tqdm
import wandb
from thop import profile



def get_parameters():
    """실험에 필요한 하이퍼파라미터와 환경을 설정하는 함수"""
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='CUDA 사용 여부')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=7, help='랜덤 시드 고정 (재현성)')
    parser.add_argument('--dataset', type=str, default='ours_doc_one_week', 
                        choices=['metr-la', 'pems-bay', 'pemsd7-m', 'METR-LA', 'ours', 'ours_doc', 'ours_doc_one_week']) 
    parser.add_argument('--n_his', type=int, default=24*7, help='입력 시계열 길이')
    parser.add_argument('--n_pred', type=int, default=48, help='예측할 타임스텝 수')
    parser.add_argument('--time_intvl', type=int, default=60, help='타임스텝 간격 (분)')
    parser.add_argument('--Kt', type=int, default=3, help='시간 축 커널 크기')
    parser.add_argument('--stblock_num', type=int, default=2, help='ST 블록 개수')
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'], help='활성 함수')
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2], help='공간 축 커널 크기')
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', 
                        choices=['cheb_graph_conv', 'graph_conv', 'wp', 'gat'], help='그래프 합성곱 방식')
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', 
                        choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'], help='그래프 시프트 연산자 유형')
    parser.add_argument('--enable_bias', type=bool, default=True, help='바이어스 사용 여부')
    parser.add_argument('--droprate', type=float, default=0.5, help='드롭아웃 비율')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='L2 가중치 감쇠')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100, help='에폭 수')
    parser.add_argument('--opt', type=str, default='adam', choices=['adam','adamw', 'nadamw', 'lion'], help='최적화 알고리즘')
    parser.add_argument('--step_size', type=int, default=10, help='LR 스케줄러 step size')
    parser.add_argument('--gamma', type=float, default=0.95, help='LR 스케줄러 감쇠율')
    parser.add_argument('--patience', type=int, default=10, help='Early Stopping patience 값')
    parser.add_argument('--target_feature_index', type=int, default=1, help='예측할 타겟 피쳐 인덱스')
    # 학습률과 배치사이즈는 튜닝을 추천 (lr [1e-3, 5e-4, 1e-4], batch_size [16, 32, 64])
    # GAT 파라미터 num_heads = (2,4), gat_dropout = (0~0.5) 튜닝 추천
    parser.add_argument('--num_heads', type=int, default=2, help='GAT multi-head attention 수')
    parser.add_argument('--gat_dropout', type=float, default=0.1, help='GAT dropout 비율')
    
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # 재현성을 위해 시드 고정
    utility.set_env(args.seed)

    # CUDA 또는 CPU 설정
    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device(args.device)
        torch.cuda.empty_cache() # GPU 캐시 비우기
    else:
        device = torch.device('cpu')
        gc.collect() # CPU 메모리 정리
    args.device = device

    # 출력 타임스텝 계산
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # ST 블록의 채널 크기 설정 (bottleneck 구조 사용)
    blocks = []
    blocks.append([4])  # 입력 채널 수, 피쳐 수 4개 (점유율, 수요, has_fast_charger, has_slow_charger) ## 피쳐 수 변경 시 수정 필요
    for l in range(args.stblock_num):
        blocks.append([64, 32, 64])  # bottleneck 구조
        # blocks.append([64, 16, 64])  # 더 작은 bottleneck도 가능
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([args.n_pred]) # 최종 출력 (예측 horizon)

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
    # 맞추고 싶은 feature가 1번 인덱스에 있다고 가정 지금 우리 데이터는 0번이 점유율, 1번이 수요, 2번이 has_fast_charger, 3번이 has_slow_charger 총 4개 
    scaler = utility.StandardScaler(mean=data['x_train'][..., args.target_feature_index].mean(), std=data['x_train'][..., args.target_feature_index].std()) # 1번 feature에 대해서만 스케일링 (정규화)
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., args.target_feature_index] = scaler.transform(data['x_' + category][..., args.target_feature_index]) # 1번 feature에 대해서만 스케일링 (정규화)
        data['y_' + category][..., args.target_feature_index] = scaler.transform(data['y_' + category][..., args.target_feature_index]) # 1번 feature에 대해서만 스케일링 (정규화)

        
    # PyTorch Dataset 객체 생성
    train_dataset = utility.CustomDataset(data['x_train'], data['y_train'])
    val_dataset = utility.CustomDataset(data['x_val'], data['y_val'])
    test_dataset = utility.CustomDataset(data['x_test'], data['y_test'])

    # PyTorch DataLoader 생성
    data['train_loader'] = utility.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data['val_loader'] = utility.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    data['test_loader'] = utility.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data['scaler'] = scaler
    
    # 평가를 위해 전체 데이터를 텐서로도 유지
    data['va_in'], data['va_tg'] = data['x_val'], data['y_val']
    data['te_in'], data['te_tg'] = data['x_test'], data['y_test']

    print("데이터 로드 및 준비 완료:")
    print(f"  - Train: {data['x_train'].shape}, {data['y_train'].shape}")
    print(f"  - Val:   {data['x_val'].shape}, {data['y_val'].shape}")
    print(f"  - Test:  {data['x_test'].shape}, {data['y_test'].shape}")

    # Log data statistics to wandb
    wandb.log({
        "data/train_samples": len(data['x_train']),
        "data/val_samples": len(data['x_val']),
        "data/test_samples": len(data['x_test']),
    })

    return data

def data_preparate(args, device):    
    adj, n_vertex = dataloader.load_adj(args.dataset)
    gso = utility.calc_gso(adj, args.gso_type)
    # if args.graph_conv_type == 'cheb_graph_conv':
    gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data = load_and_prepare_data(os.path.join(dataset_path), args.batch_size)
    train_iter = data['train_loader']
    val_iter = data['val_loader']
    test_iter = data['test_loader']
    zscore = data['scaler']
    n_vertex = data['x_train'].shape[2]  # 충전소(정점) 개수
    
    # Log graph statistics to wandb
    # wandb.log({
    #     "graph/n_vertex": n_vertex,
    #     "graph/adj_shape": adj.shape,
    #     "graph/gso_type": args.gso_type,
    #     "graph/conv_type": args.graph_conv_type
    # })
    
    return n_vertex, zscore, train_iter, val_iter, test_iter, data

def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    path = "./checkpoint/STGCN_" + args.dataset + ".pt" # 모델 저장 경로
    es = earlystopping.EarlyStopping(delta=0.0, 
                                     patience=args.patience, 
                                     verbose=True, 
                                     path=path)

    if args.graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    elif args.graph_conv_type == 'gat':
        model = models.STGCNChebGraphConvGAT(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)
    

    if args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "nadamw":
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, decoupled_weight_decay=True)
    elif args.opt == "lion":
        optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    else:
        raise ValueError(f'ERROR: The {args.opt} optimizer is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


    return loss, es, model, optimizer, scheduler, path

def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter):
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        
        # Training loop with progress bar
        train_pbar = tqdm.tqdm(train_iter, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, (x, y) in enumerate(train_pbar):
            x = x.permute(0, 3, 1, 2).to(device) 
            y = y.permute(0, 3, 1, 2)[:,args.target_feature_index,:,:].to(device) 
            optimizer.zero_grad()
            y_pred = model(x).squeeze()
            l = loss(y_pred, y)
            l.backward()
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            
            # Update progress bar
            train_pbar.set_postfix({'batch_loss': f'{l.item():.6f}'})
            
            # Log batch-level metrics every 100 batches
            if batch_idx % 100 == 0:
                wandb.log({
                    "batch/train_loss": l.item(),
                    "batch/learning_rate": optimizer.param_groups[0]['lr'],
                    "batch/epoch": epoch,
                    "batch/batch_idx": batch_idx
                })
        
        scheduler.step()
        val_loss = val(model, val_iter, loss, args.device)
        
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        
        # Calculate epoch metrics
        epoch_train_loss = l_sum / n
        current_lr = optimizer.param_groups[0]['lr']
        
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, current_lr, epoch_train_loss, val_loss, gpu_mem_alloc))

        # Log epoch metrics to wandb
        wandb.log({
            "epoch/train_loss": epoch_train_loss,
            "epoch/val_loss": val_loss.item(),
            "epoch/learning_rate": current_lr,
            "epoch/gpu_memory_mb": gpu_mem_alloc,
            "epoch/epoch": epoch + 1
        })

        es(val_loss, model)
        if es.early_stop:
            print("Early stopping")
            wandb.log({"training/early_stopped_epoch": epoch + 1})
            break
    
    # Log training completion
    wandb.log({
        "training/completed": True,
        "training/total_epochs": epoch + 1
    })

@torch.no_grad()
def val(model, val_iter, loss, device):
    model.eval()

    l_sum, n = 0.0, 0
    for x, y in val_iter:
        x = x.permute(0, 3, 1, 2).to(device) 
        y = y.permute(0, 3, 1, 2)[:,args.target_feature_index,:,:].to(device) 
        y_pred = model(x).squeeze()  
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

def evaluate_model(model, loss, data_iter, time_step=1, args=None):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            x = x.permute(0, 3, 1, 2).to(args.device) 
            y = y.permute(0, 3, 1, 2)[:,args.target_feature_index,time_step,:].to(args.device) 
            y_pred = model(x)[:,time_step,:] 
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse
def evaluate_metric(model,
                    data_iter,
                    scaler,
                    args=None):
    model.eval()
    with torch.no_grad():
        pred, true = None, None
        
        for x, y in data_iter:
            y = y.permute(0, 3, 1, 2)[:,args.target_feature_index,:,:].to(args.device)
            x = x.permute(0, 3, 1, 2).to(args.device)
            y_pred_tensor = model(x)
            if len(y_pred_tensor.shape) > len(y.shape):
                y_pred_tensor = y_pred_tensor.squeeze()
            y_true = scaler.inverse_transform(y).cpu().numpy()
            y_pred = scaler.inverse_transform(y_pred_tensor).cpu().numpy()
            if pred is None:
                pred = y_pred
                true = y_true
            else:
                if len(y_pred.shape) == 2 : y_pred = np.expand_dims(y_pred, axis=0)
                pred = np.concatenate((pred, y_pred), axis=0)
                true = np.concatenate((true, y_true), axis=0)

        return pred, true


def calculate_metrics(y_true, y_pred, use_mask=False, null_val=np.nan):
    """
    Calculate MAE, RMSE, and WMAPE metrics
    """
    if use_mask and null_val is not None:
        mask = (y_true != null_val)
        if not np.any(mask):  # If no valid values, return NaN
            return np.nan, np.nan, np.nan
        
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
    else:
        y_true_masked = y_true.flatten()
        y_pred_masked = y_pred.flatten()
    
    # Calculate metrics
    ae = np.abs(y_true_masked - y_pred_masked)
    se = (y_true_masked - y_pred_masked) ** 2
    
    MAE = np.mean(ae)
    RMSE = np.sqrt(np.mean(se))
    
    # WMAPE calculation with division by zero protection
    sum_abs_true = np.sum(np.abs(y_true_masked))
    if sum_abs_true > 0:
        WMAPE = np.sum(ae) / sum_abs_true
    else:
        WMAPE = np.nan
    
    return MAE, RMSE, WMAPE
@torch.no_grad() 
def test(zscore, loss, model, test_iter, args, path, data):
    # Load the trained model
    model.load_state_dict(torch.load(path, map_location=args.device))
    model.eval()
    
    # Get predictions and ground truth
    pred, real = evaluate_metric(model, test_iter, zscore, use_mask=False, args=args)
    print(pred.shape, real.shape)
    print(f"Prediction shape: {pred.shape}, Ground truth shape: {real.shape}")
    
    # Store all test results for summary
    test_results = {}
    
    # Calculate metrics for each time step
    for i in range(args.n_pred):
        
        if pred.shape[1] == args.n_pred:  # [batch, time, nodes]
            pred_step = pred[:, i, :]
            real_step = real[:, i, :]

        # Calculate metrics for this time step
        test_MAE, test_RMSE, test_WMAPE = calculate_metrics(
            real_step, pred_step, use_mask=False
        )
        
        print(f'Time Step {i+1} | Dataset {args.dataset:s} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')
        
        # Store for summary
        test_results[f'time_step_{i+1}'] = {
            'MAE': test_MAE,
            'RMSE': test_RMSE,
            'WMAPE': test_WMAPE
        }
    
    # Calculate and log average metrics across all time steps
    # avg_mse = np.mean([test_results[f'time_step_{i+1}']['MSE'] for i in range(args.n_pred)])
    avg_mae = np.mean([test_results[f'time_step_{i+1}']['MAE'] for i in range(args.n_pred)])
    avg_rmse = np.mean([test_results[f'time_step_{i+1}']['RMSE'] for i in range(args.n_pred)])
    avg_wmape = np.mean([test_results[f'time_step_{i+1}']['WMAPE'] for i in range(args.n_pred)])
    
    wandb.log({
        # "test/average/MSE": avg_mse,
        "test/average/MAE": avg_mae,
        "test/average/RMSE": avg_rmse,
        "test/average/WMAPE": avg_wmape,
    })
    print(f'Average Test Results | MAE: {avg_mae:.6f} | RMSE: {avg_rmse:.6f} | WMAPE: {avg_wmape:.8f}')
    
    # Create a summary table for wandb
    test_table = wandb.Table(columns=["Time Step",  "MAE", "RMSE", "WMAPE"])
    for i in range(args.n_pred):
        result = test_results[f'time_step_{i+1}']
    
    
    return avg_mae, avg_rmse, avg_wmape  # Return key metrics for grid search comparison

def init_wandb(args):
    """Initialize wandb with proper configuration"""
    config = vars(args) # is this right? 
    
    wandb.init(
        project="STGCN_ours",
        config=config,
        name=f"STGCN_{args.dataset}_{args.graph_conv_type}_seed{args.seed}",
        tags=[args.dataset, args.graph_conv_type, args.gso_type],
        notes=f"STGCN model training on {args.dataset} dataset with {args.graph_conv_type} convolution"
    )

if __name__ == "__main__":
    # Logging
    logging.basicConfig(level=logging.INFO)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args, device, blocks = get_parameters()
    
    # Initialize wandb
    init_wandb(args)
    
    n_vertex, zscore, train_iter, val_iter, test_iter, data = data_preparate(args, device)
    loss, es, model, optimizer, scheduler, path = prepare_model(args, blocks, n_vertex)
    
    try:
        train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter)
        test(zscore, loss, model, test_iter, args, path, data)
    except Exception as e:
        wandb.log({"error": str(e)})
        raise
    finally:
        # Finish wandb run
        wandb.finish()