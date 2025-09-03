import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix


def load_adj(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()
    
    if dataset_name == 'metr-la' or dataset_name == 'METR-LA':
        n_vertex = 207
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'pemsd7-m':
        n_vertex = 228
    ######################## 수정 필요 ######################################
    elif dataset_name == 'ours_doc' or dataset_name == 'ours' or dataset_name == 'ours_doc_bfs' or dataset_name == "ours_doc_bfs_7":
        n_vertex = 396
    return adj, n_vertex

def load_data(dataset_name, len_train, len_val):
    dataset_path = 'EMPACTLABS/data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))
    print(vel.shape)
    # 데이터 분할
    train = vel[:len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    
    print(f"Train shape: {train.shape}, Val shape: {val.shape}, Test shape: {test.shape}")
    return train, val, test

def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data
    n_vertex = data.shape[1]

    len_record = len(data)
    num = len_record - n_his - n_pred
    
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)