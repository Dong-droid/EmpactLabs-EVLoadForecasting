# -*- coding: utf-8 -*-
"""
EV 충전소 데이터 생성 및 전처리 스크립트

이 스크립트는 EV 충전소 이력 데이터를 기반으로 시공간 예측 모델을 위한 데이터를 생성합니다.
- 충전 수요 데이터 생성 (시간별/충전소별)
- 시간 특성 데이터 생성 (요일, 시간)
- 지리적 인접성 기반 그래프 구조 생성
- 데이터 시각화 및 저장

"""

import json
import pandas as pd
import numpy as np
import torch
import os
import pickle
import argparse
from math import radians, sin, cos, sqrt, asin
from scipy.sparse import coo_matrix, save_npz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


def haversine(lat1, lon1, lat2, lon2):
    """
    두 지점 간의 대원거리를 계산하는 함수 (Haversine 공식)
    
    Parameters:
    -----------
    lat1, lon1 : float
        첫 번째 지점의 위도, 경도 (도 단위)
    lat2, lon2 : float  
        두 번째 지점의 위도, 경도 (도 단위)
        
    Returns:
    --------
    float
        두 지점 간의 거리 (킬로미터)
    """
    # 도를 라디안으로 변환
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine 공식 적용
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # 동일한 지점인 경우 거리는 0
    if dlat == 0 and dlon == 0:
        return 0
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # 지구 반지름 (킬로미터)
    
    return c * r


def load_and_preprocess_data(history_path, station_path, station_type_path=None):
    """
    원시 데이터를 로드하고 전처리하는 함수
    
    Parameters:
    -----------
    history_path : str
        충전 이력 데이터 CSV 파일 경로
    station_path : str
        충전소 정보 CSV 파일 경로  
    station_type_path : str, optional
        충전소 타입 정보 CSV 파일 경로
        
    Returns:
    --------
    mdf : pandas.DataFrame
        병합된 데이터프레임
    null_stations : numpy.ndarray
        좌표가 없는 충전소 ID 목록
    """
    print("Loading raw data...")
    
    # 데이터 파일 로드
    hdf = pd.read_csv(history_path)
    sdf = pd.read_csv(station_path)
    
    # 충전 이력과 충전소 정보 병합
    mdf = pd.merge(hdf, sdf, how='left', left_on='_station', right_on='statId')
    
    # 충전소 타입 데이터가 있는 경우 추가 병합
    if station_type_path and os.path.exists(station_type_path):
        station_type = pd.read_csv(station_type_path)
        mdf = pd.merge(mdf, station_type, how='left', left_on='_station', right_on='_station')
    
    print(f"Initial data shape: {mdf.shape}")
    
    # 좌표가 없는 충전소 제거
    print("Removing stations with null coordinates...")
    null_stations = mdf[mdf.lat.isnull()]._station.unique()
    print(f"Stations with null coordinates: {len(null_stations)}")
    
    mdf = mdf[mdf.lat.notnull()]
    print(f"Data shape after removing null coordinates: {mdf.shape}")
    print(f"Number of unique stations: {mdf._station.nunique()}")
    
    return mdf, null_stations


def create_station_mapping(mdf):
    """
    충전소 ID와 라벨 간의 매핑을 생성하는 함수
    
    Parameters:
    -----------
    mdf : pandas.DataFrame
        병합된 데이터프레임
        
    Returns:
    --------
    mdf : pandas.DataFrame
        location 컬럼이 추가된 데이터프레임
    label_to_station : dict
        라벨(정수)에서 충전소 ID로의 매핑
    station_to_label : dict
        충전소 ID에서 라벨(정수)로의 매핑
    """
    print("Creating station mapping...")
    
    # 충전소를 위한 라벨 매핑 생성
    labels, uniques = pd.factorize(mdf['_station'])
    mdf['location'] = labels
    
    # 매핑을 위한 딕셔너리 생성
    label_to_station = {i: station for i, station in enumerate(uniques)}
    station_to_label = {station: i for i, station in enumerate(uniques)}
    
    print(f"Number of unique stations after mapping: {len(label_to_station)}")
    
    return mdf, label_to_station, station_to_label


def generate_demand_data(mdf):
    """
    수요 데이터 생성 (시간당 충전소별 충전 이벤트 수)
    
    Parameters:
    -----------
    mdf : pandas.DataFrame
        전처리된 데이터프레임
        
    Returns:
    --------
    pivot_df : pandas.DataFrame
        피벗 테이블 형태의 수요 데이터 (시간 × 충전소)
    data_tensor : torch.Tensor
        텐서 형태의 수요 데이터
    agg_full : pandas.DataFrame
        집계된 전체 데이터
    """
    print("Generating demand data...")
    
    # datetime 변환 및 시간 컬럼 생성
    mdf['datetime'] = pd.to_datetime(mdf['startedAt'])
    mdf['hour'] = mdf['datetime'].dt.floor('H')  # 시간 단위로 내림
    
    # 시간별, 위치별 충전 이벤트 집계
    agg = mdf.groupby(['hour', 'location']).size().reset_index(name='count')
    
    # 전체 시간 범위 생성
    full_hours = pd.date_range(start=agg['hour'].min(), end=agg['hour'].max(), freq='H')
    all_locs = agg['location'].unique()
    
    # 모든 시간-위치 조합을 위한 전체 인덱스 생성
    full_index = pd.MultiIndex.from_product([full_hours, all_locs], names=['hour', 'location'])
    
    # 누락된 시간대를 0으로 채움
    agg_full = agg.set_index(['hour', 'location']).reindex(full_index, fill_value=0).reset_index()
    
    # 피벗 테이블 생성 (행: 시간, 열: 충전소)
    pivot_df = agg_full.pivot(index='hour', columns='location', values='count').fillna(0)
    
    print(f"Demand data shape: {pivot_df.shape}")
    print(f"Time range: {pivot_df.index.min()} to {pivot_df.index.max()}")
    
    # 텐서로 변환
    data_tensor = torch.tensor(pivot_df.values, dtype=torch.float32)
    
    return pivot_df, data_tensor, agg_full


def generate_time_features(pivot_df):
    """
    시간 특성 생성 (요일, 시간)
    
    Parameters:
    -----------
    pivot_df : pandas.DataFrame
        피벗 테이블 형태의 수요 데이터
        
    Returns:
    --------
    time_feat : torch.Tensor
        시간 특성 텐서 (timestep, 2) - [요일, 시간]
    """
    print("Generating time features...")
    
    # 피벗 인덱스에서 시간 특성 추출
    time_index = pivot_df.index
    
    # 요일 (0=월요일, 6=일요일)
    dayofweek = torch.tensor(time_index.weekday).reshape(-1, 1)
    # 시간 (0-23)
    hourofday = torch.tensor(time_index.hour).reshape(-1, 1)
    
    # 시간 특성 결합
    time_feat = torch.cat([dayofweek, hourofday], dim=-1)  # (timestep, 2)
    
    return time_feat


def generate_adjacency_matrix(mdf, label_to_station, distance_threshold=5.0, sigma=1.0):
    """
    지리적 거리를 기반으로 인접성 행렬 생성
    
    Parameters:
    -----------
    mdf : pandas.DataFrame
        전처리된 데이터프레임
    label_to_station : dict
        라벨에서 충전소 ID로의 매핑
    distance_threshold : float, default=5.0
        인접성 판단 기준 거리 (킬로미터)
    sigma : float, default=1.0
        가우시안 커널의 시그마 파라미터
        
    Returns:
    --------
    df_edges : pandas.DataFrame
        엣지 정보 (from, to, weight)
    adj_matrix : scipy.sparse.coo_matrix
        희소 인접성 행렬
    """
    print("Generating adjacency matrix...")
    
    # 각 고유 충전소의 좌표 추출
    lat_list = []
    lon_list = []
    
    for station_id in label_to_station.values():
        station_data = mdf[mdf['_station'] == station_id].iloc[0]
        lat_list.append(station_data.lat)
        lon_list.append(station_data.lng)
    
    print(f"Processing {len(lat_list)} stations...")
    
    # 거리 임계값 기반으로 엣지 생성
    edges = []
    num_nodes = len(lat_list)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # 두 충전소 간의 거리 계산
                dist = haversine(lat_list[i], lon_list[i], lat_list[j], lon_list[j])
                
                # 0으로 나누는 것을 방지
                if dist == 0:
                    dist = 0.0001
                
                # 임계값 이내인 경우에만 엣지 생성
                if dist <= distance_threshold:
                    # 가우시안 커널을 사용한 가중치 계산
                    weight = np.exp(-dist / sigma)
                    edges.append((i, j, weight))
    
    print(f"Generated {len(edges)} edges")
    
    # DataFrame 생성 및 CSV로 저장 준비
    df_edges = pd.DataFrame(edges, columns=["from", "to", "weight"])
    
    # 희소 인접성 행렬 생성
    n_vertex = max(df_edges['from'].max(), df_edges['to'].max()) + 1
    adj_matrix = coo_matrix((df_edges['weight'], (df_edges['from'], df_edges['to'])), 
                           shape=(n_vertex, n_vertex))
    
    return df_edges, adj_matrix


def visualize_charging_patterns(mdf, station_ids, save_plots=True):
    """
    선택된 충전소들의 충전 패턴을 시각화하는 함수
    
    Parameters:
    -----------
    mdf : pandas.DataFrame
        전처리된 데이터프레임
    station_ids : list
        시각화할 충전소 ID 목록
    save_plots : bool, default=True
        플롯을 파일로 저장할지 여부
    """
    print(f"Visualizing charging patterns for stations: {station_ids}")
    
    for station_id in station_ids:
        # alternateStatId 컬럼이 있는지 확인 (없으면 _station 사용)
        id_column = 'alternateStatId' if 'alternateStatId' in mdf.columns else '_station'
        
        if station_id not in mdf[id_column].values:
            print(f"Station {station_id} not found in data")
            continue
            
        # 해당 충전소 데이터 필터링
        station_data = mdf[mdf[id_column] == station_id].copy()
        station_data['startedAt'] = pd.to_datetime(station_data['startedAt'])
        
        # 시간별 집계
        charging_counts = station_data.groupby(station_data['startedAt'].dt.floor('H')).size()
        
        # 전체 시간 범위 생성 및 누락된 시간을 0으로 채움
        full_range = pd.date_range(charging_counts.index.min(), charging_counts.index.max(), freq='H')
        charging_counts = charging_counts.reindex(full_range, fill_value=0)
        
        # 플롯 생성
        fig, ax = plt.subplots(figsize=(15, 6))
        charging_counts.plot(ax=ax, linewidth=1)
        
        # 축 설정
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(charging_counts) // 20)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        
        plt.title(f'Charging Events per Hour: {station_id}', fontsize=14)
        plt.xlabel('Time')
        plt.ylabel('Charging Count')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 파일로 저장
        if save_plots:
            plt.savefig(f'charging_pattern_{station_id}.png', dpi=300, bbox_inches='tight')
        
        plt.show()


def save_outputs(data_tensor, time_feat, pivot_df, adj_matrix, df_edges, label_to_station, 
                output_dir='./output', reordered_data=None, old_to_new=None):
    """
    생성된 모든 데이터를 파일로 저장하는 함수
    
    Parameters:
    -----------
    data_tensor : torch.Tensor
        수요 데이터 텐서
    time_feat : torch.Tensor  
        시간 특성 텐서
    pivot_df : pandas.DataFrame
        피벗 테이블
    adj_matrix : scipy.sparse.coo_matrix
        인접성 행렬
    df_edges : pandas.DataFrame
        엣지 정보
    label_to_station : dict
        라벨-충전소 매핑
    output_dir : str, default='./output'
        출력 디렉토리
    reordered_data : dict, optional
        재정렬된 데이터
    old_to_new : dict, optional
        재정렬 매핑
    """
    print("Saving outputs...")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 수요 데이터 저장
    torch.save(data_tensor, os.path.join(output_dir, 'demand_data.pt'))
    pivot_df.to_csv(os.path.join(output_dir, 'demand_pivot.csv'), index=True)
    
    # 시간 특성 저장
    torch.save(time_feat, os.path.join(output_dir, 'time_features.pt'))
    
    # 인접성 행렬 저장
    save_npz(os.path.join(output_dir, 'adj.npz'), adj_matrix)
    df_edges.to_csv(os.path.join(output_dir, 'distance.csv'), index=False)
    
    # 충전소 매핑 저장
    with open(os.path.join(output_dir, 'label_to_station.pkl'), 'wb') as f:
        pickle.dump(label_to_station, f)
    
    # 시간 특성과 수요 데이터를 결합한 데이터 생성
    # 시간 특성을 충전소 수만큼 확장
    num_stations = data_tensor.shape[1]
    time_feat_expanded = time_feat.unsqueeze(1).repeat(1, num_stations, 1)  # (T, N, 2)
    data_expanded = data_tensor.unsqueeze(-1)  # (T, N, 1)
    
    # 수요 데이터와 시간 특성 결합
    combined_data = torch.cat([data_expanded, time_feat_expanded], dim=-1)  # (T, N, 3)
    torch.save(combined_data, os.path.join(output_dir, 'combined_data.pt'))
    
    # 재정렬된 데이터가 있는 경우 저장
    if reordered_data is not None:
        np.savez(os.path.join(output_dir, 'reordered_data.npz'), **reordered_data)
        
    if old_to_new is not None:
        with open(os.path.join(output_dir, 'old_to_new_mapping.pkl'), 'wb') as f:
            pickle.dump(old_to_new, f)
    
    print(f"All outputs saved to {output_dir}")
    print(f"Files saved:")
    print(f"  - demand_data.pt: Shape {data_tensor.shape}")
    print(f"  - time_features.pt: Shape {time_feat.shape}")
    print(f"  - combined_data.pt: Shape {combined_data.shape}")
    print(f"  - adj.npz: Shape {adj_matrix.shape}")
    print(f"  - distance.csv: {len(df_edges)} edges")
    print(f"  - label_to_station.pkl: {len(label_to_station)} stations")
    if reordered_data is not None:
        print(f"  - reordered_data.npz: Reordered data with keys {list(reordered_data.keys())}")


def main():
    """
    메인 실행 함수
    """
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='Generate EV charging station data')
    parser.add_argument('--history_path', type=str, default='./history.csv',
                       help='Path to history CSV file')
    parser.add_argument('--station_path', type=str, default='station.csv',
                       help='Path to station CSV file')
    parser.add_argument('--station_type_path', type=str, default=None,
                       help='Path to station type CSV file (optional)')
    parser.add_argument('--occupied_data_path', type=str, default=None,
                       help='Path to occupied rate data npz file (optional)')
    parser.add_argument('--distance_threshold', type=float, default=5.0,
                       help='Distance threshold for adjacency matrix (km)')
    parser.add_argument('--sigma', type=float, default=1.0,
                       help='Sigma parameter for Gaussian kernel')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for generated files')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--sample_stations', nargs='+', default=["HM001132", "PC000302", "PI203005"],
                       help='Sample station IDs for visualization')
    
    args = parser.parse_args()

    print("=== EV Charging Station Data Generation ===")
    print(f"History file: {args.history_path}")
    print(f"Station file: {args.station_path}")
    print(f"Distance threshold: {args.distance_threshold} km")
    print(f"Output directory: {args.output_dir}")
    if args.occupied_data_path:
        print(f"Occupied data file: {args.occupied_data_path}")
    print()
    
    try:
        # 1. 데이터 로드 및 전처리
        mdf, null_stations = load_and_preprocess_data(
            args.history_path, args.station_path, args.station_type_path
        )
        
        # 2. 충전소 매핑 생성
        mdf, label_to_station, station_to_label = create_station_mapping(mdf)
        
        # 3. 수요 데이터 생성
        pivot_df, data_tensor, agg_full = generate_demand_data(mdf)
        
        # 4. 시간 특성 생성
        time_feat = generate_time_features(pivot_df)
        
        # 5. 인접성 행렬 생성
        df_edges, adj_matrix = generate_adjacency_matrix(
            mdf, label_to_station, args.distance_threshold, args.sigma
        )
        
        # 재정렬 관련 변수 초기화
        reordered_data = None
        old_to_new = None
        final_adj_matrix = adj_matrix
        final_label_to_station = label_to_station
        
        # 6. 점유율 데이터 로드 및 처리 (선택사항)
        if args.occupied_data_path and os.path.exists(args.occupied_data_path):
            print("Loading occupied rate data...")
            try:
                occupied_data = np.load(args.occupied_data_path)
                occupied_dict = {key: occupied_data[key] for key in occupied_data.files}
                print(f"Loaded occupied data with keys: {list(occupied_dict.keys())}")
                
                # 수요 데이터를 점유율 데이터 딕셔너리에 추가
                occupied_dict['demand'] = data_tensor.numpy()
                reordered_data = occupied_dict
                
            except Exception as e:
                print(f"Error loading occupied data: {e}")
                print("Proceeding without occupied data...")
        
        # 7. 시간 특성 재생성 (데이터 형태가 변경된 경우를 대비)
        time_feat = generate_time_features(pivot_df)
        
        # 8. 충전 패턴 시각화 (선택사항)
        if args.visualize:
            visualize_charging_patterns(mdf, args.sample_stations, save_plots=True)
        
        # 9. 모든 출력 저장
        save_outputs(data_tensor, time_feat, pivot_df, final_adj_matrix, df_edges, 
                    final_label_to_station, args.output_dir, reordered_data, old_to_new)
        
        print("=== Data Generation Complete ===")
        print(f"Summary:")
        print(f"  - Total stations: {len(final_label_to_station)}")
        print(f"  - Time steps: {data_tensor.shape[0]}")
        print(f"  - Final adjacency matrix shape: {final_adj_matrix.shape}")
        print(f"  - Excluded stations (null coordinates): {len(null_stations)}")
        
    except Exception as e:
        print(f"Error occurred during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


#  기본 실행
# python ev_charging_data_generator.py

# # 커스텀 설정으로 실행
# python ev_charging_data_generator.py \
#     --history_path ./data/history.csv \
#     --station_path ./data/station.csv \
#     --distance_threshold 10.0 \
#     --output_dir ./results \
#     --visualize
