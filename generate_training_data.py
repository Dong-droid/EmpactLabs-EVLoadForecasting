# -*- coding: utf-8 -*-
"""
Graph Seq2Seq 모델을 위한 시계열 데이터 생성기

이 스크립트는 시공간 데이터를 Graph Seq2Seq 모델에 적합한 형태로 변환합니다.
- NPZ 또는 CSV 파일에서 데이터 로드
- 시간 특성 추가 (시간, 요일)
- 훈련/검증/테스트 세트로 분할
- Sequence-to-sequence 형태로 데이터 구성

주요 기능:
- 다변량 시계열 데이터 처리
- 시간 오프셋 기반 입력/출력 시퀀스 생성
- 시간적/공간적 특성 추가

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Graph Seq2Seq 모델을 위한 입출력 데이터 생성 함수
    
    시계열 데이터를 슬라이딩 윈도우 방식으로 분할하여 
    입력 시퀀스(x)와 출력 시퀀스(y)로 구성합니다.
    
    Parameters:
    -----------
    df : numpy.ndarray or pandas.DataFrame
        시계열 데이터 (timesteps, nodes, features) 또는 (timesteps, nodes)
    x_offsets : numpy.ndarray
        입력 시퀀스를 위한 시간 오프셋 배열 (음수, 과거 시점)
    y_offsets : numpy.ndarray  
        출력 시퀀스를 위한 시간 오프셋 배열 (양수, 미래 시점)
    add_time_in_day : bool, default=True
        하루 중 시간 정보를 특성으로 추가할지 여부
    add_day_in_week : bool, default=False
        주중 요일 정보를 특성으로 추가할지 여부
    scaler : object, optional
        데이터 정규화를 위한 스케일러 (현재 미사용)
        
    Returns:
    --------
    x : numpy.ndarray
        입력 데이터 (batch_size, input_length, num_nodes, input_dim)
    y : numpy.ndarray
        출력 데이터 (batch_size, output_length, num_nodes, output_dim)
        
    Notes:
    ------
    - df의 인덱스가 datetime 타입이어야 시간 특성 추가 가능
    - x_offsets는 음수, y_offsets는 양수여야 함
    - 최종 배치 크기는 사용 가능한 시간 윈도우 수에 의해 결정됨
    """
    
    # NPZ 파일에서 로드된 numpy 배열 처리
    num_samples, num_nodes, num_features = df.shape
    
    # 데이터를 4차원으로 확장 (samples, nodes, features, 1)
    data = np.expand_dims(df, axis=-1)
    feature_list = [data]
    
    # 시간 특성 추가 (하루 중 시간을 0~1로 정규화)
    if add_time_in_day:
        # df.index가 datetime이어야 함
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        # 모든 노드에 대해 시간 정보를 타일링 (시간, 노드, 1)
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
        
    # 요일 특성 추가 (월요일=0, 일요일=6)
    if add_day_in_week:
        # df.index가 datetime이어야 함
        dow = df.index.dayofweek
        # 모든 노드에 대해 요일 정보를 타일링
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    # 모든 특성을 마지막 차원으로 결합
    data = np.concatenate(feature_list, axis=-1)
    
    # 입력/출력 시퀀스 생성을 위한 리스트 초기화
    x, y = [], []
    
    # 유효한 시간 범위 계산
    min_t = abs(min(x_offsets))  # 입력 시퀀스를 위한 최소 시간 인덱스
    max_t = abs(num_samples - abs(max(y_offsets)))  # 출력 시퀀스를 위한 최대 시간 인덱스 (exclusive)
    
    # 슬라이딩 윈도우 방식으로 시퀀스 생성
    for t in range(min_t, max_t):  # t는 현재 관찰의 마지막 인덱스
        # 입력 시퀀스: 과거 데이터 (x_offsets는 음수)
        x.append(data[t + x_offsets, ...])
        # 출력 시퀀스: 미래 데이터 (y_offsets는 양수)
        y.append(data[t + y_offsets, ...])
        
    # 리스트를 numpy 배열로 변환
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    
    print(f"Generated sequences - x shape: {x.shape}, y shape: {y.shape}")
    
    # 불필요한 차원 제거
    x = x.squeeze()
    y = y.squeeze()
    
    return x, y


def generate_train_val_test(args):
    """
    훈련/검증/테스트 세트를 생성하고 저장하는 메인 함수
    
    Parameters:
    -----------
    args : argparse.Namespace
        명령행 인수를 포함한 설정 객체
        
    주요 처리 과정:
    1. 데이터 파일 로드 (NPZ 또는 CSV)
    2. 시간 오프셋 설정
    3. 입출력 시퀀스 생성
    4. 훈련/검증/테스트로 분할
    5. NPZ 파일로 저장
    """
    
    # 시퀀스 길이 설정
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    
    # NPZ 파일 처리
    if args.traffic_df_filename.endswith('.npz'):
        print("Loading NPZ file...")
        
        # NPZ 파일 로드 (다중 배열 포함 가능)
        data_dict = np.load(args.traffic_df_filename)
        print(f"Available keys in NPZ file: {list(data_dict.keys())}")
        
        # 각 데이터 타입별로 로드
        occupancy = data_dict['occupied_rate']  # 점유율 데이터
        demand = data_dict['demand']  # 수요 데이터
        
        # 선택적 특성들 (있는 경우에만 로드)
        has_fast_charger = data_dict['has_fast_charger'] if 'has_fast_charger' in data_dict else None
        has_slow_charger = data_dict['has_slow_charger'] if 'has_slow_charger' in data_dict else None
        
        print(f"Loaded data shapes:")
        print(f"  - Occupancy: {occupancy.shape}")
        print(f"  - Demand: {demand.shape}")
        print(f"  - Has fast charger: {has_fast_charger.shape if has_fast_charger is not None else 'N/A'}")
        print(f"  - Has slow charger: {has_slow_charger.shape if has_slow_charger is not None else 'N/A'}")
        
        # 모든 특성을 마지막 차원으로 스택
        # (timesteps, nodes, features) 형태로 구성
        feature_list = [occupancy, demand]
        if has_fast_charger is not None:
            feature_list.append(has_fast_charger)
        if has_slow_charger is not None:
            feature_list.append(has_slow_charger)
            
        data = np.stack(feature_list, axis=-1)
        print(f"Combined data shape: {data.shape}")
        
    else:
        # CSV 파일 처리 (datetime 인덱스 포함)
        print("Loading CSV file...")
        df = pd.read_csv(args.traffic_df_filename, index_col=0, parse_dates=True)
        data = df
        datetime_index = df.index
        print(f"Loaded CSV data with shape: {df.shape}")
    
    # 시간 오프셋 설정
    # 입력 시퀀스: 과거 seq_length_x 시점부터 현재(0)까지
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    
    # 출력 시퀀스: y_start부터 seq_length_y 시점까지 예측
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    
    print(f"Time offsets:")
    print(f"  - Input (x_offsets): {x_offsets}")
    print(f"  - Output (y_offsets): {y_offsets}")
    
    # 입출력 시퀀스 생성
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        data,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,  # 시간 특성 추가 여부
        add_day_in_week=False,  # 요일 특성 추가 여부 (args.dow로 설정 가능)
    )

    print(f"Final data shapes - x: {x.shape}, y: {y.shape}")
    
    # 훈련/검증/테스트 세트 분할 (6:2:2 비율)
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)    # 테스트: 20%
    num_train = round(num_samples * 0.6)   # 훈련: 60% 
    num_val = num_samples - num_test - num_train  # 검증: 나머지 (약 20%)
    
    print(f"Data split:")
    print(f"  - Total samples: {num_samples}")
    print(f"  - Train samples: {num_train}")
    print(f"  - Validation samples: {num_val}")
    print(f"  - Test samples: {num_test}")
    
    # 시간 순서대로 분할 (과거 -> 현재 -> 미래)
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    # 각 세트별로 NPZ 파일 저장
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(f"{cat.capitalize()} set - x: {_x.shape}, y: {_y.shape}")
        
        # 압축된 NPZ 파일로 저장
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,  # 입력 시퀀스
            y=_y,  # 출력 시퀀스
            # 오프셋 정보도 함께 저장 (모델 학습 시 참고용)
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
        print(f"Saved {cat}.npz to {args.output_dir}")


if __name__ == "__main__":
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='Generate train/val/test datasets for Graph Seq2Seq model')
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/ours_doc_one_week", 
        help="출력 디렉토리 경로"
    )
    
    parser.add_argument(
        "--traffic_df_filename", 
        type=str, 
        default="data/ours_doc_one_week/adj.npz", 
        help="원시 시계열 데이터 파일 경로 (NPZ 또는 CSV)"
    )
    
    parser.add_argument(
        "--seq_length_x", 
        type=int, 
        default=24*7,  # 1주일 = 168시간
        help="입력 시퀀스 길이 (시간 단위). 기본값: 1주일"
    )
    
    parser.add_argument(
        "--seq_length_y", 
        type=int, 
        default=48,  # 2일 = 48시간
        help="출력 시퀀스 길이 (시간 단위). 기본값: 2일"
    )
    
    parser.add_argument(
        "--y_start", 
        type=int, 
        default=1, 
        help="예측 시작 시점 (현재 시점으로부터 몇 시간 후). 기본값: 1시간 후"
    )
    
    parser.add_argument(
        "--dow", 
        action='store_true',
        help="요일 특성을 추가할지 여부 (현재 미사용)"
    )

    args = parser.parse_args()
    
    # 출력 디렉토리 존재 여부 확인 및 생성
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} 디렉토리가 이미 존재합니다. 덮어쓰시겠습니까? (y/n): ')).lower().strip()
        if reply[0] != 'y': 
            print("작업이 취소되었습니다.")
            exit()
    else:
        os.makedirs(args.output_dir)
        print(f"출력 디렉토리 생성: {args.output_dir}")
    
    print("=== Graph Seq2Seq 데이터 생성 시작 ===")
    print(f"설정:")
    print(f"  - 입력 데이터: {args.traffic_df_filename}")
    print(f"  - 출력 경로: {args.output_dir}")
    print(f"  - 입력 시퀀스 길이: {args.seq_length_x}시간")
    print(f"  - 출력 시퀀스 길이: {args.seq_length_y}시간")
    print(f"  - 예측 시작 시점: {args.y_start}시간 후")
    print()
    
    # 메인 데이터 생성 함수 실행
    generate_train_val_test(args)
    
    print("=== 데이터 생성 완료 ===")
    print(f"생성된 파일:")
    print(f"  - {args.output_dir}/train.npz")
    print(f"  - {args.output_dir}/val.npz") 
    print(f"  - {args.output_dir}/test.npz")
