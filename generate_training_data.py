# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os

import numpy as np
import pandas as pd


def add_temporal_features(data, time_range, add_time_of_day, add_day_of_week):
    """
    원본 데이터에 주기성을 고려한 시간 관련 특징(Sine/Cosine)을 추가합니다.
    """
    if not add_time_of_day and not add_day_of_week:
        return datas

    # pd.date_range 사용을 권장합니다.
    timestamps = pd.date_range(
        start=time_range["start"],
        end=time_range["end"],
        freq='h',
        tz='UTC'
    )
    
    num_timesteps, num_nodes, _ = data.shape
    if len(timestamps) != num_timesteps:
        raise ValueError("Time range duration does not match the number of timesteps in data.")

    feature_list = [data]

    if add_time_of_day:
        hour_of_day = timestamps.hour
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
        
        # .to_numpy()를 추가하여 Pandas 객체를 NumPy 배열로 변환
        hour_sin_feature = np.tile(hour_sin.to_numpy()[:, np.newaxis, np.newaxis], (1, num_nodes, 1))
        hour_cos_feature = np.tile(hour_cos.to_numpy()[:, np.newaxis, np.newaxis], (1, num_nodes, 1))
        
        feature_list.extend([hour_sin_feature, hour_cos_feature])
        print("Added cyclical 'time_of_day' features (sin, cos).")

    if add_day_of_week:
        day_of_week = timestamps.dayofweek
        day_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        day_cos = np.cos(2 * np.pi * day_of_week / 7.0)
        
        # .to_numpy()를 추가하여 Pandas 객체를 NumPy 배열로 변환
        day_sin_feature = np.tile(day_sin.to_numpy()[:, np.newaxis, np.newaxis], (1, num_nodes, 1))
        day_cos_feature = np.tile(day_cos.to_numpy()[:, np.newaxis, np.newaxis], (1, num_nodes, 1))

        feature_list.extend([day_sin_feature, day_cos_feature])
        print("Added cyclical 'day_of_week' features (sin, cos).")

    return np.concatenate(feature_list, axis=-1)

def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets):
    """
    Graph Seq2Seq 모델을 위한 입출력 데이터 생성 함수
    """
    num_samples, num_nodes, num_features = data.shape
    x, y = [], []

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))

    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    print(f"Generated sequences - x shape: {x.shape}, y shape: {y.shape}")
    return x, y


def generate_train_val_test(args):
    """
    훈련/검증/테스트 세트를 생성하고 저장하는 메인 함수
    """
    print("Loading metadata and data files...")
    with open(args.metadata_filename, "r") as f:
        metadata = json.load(f)

    data_dict = np.load(args.input_data_filename)
    # npz 데이터 로드 및 검증
    print(f"Loaded NPZ file with keys: {list(data_dict.keys())}")
    data = data_dict["data"]
    print(f"Original data shape: {data.shape}")
    data = add_temporal_features(
        data,
        metadata["time_range"],
        args.add_time_of_day,
        args.add_day_of_week,
    )
    print(f"Data shape after adding temporal features: {data.shape}")

    x_offsets = np.arange(-(args.seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, args.seq_length_y + 1, 1)
    
    x, y = generate_graph_seq2seq_io_data(data, x_offsets, y_offsets)

    print("\nSplitting data...")
    num_samples = x.shape[0]
    train_samples = int(num_samples * args.train_ratio)
    val_samples = int(num_samples * args.val_ratio)
    test_samples = num_samples - train_samples - val_samples

    print(f"Data split (ratio: {args.train_ratio}:{args.val_ratio}:{args.test_ratio}):")
    print(f"  - Train: {train_samples}, Validation: {val_samples}, Test: {test_samples}")

    x_train, y_train = x[:train_samples], y[:train_samples]
    x_val, y_val = x[train_samples : train_samples + val_samples], y[train_samples : train_samples + val_samples]
    x_test, y_test = x[-test_samples:], y[-test_samples:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(f"Saving {cat} set - x: {_x.shape}, y: {_y.shape}")
        
        save_path = os.path.join(args.output_dir, f"{cat}.npz")
        np.savez_compressed(save_path, x=_x, y=_y, x_offsets=x_offsets.reshape(-1, 1), y_offsets=y_offsets.reshape(-1, 1))
    print(f"\nAll datasets saved successfully in '{args.output_dir}' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate train/val/test datasets for Graph Seq2Seq model.")
    
    parser.add_argument("--output_dir", type=str, default="data/ours", help="출력 디렉토리 경로")
    parser.add_argument("--input_data_filename", type=str, default="data/combined_data.npz", help="입력 데이터 파일 경로 (.npz)")
    parser.add_argument("--metadata_filename", type=str, default="data/metadata.json", help="메타데이터 파일 경로 (.json)")
    
    parser.add_argument("--seq_length_x", type=int, default=24*7, help="입력 시퀀스 길이")
    parser.add_argument("--seq_length_y", type=int, default=48, help="출력 시퀀스 길이")
    # 6:2:2 비율로 나누기 또는 7:1:2 비율로 나누기
    parser.add_argument("--train_ratio", type=float, default=0.6, help="훈련 세트 비율")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="검증 세트 비율")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="테스트 세트 비율")

    parser.add_argument("--add_time_of_day", action="store_true", help="하루 중 시간 특징(sin/cos) 추가 여부")
    parser.add_argument("--add_day_of_week", action="store_true", help="요일 특징(sin/cos) 추가 여부")

    args = parser.parse_args()
    
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Graph Seq2Seq 데이터 생성 시작 ===")
    generate_train_val_test(args)
    print("=== 데이터 생성 완료 ===")


# 예1
    # python generate_training_data.py \
    # --input_data_filename data/combined_data.npz \
    # --metadata_filename data/metadata.json \
    # --output_dir data/processed

# 예2
    # python generate_training_data.py \
    # --input_data_filename data/combined_data.npz \
    # --metadata_filename data/metadata.json \
    # --output_dir data/ours \
    # --train_ratio 0.7 \
    # --val_ratio 0.1 \
    # --test_ratio 0.2 \
    # --add_time_of_day \
    # --add_day_of_week
