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
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    # print(df.shape)
    # npz file is loaded as a numpy array.
    # if isinstance(df, np.ndarray):
    #     df = pd.DataFrame(df)
    # how to npz file is loaded as a numpy array.
    num_samples, num_nodes, num_features = df.shape
    # data = np.expand_dims(df.values, axis=-1) 
    data = np.expand_dims(df, axis=-1)
    print(data.shape)
    feature_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last obyservation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    print(x.shape, y.shape)
    x = x.squeeze()
    y= y.squeeze()
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    # df = pd.read_csv(args.traffic_df_filename)
    # how to load npz file as a numpy array?
    if args.traffic_df_filename.endswith('.npz'):
        data_dict = np.load(args.traffic_df_filename)
        print(data_dict.keys())
        occupancy = data_dict['occupied_rate']
        demand = data_dict['demand']
        has_fast_charger = data_dict['has_fast_charger'] if 'has_fast_charger' in data_dict else None
        has_slow_charger = data_dict['has_slow_charger'] if 'has_slow_charger' in data_dict else None
        print(f"Loaded data with shapes: occupancy {occupancy.shape}, demand {demand.shape}, "
              f"has_fast_charger {has_fast_charger.shape}, has_slow_charger {has_slow_charger.shape if has_slow_charger is not None else 'N/A'}")
        data = np.stack([occupancy, demand, has_fast_charger, has_slow_charger], axis=-1)
        print(occupancy.shape, demand.shape, data.shape)
        
    else:
        # Assume CSV with datetime index
        df = pd.read_csv(args.traffic_df_filename, index_col=0, parse_dates=True)
        data = df
        datetime_index = df.index
        print(f"Loaded CSV data with shape: {df.shape}")
    # 0 is the latest observed sample.y
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        data,
        # vel_3d,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False,#args.dow,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.15)
    num_train = round(num_samples * 0.7) # 0.6 for ours dataset
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/ours_doc_one_week", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="data/ours_doc_one_week/adj.npz", help="Raw traffic readings.",)
    parser.add_argument("--seq_length_x", type=int, default=24*7, help="Sequence Length. 1주일",)
    parser.add_argument("--seq_length_y", type=int, default=48, help="Sequence Length. 2일",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true',)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
