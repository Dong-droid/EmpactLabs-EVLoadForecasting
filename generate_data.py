import pandas as pd
import numpy as np
import torch
import pickle
from math import radians, cos, sin, asin, sqrt
from scipy.sparse import coo_matrix, save_npz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import argparse
import os
from datetime import datetime


def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth (specified in decimal degrees)"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    if dlat == 0 and dlon == 0:
        return 0
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth's radius in kilometers
    
    return c * r


def load_and_preprocess_data(history_path, station_path, station_type_path=None):
    """Load and preprocess the raw data"""
    print("Loading raw data...")
    
    # Load data files
    hdf = pd.read_csv(history_path)
    sdf = pd.read_csv(station_path)
    
    # Merge history with station data
    mdf = pd.merge(hdf, sdf, how='left', left_on='_station', right_on='statId')
    
    # Load station type data if provided
    if station_type_path and os.path.exists(station_type_path):
        station_type = pd.read_csv(station_type_path)
        mdf = pd.merge(mdf, station_type, how='left', left_on='_station', right_on='_station')
    
    print(f"Initial data shape: {mdf.shape}")
    
    # Remove stations with null coordinates
    print("Removing stations with null coordinates...")
    null_stations = mdf[mdf.lat.isnull()]._station.unique()
    print(f"Stations with null coordinates: {len(null_stations)}")
    
    mdf = mdf[mdf.lat.notnull()]
    print(f"Data shape after removing null coordinates: {mdf.shape}")
    print(f"Number of unique stations: {mdf._station.nunique()}")
    
    return mdf, null_stations


def create_station_mapping(mdf):
    """Create mapping between station IDs and labels"""
    print("Creating station mapping...")
    
    # Create label mapping for stations
    labels, uniques = pd.factorize(mdf['_station'])
    mdf['location'] = labels
    
    # Create dictionaries for mapping
    label_to_station = {i: station for i, station in enumerate(uniques)}
    station_to_label = {station: i for i, station in enumerate(uniques)}
    
    print(f"Number of unique stations after mapping: {len(label_to_station)}")
    
    return mdf, label_to_station, station_to_label


def generate_demand_data(mdf):
    """Generate demand data (charging events per hour per station)"""
    print("Generating demand data...")
    
    # Convert to datetime and create hour column
    mdf['datetime'] = pd.to_datetime(mdf['startedAt'])
    mdf['hour'] = mdf['datetime'].dt.floor('H')
    
    # Aggregate charging events by hour and location
    agg = mdf.groupby(['hour', 'location']).size().reset_index(name='count')
    
    # Create full time range
    full_hours = pd.date_range(start=agg['hour'].min(), end=agg['hour'].max(), freq='H')
    all_locs = agg['location'].unique()
    
    # Create full index for all hour-location combinations
    full_index = pd.MultiIndex.from_product([full_hours, all_locs], names=['hour', 'location'])
    
    # Fill missing time slots with 0
    agg_full = agg.set_index(['hour', 'location']).reindex(full_index, fill_value=0).reset_index()
    
    # Create pivot table (rows: time, columns: locations)
    pivot_df = agg_full.pivot(index='hour', columns='location', values='count').fillna(0)
    
    print(f"Demand data shape: {pivot_df.shape}")
    print(f"Time range: {pivot_df.index.min()} to {pivot_df.index.max()}")
    
    # Convert to tensor
    data_tensor = torch.tensor(pivot_df.values, dtype=torch.float32)
    
    return pivot_df, data_tensor, agg_full


def generate_time_features(pivot_df):
    """Generate time features (day of week, hour of day)"""
    print("Generating time features...")
    
    # Extract time features from the pivot index
    time_index = pivot_df.index
    
    dayofweek = torch.tensor(time_index.weekday).reshape(-1, 1)  # 0=Monday, 6=Sunday
    hourofday = torch.tensor(time_index.hour).reshape(-1, 1)     # 0-23
    
    time_feat = torch.cat([dayofweek, hourofday], dim=-1)  # (timestep, 2)
    
    return time_feat


def generate_adjacency_matrix(mdf, label_to_station, distance_threshold=5.0, sigma=1.0):
    """Generate adjacency matrix based on geographical distance"""
    print("Generating adjacency matrix...")
    
    # Extract coordinates for each unique station
    lat_list = []
    lon_list = []
    
    for station_id in label_to_station.values():
        station_data = mdf[mdf['_station'] == station_id].iloc[0]
        lat_list.append(station_data.lat)
        lon_list.append(station_data.lng)
    
    print(f"Processing {len(lat_list)} stations...")
    
    # Generate edges based on distance threshold
    edges = []
    num_nodes = len(lat_list)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist = haversine(lat_list[i], lon_list[i], lat_list[j], lon_list[j])
                
                # Avoid division by zero
                if dist == 0:
                    dist = 0.0001
                
                if dist <= distance_threshold:
                    # Use Gaussian kernel for weight calculation
                    weight = np.exp(-dist / sigma)
                    edges.append((i, j, weight))
    
    print(f"Generated {len(edges)} edges")
    
    # Create DataFrame and save as CSV
    df_edges = pd.DataFrame(edges, columns=["from", "to", "weight"])
    
    # Create sparse adjacency matrix
    n_vertex = max(df_edges['from'].max(), df_edges['to'].max()) + 1
    adj_matrix = coo_matrix((df_edges['weight'], (df_edges['from'], df_edges['to'])), 
                           shape=(n_vertex, n_vertex))
    
    return df_edges, adj_matrix


def visualize_charging_patterns(mdf, station_ids, save_plots=True):
    """Visualize charging patterns for selected stations"""
    print(f"Visualizing charging patterns for stations: {station_ids}")
    
    for station_id in station_ids:
        if station_id not in mdf['alternateStatId'].values:
            print(f"Station {station_id} not found in data")
            continue
            
        station_data = mdf[mdf.alternateStatId == station_id].copy()
        station_data['startedAt'] = pd.to_datetime(station_data['startedAt'])
        
        # Aggregate by hour
        charging_counts = station_data.groupby(station_data['startedAt'].dt.floor('H')).size()
        
        # Create full time range and fill missing hours with 0
        full_range = pd.date_range(charging_counts.index.min(), charging_counts.index.max(), freq='H')
        charging_counts = charging_counts.reindex(full_range, fill_value=0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(15, 6))
        charging_counts.plot(ax=ax, linewidth=1)
        
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(charging_counts) // 20)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        
        plt.title(f'Charging Events per Hour: {station_id}', fontsize=14)
        plt.xlabel('Time')
        plt.ylabel('Charging Count')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'charging_pattern_{station_id}.png', dpi=300, bbox_inches='tight')
        
        plt.show()


def save_outputs(data_tensor, time_feat, pivot_df, adj_matrix, df_edges, label_to_station, 
                output_dir='./output'):
    """Save all generated data to files"""
    print("Saving outputs...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save demand data
    torch.save(data_tensor, os.path.join(output_dir, 'demand_data.pt'))
    pivot_df.to_csv(os.path.join(output_dir, 'demand_pivot.csv'), index=True)
    
    # Save time features
    torch.save(time_feat, os.path.join(output_dir, 'time_features.pt'))
    
    # Save adjacency matrix
    save_npz(os.path.join(output_dir, 'adj.npz'), adj_matrix)
    df_edges.to_csv(os.path.join(output_dir, 'distance.csv'), index=False)
    
    # Save station mapping
    with open(os.path.join(output_dir, 'label_to_station.pkl'), 'wb') as f:
        pickle.dump(label_to_station, f)
    
    # Create combined data with time features
    # Expand time features to match number of stations
    num_stations = data_tensor.shape[1]
    time_feat_expanded = time_feat.unsqueeze(1).repeat(1, num_stations, 1)  # (T, N, 2)
    data_expanded = data_tensor.unsqueeze(-1)  # (T, N, 1)
    
    # Combine demand data with time features
    combined_data = torch.cat([data_expanded, time_feat_expanded], dim=-1)  # (T, N, 3)
    torch.save(combined_data, os.path.join(output_dir, 'combined_data.pt'))
    
    print(f"All outputs saved to {output_dir}")
    print(f"Files saved:")
    print(f"  - demand_data.pt: Shape {data_tensor.shape}")
    print(f"  - time_features.pt: Shape {time_feat.shape}")
    print(f"  - combined_data.pt: Shape {combined_data.shape}")
    print(f"  - adj.npz: Shape {adj_matrix.shape}")
    print(f"  - distance.csv: {len(df_edges)} edges")
    print(f"  - label_to_station.pkl: {len(label_to_station)} stations")


def main():
    parser = argparse.ArgumentParser(description='Generate EV charging station data')
    parser.add_argument('--history_path', type=str, default='history.csv',
                       help='Path to history CSV file')
    parser.add_argument('--station_path', type=str, default='station.csv',
                       help='Path to station CSV file')
    parser.add_argument('--station_type_path', type=str, default=None,
                       help='Path to station type CSV file (optional)')
    parser.add_argument('--occupied_data_path', type=str, default=None,
                       help='Path to occupied rate data npz file (optional)')
    parser.add_argument('--distance_threshold', type=float, default=5.0, # 5km 이내의 충전소를 이웃으로 설정
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
    print(f"Apply BFS reordering: {args.apply_bfs_reordering}")
    if args.occupied_data_path:
        print(f"Occupied data file: {args.occupied_data_path}")
    print()
    
    # 1. Load and preprocess data
    mdf, null_stations = load_and_preprocess_data(
        args.history_path, args.station_path, args.station_type_path
    )
    
    # 2. Create station mapping
    mdf, label_to_station, station_to_label = create_station_mapping(mdf)
    
    # 3. Generate demand data
    pivot_df, data_tensor, agg_full = generate_demand_data(mdf)
    
    # 4. Generate time features
    time_feat = generate_time_features(pivot_df)
    
    # 5. Generate adjacency matrix
    df_edges, adj_matrix = generate_adjacency_matrix(
        mdf, label_to_station, args.distance_threshold, args.sigma
    )
    
    # Initialize variables for reordered data
    reordered_data = None
    old_to_new = None
    final_adj_matrix = adj_matrix
    final_label_to_station = label_to_station
    
    # 6. Load and process occupied rate data if provided
    if args.occupied_data_path and os.path.exists(args.occupied_data_path):
        print("Loading occupied rate data...")
        try:
            occupied_data = np.load(args.occupied_data_path)
            occupied_dict = {key: occupied_data[key] for key in occupied_data.files}
            print(f"Loaded occupied data with keys: {list(occupied_dict.keys())}")
            
            # Add demand data to the occupied data dictionary
            occupied_dict['demand'] = data_tensor.numpy()
            
            # 7. Apply BFS reordering if requested           
            reordered_data = occupied_dict
                
        except Exception as e:
            print(f"Error loading occupied data: {e}")
            print("Proceeding without occupied data...")
        # 7. Apply BFS reordering if requested (without occupied data)
    
    # 9. Regenerate time features (in case data shape changed)
    time_feat = generate_time_features(pivot_df)
    
    # 10. Visualize charging patterns (optional)
    if args.visualize:
        visualize_charging_patterns(mdf, args.sample_stations, save_plots=True)
    
    # 11. Save all outputs
    save_outputs(data_tensor, time_feat, pivot_df, final_adj_matrix, df_edges, 
                final_label_to_station, args.output_dir, reordered_data, old_to_new)
    
    print("=== Data Generation Complete ===")
    print(f"Summary:")
    print(f"  - Total stations: {len(final_label_to_station)}")
    print(f"  - Time steps: {data_tensor.shape[0]}")
    print(f"  - Final adjacency matrix shape: {final_adj_matrix.shape}")
    print(f"  - Excluded stations (null coordinates): {len(null_stations)}")


if __name__ == "__main__":
    main()