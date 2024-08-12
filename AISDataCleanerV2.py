import pandas as pd
from zipfile import ZipFile
import os
import numpy as np
import pyproj
from scipy.spatial.distance import cdist
import interpol_lin
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371  # Radius of Earth in kilometers
    return r * c

transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4087")

cols_to_read = ['MMSI', 'Time', 'Longitude', 'Latitude','SOG','COG']
directory = r'D:\Vincent\datadl\backfill_2012-2022'
save_directory = r'D:\Vincent\Cleandata_valid_transform'
directory_lin = 'D:\\Vincent\\Cleandata_valid_transform\\transform_csv'
save_directory_lin = 'D:\\Vincent\\interpol_data_transf_lin'


def process_csv_file(csv_files):
    # Read CSV into DataFrame
    df = pd.read_csv(csv_files, usecols=cols_to_read)

    # Convert Latitude and Longitude to numeric, coerce errors to NaN
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    # Drop rows with NaN values in Latitude or Longitude
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Filter latitude and longitude within valid ranges
    df = df[(df['Latitude'] >= 30) & (df['Latitude'] <= 89.999)]
    df = df[(df['Longitude'] >= -179.999) & (df['Longitude'] <= 179.999)]

    # Convert Time column to datetime format
    df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d_%H%M%S')

    # Filter MMSI based on count threshold
    mmsi_counts = df['MMSI'].value_counts()
    mmsi_to_keep = mmsi_counts[mmsi_counts >= 10].index
    df_filtered = df[df['MMSI'].isin(mmsi_to_keep)]
    removed_data = df[~df['MMSI'].isin(mmsi_to_keep)]         
    
    # Transform coordinates using pyproj.Transformer
    transformed_coords = transformer.transform(df_filtered['Longitude'].values, df_filtered['Latitude'].values)
    df_filtered = df_filtered.assign(Transformed_Latitude=transformed_coords[1], Transformed_Longitude=transformed_coords[0])

    # Remove distant points based on distance threshold
    to_remove_loin = []
    for mmsi in df_filtered['MMSI'].unique():
        mmsi_data = df_filtered[df_filtered['MMSI'] == mmsi]
        
        # Compute Haversine distances between all pairs of points
        distances = np.zeros((len(mmsi_data), len(mmsi_data)))
        for i in range(len(mmsi_data)):
            for j in range(len(mmsi_data)):
                if i != j:
                    distances[i, j] = haversine(
                        mmsi_data.iloc[i]['Latitude'],
                        mmsi_data.iloc[i]['Longitude'],
                        mmsi_data.iloc[j]['Latitude'],
                        mmsi_data.iloc[j]['Longitude']
                    )
        
        np.fill_diagonal(distances, np.inf)
        
        min_distances = np.min(distances, axis=1)
        indices_to_remove_loin = np.where(min_distances > 0.5)[0]
        to_remove_loin.extend(mmsi_data.index[indices_to_remove_loin])

    removed_data = pd.concat([removed_data, df_filtered.loc[to_remove_loin]])
    df_filtered = df_filtered.drop(to_remove_loin)

    # Remove stationary points based on proximity threshold
    to_remove_stationary = []
    proximity_threshold_km = 0.1  # Define the proximity threshold in kilometers

    for mmsi in df_filtered['MMSI'].unique():
        mmsi_data = df_filtered[df_filtered['MMSI'] == mmsi]
        if len(mmsi_data) < 2:
            continue
        
        # Compute Haversine distances between consecutive points
        for i in range(len(mmsi_data) - 1):
            if haversine(
                mmsi_data.iloc[i]['Latitude'],
                mmsi_data.iloc[i]['Longitude'],
                mmsi_data.iloc[i + 1]['Latitude'],
                mmsi_data.iloc[i + 1]['Longitude']
            ) < proximity_threshold_km:
                to_remove_stationary.append(mmsi_data.index[i])
                
        # Handle the last point if it's stationary with the previous one
        if len(mmsi_data) > 1:
            if haversine(
                mmsi_data.iloc[-1]['Latitude'],
                mmsi_data.iloc[-1]['Longitude'],
                mmsi_data.iloc[-2]['Latitude'],
                mmsi_data.iloc[-2]['Longitude']
            ) < proximity_threshold_km:
                to_remove_stationary.append(mmsi_data.index[-1])

    removed_data = pd.concat([removed_data, df_filtered.loc[to_remove_stationary]])
    removed_data.sort_values(by=['MMSI', 'Time'], inplace=True)
    
    df_filtered = df_filtered.drop(to_remove_stationary)
    df_filtered = df_filtered.drop_duplicates(subset=['MMSI', 'Time'])
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered.sort_values(by=['MMSI', 'Time'], inplace=True)

    

    return df_filtered, removed_data
    
def process_files_in_directory(directory, save_directory):
    csv_transform_directory = os.path.join(save_directory, 'transform_csv')
    parquet_save_directory = os.path.join(save_directory, 'Saved_parquet')
    csv_remove_directory = os.path.join(save_directory, 'Removed_csv')

    for year in os.listdir(directory):
        file_year = os.path.join(directory, year)
        parquet_year = os.path.join(parquet_save_directory, year)
        transform_year = os.path.join(csv_transform_directory, year)
        removed_year = os.path.join(csv_remove_directory, year)

        if not os.path.isdir(file_year):
            continue

        for month in os.listdir(file_year):
            file_month = os.path.join(file_year, month)
            transform_month = os.path.join(transform_year, month)
            parquet_month = os.path.join(parquet_year, month)
            removed_month = os.path.join(removed_year, month)

            os.makedirs(transform_month, exist_ok=True)
            os.makedirs(parquet_month, exist_ok=True)
            os.makedirs(removed_month, exist_ok=True)

            if not os.path.isdir(file_month):
                continue

            for day in os.listdir(file_month):
                file_day = os.path.join(file_month, day)

                if file_day.endswith('.zip'):
                    with ZipFile(file_day, 'r') as zip_ref:
                        temp_dir = os.path.join(directory, 'temp')
                        zip_ref.extractall(temp_dir)

                        for extracted_file_name in os.listdir(temp_dir):
                            extracted_file_path = os.path.join(temp_dir, extracted_file_name)
                            transform_file = os.path.join(transform_month, extracted_file_name)
                            save_parquet = os.path.join(parquet_month, extracted_file_name + '.parquet')
                            removed_file = os.path.join(removed_month, extracted_file_name)

                            df_transform, removed_data = process_csv_file(extracted_file_path)

                            df_transform.to_csv(transform_file, index=False)
                            df_transform.to_parquet(save_parquet, index=False)
                            removed_data.to_csv(removed_file, index=False)

                            os.remove(extracted_file_path)

                    os.rmdir(temp_dir)
                

process_files_in_directory(directory, save_directory)
interpol_lin.process_files_in_directory(directory_lin, save_directory_lin)
