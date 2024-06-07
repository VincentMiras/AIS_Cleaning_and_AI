import pandas as pd
from zipfile import ZipFile
import os
import numpy as np

cols_to_read = ['MMSI', 'Time', 'Longitude', 'Latitude', 'COG', 'SOG']
directory='D:\Vincent\SPAR_AIS_DATA'
save_directory='D:\Vincent\Cleandata_nbp_mindist_mieux'

def process_csv_file(csv_files):
    df = pd.read_csv(csv_files, usecols=cols_to_read)

    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude'])
    df = df.dropna(subset=['Longitude'])

    df=df[(df['Latitude'] >= 30) & (df['Latitude'] <= 89.999)]
    df=df[(df['Longitude'] >= -179.999) & (df['Longitude'] <= 179.999)]
    
    df['Time']=pd.to_datetime(df['Time'], format='%Y%m%d_%H%M%S')
    
    
    mmsi_counts = df['MMSI'].value_counts()
    mmsi_to_keep = mmsi_counts[mmsi_counts >= 10].index
    df_filtered = df[df['MMSI'].isin(mmsi_to_keep)]
    
    to_remove_loin = []
    for mmsi in df_filtered['MMSI'].unique():
        mmsi_data = df_filtered[df_filtered['MMSI'] == mmsi]
        distances = np.sqrt(((mmsi_data[['Latitude', 'Longitude']].values[:, np.newaxis] - mmsi_data[['Latitude', 'Longitude']].values) ** 2).sum(axis=2))
    
    # Exclure la distance avec le point lui-même en remplissant la diagonale avec une valeur infinie
        np.fill_diagonal(distances, np.inf)
        
        min_distances = np.min(distances, axis=1)
        indices_to_remove_loin = np.where(min_distances > 0.5)[0]
        to_remove_loin.extend(mmsi_data.index[indices_to_remove_loin])
    
    df_filtered = df_filtered.drop(to_remove_loin)
    
    to_remove = []
    for mmsi in df_filtered['MMSI'].unique():
        mmsi_data = df_filtered[df_filtered['MMSI'] == mmsi]
        if mmsi_data['Latitude'].max() - mmsi_data['Latitude'].min() < 0.1 and mmsi_data['Longitude'].max() - mmsi_data['Longitude'].min() < 0.1:
            to_remove.extend(mmsi_data.index)
    
    df_filtered = df_filtered.drop(to_remove)
    
    df_filtered.sort_values(by=['MMSI','Time'], inplace=True)
    
    return df_filtered
    
def process_files_in_directory(directory):
    for year in os.listdir(directory):
        file_year = os.path.join(directory, year)
        save_year = os.path.join(save_directory, year)
        if not os.path.isdir(file_year):
            continue
        for month in os.listdir(file_year):
            file_month = os.path.join(file_year, month)
            save_month = os.path.join(save_year, month)
            os.makedirs(save_month, exist_ok=True)
            if not os.path.isdir(file_month):
                continue
            for day in os.listdir(file_month):
                file_day = os.path.join(file_month, day)
                if file_day.endswith('.zip'): 
                    with ZipFile(file_day, 'r') as zip_ref:
                        temp_dir = r'D:\Vincent\SPAR_AIS_DATA\temp'
                        zip_ref.extractall(temp_dir)
                        for extracted_file_name in os.listdir(temp_dir):
                            extracted_file_path = os.path.join(temp_dir, extracted_file_name)
                            save_file = os.path.join(save_month, extracted_file_name)
                            df = process_csv_file(extracted_file_path)
                            print(save_file)
                            df.to_csv(save_file, index=False)
                            os.remove(extracted_file_path)
                    os.rmdir(temp_dir)
                
csv_test='exactEarth_historical_data_2023-01-01.csv'

def ecart_mesure(csv_test):
    df = pd.read_csv(csv_test, usecols=cols_to_read)
    df['Time']=pd.to_datetime(df['Time'], format='%Y%m%d_%H%M%S')
    df.sort_values(by=['MMSI','Time'], inplace=True)
    df['Time_diff'] = df.groupby('MMSI')['Time'].diff().dt.total_seconds()
    
    mmsi_counts = df['MMSI'].value_counts()
    mmsi_to_keep = mmsi_counts[mmsi_counts >= 10].index
    df_filtered = df[df['MMSI'].isin(mmsi_to_keep)]
    
    to_remove_loin = []
    for mmsi in df_filtered['MMSI'].unique():
        mmsi_data = df_filtered[df_filtered['MMSI'] == mmsi]
        distances = np.sqrt(((mmsi_data[['Latitude', 'Longitude']].values[:, np.newaxis] - mmsi_data[['Latitude', 'Longitude']].values) ** 2).sum(axis=2))
    
    # Exclure la distance avec le point lui-même en remplissant la diagonale avec une valeur infinie
        np.fill_diagonal(distances, np.inf)
        
        min_distances = np.min(distances, axis=1)
        indices_to_remove_loin = np.where(min_distances > 1)[0]
        to_remove_loin.extend(mmsi_data.index[indices_to_remove_loin])
    
    df_filtered = df_filtered.drop(to_remove_loin)
    
    to_remove = []
    for mmsi in df_filtered['MMSI'].unique():
        mmsi_data = df_filtered[df_filtered['MMSI'] == mmsi]
        if mmsi_data['Latitude'].max() - mmsi_data['Latitude'].min() < 0.1 and mmsi_data['Longitude'].max() - mmsi_data['Longitude'].min() < 0.1:
            to_remove.extend(mmsi_data.index)
    
    df_filtered = df_filtered.drop(to_remove)
    
    # Calcul de la moyenne et de l'écart-type pour chaque MMSI
    result = df_filtered.groupby('MMSI')['Time_diff'].agg(['mean', 'std','count']).reset_index()
    result.sort_values(by=['mean'], inplace=True)
    df_filtered.to_csv('testclean.csv', index=False)
    return(df_filtered)
   
