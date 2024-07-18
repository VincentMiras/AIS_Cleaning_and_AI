import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d

directory = 'D:\\Vincent\\Cleandata_valid_transform\\Saved_csv'
save_directory = 'D:\\Vincent\\interpol_data_transf_lin'

def interpol(csv_file):
    try:
        df = pd.read_csv(csv_file)
        df['Time'] = pd.to_datetime(df['Time'])
        data = df.sort_values(by='Time')
        grouped = data.groupby('MMSI')
        interpolated_data = grouped.apply(interpolate_coordinates)
        return interpolated_data
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {csv_file}: {str(e)}")
        return None

def interpolate_coordinates(series):
    try:
        # Remove duplicates if any (optional step)
        data = series.drop_duplicates(subset='Time').sort_values(by='Time')
        
        # Define start and end times for interpolation
        start_time = data['Time'].min()
        end_time = data['Time'].max()
        interpolated_times = pd.date_range(start=start_time, end=end_time, freq='2T')
        
        # Interpolate Latitude and Longitude using linear interpolation
        interp_lat = interp1d(data['Time'].astype(np.int64) // 10**9, data['Latitude'], kind='linear', fill_value='extrapolate')
        interp_lon = interp1d(data['Time'].astype(np.int64) // 10**9, data['Longitude'], kind='linear', fill_value='extrapolate')
        
        interpolated_latitudes = interp_lat(interpolated_times.astype(np.int64) // 10**9)
        interpolated_longitudes = interp_lon(interpolated_times.astype(np.int64) // 10**9)
        
        # Create DataFrame for interpolated data
        interpolated_df = pd.DataFrame({
            'MMSI': series['MMSI'].iloc[0],  # Assume MMSI is constant within group
            'Time': interpolated_times,
            'Latitude': interpolated_latitudes,
            'Longitude': interpolated_longitudes,
        })
        
        return interpolated_df
    
    except ValueError as ve:
        print(f"Erreur lors de l'interpolation pour MMSI {series['MMSI'].iloc[0]} : {str(ve)}")
        return None
    except Exception as e:
        print(f"Erreur inattendue lors de l'interpolation pour MMSI {series['MMSI'].iloc[0]} : {str(e)}")
        return None

def process_files_in_directory(directory, save_directory):
    try:
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
                    save_file = os.path.join(save_month, day)
                    print(file_day)
                    try:
                        df = interpol(file_day)
                        if df is not None:
                            df.to_csv(save_file, index=False)
                    except Exception as e:
                        print(f"Erreur lors du traitement de {file_day}: {e}")

    except Exception as e:
        print(f"Erreur lors du traitement du répertoire : {str(e)}")



