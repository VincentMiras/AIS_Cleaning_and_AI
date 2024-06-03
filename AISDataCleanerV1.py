# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
from zipfile import ZipFile
import os


cols_to_read = ['MMSI', 'Time', 'Longitude', 'Latitude', 'COG', 'SOG']
directory='D:\Vincent\SPAR_AIS_DATA'
save_directory='D:\Vincent\Cleandata'

def process_csv_file(csv_files):
    df = pd.read_csv(csv_files, usecols=cols_to_read)

    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude'])
    df = df.dropna(subset=['Longitude'])

    df=df[(df['Latitude'] >= 30) & (df['Latitude'] <= 89.999)]
    df=df[(df['Longitude'] >= -179.999) & (df['Latitude'] <= 179.999)]

    df.sort_values(by=['MMSI','Time'], inplace=True)
    return df
    
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
                
                
                
            