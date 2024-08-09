# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:16:52 2024

@author: slizo080
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour calculer la distance de Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Rayon de la Terre en kilomètres
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

# Fonction pour traiter les fichiers de base et de prédiction
def process_files(base_file, prediction_file):
    print(f"Traitement des fichiers: {base_file} et {prediction_file}")
    # Lire les fichiers de base et de prédiction
    base_data = pd.read_csv(base_file)
    prediction_data = pd.read_csv(prediction_file)
    
    # Vérifier que les fichiers ont les colonnes nécessaires
    if 'Latitude' not in base_data.columns or 'Longitude' not in base_data.columns:
        print(f"Erreur : Le fichier de base {base_file} ne contient pas les colonnes nécessaires.")
        return
    
    if 'Predicted_Latitude' not in prediction_data.columns or 'Predicted_Longitude' not in prediction_data.columns:
        print(f"Erreur : Le fichier de prédiction {prediction_file} ne contient pas les colonnes nécessaires.")
        return
    
    # Calculer les distances
    distances = haversine(base_data['Latitude'], base_data['Longitude'],
                          prediction_data['Predicted_Latitude'], prediction_data['Predicted_Longitude'])
    
    prediction_data['Distance'] = distances
    
    # Calculer la moyenne et la variance des distances pour chaque MMSI
    mmsi_groups = prediction_data.groupby('MMSI')
    prediction_data['Moyenne_par_bateau'] = mmsi_groups['Distance'].transform('mean')
    prediction_data['Variance_par_bateau'] = mmsi_groups['Distance'].transform('var')
    
    # Enregistrer le fichier de prédiction avec les nouvelles colonnes
    prediction_data.to_csv(prediction_file, index=False)
    print(f"Le fichier {prediction_file} a été mis à jour avec les colonnes Moyenne_par_bateau et Variance_par_bateau.")

# Chemins des répertoires de base et de prédiction
base_dir = './interpol_data_lin/interpol_data_lin/'
prediction_dir = './predictions/'

# Liste des années à traiter
years = [2024]
#2012,2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024

# Parcourir les fichiers de base et de prédiction pour chaque année et chaque mois
for year in years:
    for month in range(1, 12):  # Si vous souhaitez traiter uniquement jusqu'à novembre
        month_str = f"{month:02d}"
        base_month_dir = os.path.join(base_dir, str(year), month_str)
        prediction_month_dir = os.path.join(prediction_dir, str(year), month_str)
        
        print(f"Vérification des répertoires: {base_month_dir} et {prediction_month_dir}")
        
        if not os.path.exists(base_month_dir):
            print(f"Erreur : Le répertoire de base {base_month_dir} n'existe pas.")
            continue

        if not os.path.exists(prediction_month_dir):
            print(f"Erreur : Le répertoire de prédiction {prediction_month_dir} n'existe pas.")
            continue
        
        for root, dirs, files in os.walk(prediction_month_dir):
            for file in files:
                if file.endswith('.csv'):
                    base_file_path = os.path.join(base_month_dir, file.replace('prediction_', ''))
                    prediction_file_path = os.path.join(root, file)
                    
                    # Vérifier que le fichier de base existe
                    if os.path.exists(base_file_path):
                        # Appeler la fonction pour traiter les fichiers
                        process_files(base_file_path, prediction_file_path)
                    else:
                        print(f"Erreur : Le fichier de base {base_file_path} n'existe pas.")
