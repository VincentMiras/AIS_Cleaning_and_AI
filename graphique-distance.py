# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:45:00 2024

@author: slizo080
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Fonction pour extraire les distances depuis les fichiers
def extract_distances(prediction_dir, years):
    all_distances = {'Année': [], 'Mois': [], 'Distance': []}
    
    for year in years:
        for month in range(1, 13):
            month_str = f"{month:02d}"
            prediction_month_dir = os.path.join(prediction_dir, str(year), month_str)
            
            if not os.path.exists(prediction_month_dir):
                print(f"Erreur : Le répertoire de prédiction {prediction_month_dir} n'existe pas.")
                continue
            
            for root, dirs, files in os.walk(prediction_month_dir):
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        
                        try:
                            data = pd.read_csv(file_path)
                            if 'Distance' in data.columns:
                                all_distances['Année'].extend([year] * len(data))
                                all_distances['Mois'].extend([month] * len(data))
                                all_distances['Distance'].extend(data['Distance'])
                        except Exception as e:
                            print(f"Erreur lors de la lecture du fichier {file_path}: {e}")

    return pd.DataFrame(all_distances)

# Chemins des répertoires de prédiction
prediction_dir = './predictions/'

# Liste des années à traiter
years = [2015,2016,2017]

# Extraire les distances
distances_df = extract_distances(prediction_dir, years)

# Calculer les moyennes des distances par année
annual_means = distances_df.groupby('Année')['Distance'].mean().reset_index()

# Tracer la moyenne des distances par année
plt.figure()
plt.bar(annual_means['Année'], annual_means['Distance'], color='skyblue')
plt.xlabel('Année')
plt.ylabel('Distance Moyenne (km)')
plt.title('Distance Moyenne par Année')
plt.xticks(annual_means['Année'])
plt.grid(axis='y')
plt.savefig('annual_distances.png')
plt.show()

# Calculer les moyennes des distances par mois pour chaque année
monthly_means = distances_df.groupby(['Année', 'Mois'])['Distance'].mean().reset_index()

# Tracer la moyenne des distances par mois pour chaque année
for year in years:
    monthly_data = monthly_means[monthly_means['Année'] == year]
    plt.figure()
    plt.bar(monthly_data['Mois'], monthly_data['Distance'], color='lightgreen')
    plt.xlabel('Mois')
    plt.ylabel('Distance Moyenne (km)')
    plt.title(f'Distance Moyenne par Mois - Année {year}')
    plt.xticks(monthly_data['Mois'])
    plt.grid(axis='y')
    plt.savefig(f'monthly_distances_{year}.png')
    plt.show()
