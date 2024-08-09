# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:40:40 2024

@author: slizo080
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chemin du dossier contenant les fichiers CSV de novembre 2023
november_data_dir = './interpol_data_lin/interpol_data_lin/2023/11/11/'
# Chemin du dossier où enregistrer les nouvelles données
incomplete_data_dir = './incomplet500/'

# Créer le dossier s'il n'existe pas
os.makedirs(incomplete_data_dir, exist_ok=True)

# Fonction pour supprimer une dizaine de points pour chaque bateau
def remove_points(data, points_to_remove=500):
    new_data = pd.DataFrame()
    for mmsi in data['MMSI'].unique():
        ship_data = data[data['MMSI'] == mmsi]
        if len(ship_data) > points_to_remove:
            # Sélectionner des indices aléatoires à supprimer
            indices_to_remove = np.random.choice(ship_data.index, size=points_to_remove, replace=False)
            ship_data = ship_data.drop(indices_to_remove)
        new_data = pd.concat([new_data, ship_data])
    return new_data

def plot_trajectories(data, title):
    for mmsi in data['MMSI'].unique()[:5]:  # Limiter à cinq bateaux
        ship_data = data[data['MMSI'] == mmsi]
        plt.figure(figsize=(10, 6))
        plt.plot(ship_data['Longitude'], ship_data['Latitude'], marker='o', linestyle='-', label=f'MMSI {mmsi}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'{title} - MMSI {mmsi}')
        plt.legend()

        plt.show()

# Parcourir tous les fichiers CSV de novembre 2023
for filename in os.listdir(november_data_dir):
    if filename.endswith('.csv'):
        # Lire le fichier CSV
        filepath = os.path.join(november_data_dir, filename)
        data = pd.read_csv(filepath)
        

        
        # Tracer les trajets avant suppression
        plot_trajectories(data, f'Trajets avant suppression - {filename}')
        
        # Supprimer une dizaine de points aléatoires pour chaque bateau
        incomplete_data = remove_points(data, points_to_remove=500)
        
        # Tracer les trajets après suppression
        plot_trajectories(incomplete_data, f'Trajets après suppression - {filename}')
        
        # Enregistrer les nouvelles données dans le dossier './imcomplet'
        new_filepath = os.path.join(incomplete_data_dir, filename)
        incomplete_data.to_csv(new_filepath, index=False)

print("Traitement terminé, tracés affichés et nouvelles données enregistrées dans le dossier './imcomplet'.")



















