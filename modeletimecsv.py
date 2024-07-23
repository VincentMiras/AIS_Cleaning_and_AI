# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:34:10 2024

@author: slizo080
"""


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Chemin des fichiers CSV et du dossier pour enregistrer les prédictions
data_dirs = [
    './interpol_data_lin/interpol_data_lin/2023/01/01',
    './interpol_data_lin/interpol_data_lin/2022/2022/01/01/',
    './interpol_data_lin/interpol_data_lin/2022/2022/02/02/',
    './interpol_data_lin/interpol_data_lin/2022/2022/03/03',
    './interpol_data_lin/interpol_data_lin/2022/2022/04/04/',
    './interpol_data_lin/interpol_data_lin/2022/2022/05/05/',
    './interpol_data_lin/interpol_data_lin/2022/2022/06/06/',
    './interpol_data_lin/interpol_data_lin/2022/2022/07/07/',
    './interpol_data_lin/interpol_data_lin/2022/2022/08/08/',
    './interpol_data_lin/interpol_data_lin/2022/2022/09/09/',
    './interpol_data_lin/interpol_data_lin/2022/2022/10/10/',
    './interpol_data_lin/interpol_data_lin/2022/2022/11/11/',
    './interpol_data_lin/interpol_data_lin/2022/2022/12/12/'
]
output_dirs = [
    './predictions/2022/01',
    './predictions/2022/02',
    './predictions/2022/03',
    './predictions/2022/04',
    './predictions/2022/05',
    './predictions/2022/06',
    './predictions/2022/07',
    './predictions/2022/08',
    './predictions/2022/09',
    './predictions/2022/10',
    './predictions/2022/11',
    './predictions/2022/12'
]

for output_dir in output_dirs:
    os.makedirs(output_dir, exist_ok=True)

# Liste des fichiers CSV pour l'entraînement et la validation
csv_files = [f'exactEarth_historical_data_2023-01-0{d}.csv' for d in range(1, 6)]

# Lire les fichiers CSV pour l'entraînement et la validation
data = pd.concat([pd.read_csv(os.path.join(data_dirs[0], file)) for file in csv_files], ignore_index=True)

# Convertir la colonne 'Time' en datetime
data['Time'] = pd.to_datetime(data['Time'])

# Trier les données par MMSI et par time
data = data.sort_values(by=['MMSI', 'Time'])

# Normaliser les données en utilisant le temps
data['TimeSeconds'] = (data['Time'] - data['Time'].min()).dt.total_seconds()

# Liste pour stocker les trajectoires normalisées
trajectories = []

# Normaliser les données par segments de 24 heures
for mmsi in data['MMSI'].unique():
    ship_data = data[data['MMSI'] == mmsi].copy()
    
    # Diviser les données en segments de 24 heures
    ship_data['TimeSegment'] = (ship_data['Time'] - ship_data['Time'].min()).dt.total_seconds() // (24 * 3600)
    
    for segment in ship_data['TimeSegment'].unique():
        segment_data = ship_data[ship_data['TimeSegment'] == segment].copy()
        time_norm = (segment_data['TimeSeconds'] - segment_data['TimeSeconds'].min()) / (24 * 3600)
        
        if len(segment_data) > 1:  # Vérifier que le segment a plus d'un point
            # Normaliser les latitudes et longitudes
            scaler = MinMaxScaler()
            segment_data[['Latitude', 'Longitude']] = scaler.fit_transform(segment_data[['Latitude', 'Longitude']])
            
            # Combiner les valeurs de temps normalisées avec les latitudes et longitudes
            segment_data_normalized = np.column_stack((time_norm, segment_data[['Latitude', 'Longitude']]))
            
            # Ajout des trajectoires normalisées à la liste
            trajectories.append((segment_data_normalized, scaler))

# Conversion de la liste en array numpy
trajectories = [(np.array(t[0]), t[1]) for t in trajectories]

# Création des ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = [], [], [], []

for trajectory, scaler in trajectories:
    # Créer les features (X) et les targets (y)
    X = trajectory[:-1]  # Tout sauf le dernier point
    y = trajectory[1:, 1:]  # Tout sauf la première colonne (temps) et le premier point
    
    # Division en ensembles d'entraînement et de test
    if len(X) > 1 and len(y) > 1:  # Vérifier que la trajectoire a plus d'un point
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train.append(X_tr)
        X_test.append(X_te)
        y_train.append(y_tr)
        y_test.append(y_te)

# Conversion des listes en arrays numpy
X_train = np.concatenate(X_train, axis=0)
X_test = np.concatenate(X_test, axis=0)
y_train = np.concatenate(y_train, axis=0)
y_test = np.concatenate(y_test, axis=0)

# Reshape pour LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Création du modèle Bi-LSTM
model = Sequential()
model.add(Bidirectional(LSTM(units=200, return_sequences=True, kernel_initializer='orthogonal'), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=100, kernel_initializer='orthogonal')))
model.add(Dropout(0.3))
model.add(Dense(2))  # Couche de sortie pour la prédiction (lat, lon)

# Utiliser un taux d'apprentissage plus bas
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mean_absolute_error')

# Entraînement du modèle avec un nombre d'époques augmenté
history = model.fit(X_train, y_train, epochs=1, batch_size=64, validation_split=0.2)

# Tracer les courbes de perte
plt.plot(history.history['loss'], label='Loss d\'entraînement')
plt.plot(history.history['val_loss'], label='Loss de validation')
plt.title('Courbes de perte du modèle Bi-LSTM')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()
plt.show()

def predict_and_save(data_file, output_dir, plot_limit=10):
    # Charger les données de prédiction
    data = pd.read_csv(data_file)
    
    # Convertir la colonne 'Time' en datetime
    data['Time'] = pd.to_datetime(data['Time'])
    
    # Trier les données par MMSI et par time
    data = data.sort_values(by=['MMSI', 'Time'])
    
    # Normaliser les données en utilisant le temps
    data['TimeSeconds'] = (data['Time'] - data['Time'].min()).dt.total_seconds()
    
    # Liste pour stocker les trajectoires normalisées
    trajectories = []
    
    for mmsi in data['MMSI'].unique():
        ship_data = data[data['MMSI'] == mmsi].copy()
        
        # Diviser les données en segments de 24 heures
        ship_data['TimeSegment'] = (ship_data['Time'] - ship_data['Time'].min()).dt.total_seconds() // (24 * 3600)
        
        for segment in ship_data['TimeSegment'].unique():
            segment_data = ship_data[ship_data['TimeSegment'] == segment].copy()
            time_norm = (segment_data['TimeSeconds'] - segment_data['TimeSeconds'].min()) / (24 * 3600)
            
            if len(segment_data) > 1:  # Vérifier que le segment a plus d'un point
                # Normaliser les latitudes et longitudes
                scaler = MinMaxScaler()
                segment_data[['Latitude', 'Longitude']] = scaler.fit_transform(segment_data[['Latitude', 'Longitude']])
                
                # Combiner les valeurs de temps normalisées avec les latitudes et longitudes
                segment_data_normalized = np.column_stack((time_norm, segment_data[['Latitude', 'Longitude']]))
                
                # Ajout des trajectoires normalisées à la liste
                trajectories.append((segment_data_normalized, scaler, mmsi, segment_data.index))
    
    # Conversion de la liste en array numpy
    trajectories = [(np.array(t[0]), t[1], t[2], t[3]) for t in trajectories]
    
    # Limiter le nombre de tracés à un certain nombre
    plot_count = 0
    
    # Sélectionner les 10 premières trajectoires pour tracer les prédictions
    for trajectory, scaler, mmsi, indices in trajectories:
        if plot_count >= plot_limit:
            break
        
        # Préparer les données pour la prédiction
        X_test_mmsi = trajectory[:-1].reshape(-1, 1, 3)
        
        # Prédire les positions
        predicted_positions = model.predict(X_test_mmsi)
        
        # Dénormaliser les positions prédites
        predicted_positions = scaler.inverse_transform(predicted_positions)
        
        # Dénormaliser les positions réelles
        trajectory_original = scaler.inverse_transform(trajectory[:, 1:])
        
        # Tracer le trajet du bateau en bleu
        plt.plot(trajectory_original[:, 1], trajectory_original[:, 0], label=f'Trajet {mmsi}', color='blue')
        
        # Tracer les prédictions du modèle en rouge
        plt.plot(predicted_positions[:, 1], predicted_positions[:, 0], label=f'Prédiction {mmsi}', color='red', linestyle='--')
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Trajet et prédiction pour le MMSI {mmsi}')
        plt.legend()
        plt.show()
        
        plot_count += 1
    
    # Enregistrer les prédictions dans un fichier CSV
    output_file = os.path.join(output_dir, f'prediction_{os.path.basename(data_file)}')
    
    # Ajouter les colonnes pour les prédictions
    data['Predicted_Latitude'] = np.nan
    data['Predicted_Longitude'] = np.nan
    
    for trajectory, scaler, mmsi, indices in trajectories:
        # Préparer les données pour la prédiction
        X_test_mmsi = trajectory[:-1].reshape(-1, 1, 3)
        
        # Prédire les positions
        predicted_positions = model.predict(X_test_mmsi)
        predicted_positions = scaler.inverse_transform(predicted_positions)
        
        # Assigner les prédictions au DataFrame
        for i, idx in enumerate(indices):
            if i < len(predicted_positions):
                data.loc[idx, 'Predicted_Latitude'] = predicted_positions[i, 0]
                data.loc[idx, 'Predicted_Longitude'] = predicted_positions[i, 1]

    data.to_csv(output_file, index=False)
    # Sélectionner uniquement les colonnes souhaitées
    result = data[['Time', 'MMSI', 'Predicted_Latitude', 'Predicted_Longitude']]
    
    result.to_csv(output_file, index=False)

# Appel de la fonction pour chaque fichier de prédiction
for data_dir, output_dir in zip(data_dirs[1:], output_dirs):
    remaining_csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    for csv_file in remaining_csv_files:
        predict_and_save(os.path.join(data_dir, csv_file), output_dir)
