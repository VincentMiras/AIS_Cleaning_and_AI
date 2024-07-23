# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:57:36 2024

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
data_dir = '/interpol_data_lin/interpol_data_lin/2023/01/'
output_dir = './predictions'
os.makedirs(output_dir, exist_ok=True)

# Liste des fichiers CSV
csv_files = [f'exactEarth_historical_data_2023-01-0{d}lin.csv' for d in range(1, 6)]

# Lire les fichiers CSV pour l'entraînement et la validation
data = pd.concat([pd.read_csv(os.path.join(data_dir, file)) for file in csv_files], ignore_index=True)

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

# Charger les données de prédiction 
data6 = pd.read_csv('./exactEarth_historical_data_2023-01-08lin.csv')

# Convertir la colonne 'Time' en datetime
data6['Time'] = pd.to_datetime(data6['Time'])

# Trier les données par MMSI et par time
data6 = data6.sort_values(by=['MMSI', 'Time'])

# Normaliser les données en utilisant le temps
data6['TimeSeconds'] = (data6['Time'] - data6['Time'].min()).dt.total_seconds()

# Liste pour stocker les trajectoires normalisées de data6
trajectories2 = []

for mmsi in data6['MMSI'].unique():
    ship_data = data6[data6['MMSI'] == mmsi].copy()
    
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
            trajectories2.append((segment_data_normalized, scaler))

# Conversion de la liste en array numpy
trajectories2 = [(np.array(t[0]), t[1]) for t in trajectories2]

# Sélectionner 10 MMSI au hasard
selected_mmsi = data6['MMSI'].unique()

# Tracer les trajectoires originales et les prédictions pour les MMSI sélectionnés
for mmsi in selected_mmsi:
    ship_data = data6[data6['MMSI'] == mmsi].copy()
    ship_data['TimeSegment'] = (ship_data['Time'] - ship_data['Time'].min()).dt.total_seconds() // (24 * 3600)
    
    for segment in ship_data['TimeSegment'].unique():
        segment_data = ship_data[ship_data['TimeSegment'] == segment].copy()
        time_norm = (segment_data['TimeSeconds'] - segment_data['TimeSeconds'].min()) / (24 * 3600)
        time_data = segment_data['Time'].values  # Récupérer les valeurs de temps
        scaler = MinMaxScaler()
        segment_data[['Latitude', 'Longitude']] = scaler.fit_transform(segment_data[['Latitude', 'Longitude']])
        segment_data_normalized = np.column_stack((time_norm, segment_data[['Latitude', 'Longitude']]))
        
        # Préparer les données pour la prédiction
        X_test_mmsi = segment_data_normalized[:-1].reshape(-1, 1, 3)
        
        # Prédire les positions
        predicted_positions = model.predict(X_test_mmsi)
        
        # Dénormaliser les positions prédites
        predicted_positions = scaler.inverse_transform(predicted_positions)
        
        # Dénormaliser les positions réelles
        segment_data[['Latitude', 'Longitude']] = scaler.inverse_transform(segment_data[['Latitude', 'Longitude']])
        
        # Tracer le trajet du bateau en bleu
        plt.plot(segment_data['Longitude'], segment_data['Latitude'], label=f'Trajet {mmsi} - Segment {segment}', color='blue')
        
        # Tracer les prédictions du modèle en rouge
        plt.plot(predicted_positions[:, 1], predicted_positions[:, 0], label=f'Prédiction {mmsi} - Segment {segment}', color='red', linestyle='--')
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Trajet et prédiction pour le MMSI {mmsi} - Segment {segment}')
        plt.legend()
        plt.show()
