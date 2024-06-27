# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:20:32 2024

@author: slizo080
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.interpolate import interp1d
from tensorflow.keras.layers import RepeatVector
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from scipy.interpolate import CubicSpline
import random

# Charger les données AIS avec latitude, longitude, temps et MMSI
csv_filename = './exactEarth_historical_data_2023-01-05_modif2.csv'
df = pd.read_csv(csv_filename)

# Convertir la colonne 'Time' en datetime
#df['Time'] = pd.to_datetime(df['Time'])
sequence_length = 40 # Longueur de la séquence de coordonnées du trajet
selected_mmsi = df['MMSI'].unique()[11]
selected_sequence = df[df['MMSI'] == selected_mmsi][['Latitude', 'Longitude']].values
# # Préparer les données d'entrée pour la prédiction
X_single_mmsi = pad_sequences([selected_sequence], maxlen=sequence_length, dtype='float32')




# Normalisation des données
scaler = StandardScaler()
#df[['Latitude', 'Longitude', 'Time_norm']] = scaler.fit_transform(df[['Latitude', 'Longitude', 'Time_norm']])

# Normaliser la colonne 'Time' en secondes
#df['Time_norm'] = (df['Time'] - df.groupby('MMSI')['Time'].transform('min')).dt.total_seconds()

# Diviser les données par bateau en séquences temporelles
unique_ships = df['MMSI'].unique()
ship_sequences = [df[df['MMSI'] == mmsi][['Latitude', 'Longitude', 'Time_norm']].values for mmsi in unique_ships]

def prepare_data_with_trajectories(df, sequence_length):
  
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i+sequence_length].values)
        y.append(df.iloc[i+sequence_length][['Latitude', 'Longitude']].values)
    return np.array(X), np.array(y)


X_train, y_train = prepare_data_with_trajectories(df[['Latitude', 'Longitude']], sequence_length)




from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Définir la forme de l'entrée avec Input
inputs = Input(shape=(sequence_length, 2))

# Ajouter une couche SimpleRNN avec return_sequences=True pour obtenir des séquences en sortie
x = SimpleRNN(50, activation='relu', return_sequences=True)(inputs)

# Ajouter une couche LSTM pour capturer des dépendances à long terme
x =  LSTM(50, activation='relu')(x)

# Couche de sortie avec 2 neurones pour prédire Latitude et Longitude
outputs = Dense(2)(x)

# Créer le modèle en spécifiant les entrées et les sorties
model = Model(inputs=inputs, outputs=outputs)

# Compilation du modèle avec l'optimiseur et la fonction de perte
model.compile(optimizer='adam', loss='mse')

# Entraînement du modèle
model.fit(X_train, y_train, epochs=10, batch_size=32)
# model.fit(X_single_mmsi, X_single_mmsi, epochs=50, batch_size=32)

# Prédiction pour une seule séquence MMSI
#predicted_trajectory = model.predict(X_multiple_mmsis)

predicted_trajectories = model.predict(X_single_mmsi)




def sequence_mse(y_true, y_pred):
    """
    Calcul de la perte MSE sur les séquences de coordonnées.
    
    Args:
        y_true: Tenseur des coordonnées de la trajectoire réelle (batch_size, sequence_length, num_features).
        y_pred: Tenseur des coordonnées de la trajectoire prédite (batch_size, sequence_length, num_features).
        
    Returns:
        MSE: Moyenne des erreurs quadratiques moyennes sur chaque pas de temps.
    """
    # Calcul de la différence entre les prédictions et les vraies valeurs
    squared_difference = K.square(y_true - y_pred)
    # Calcul de la moyenne des erreurs quadratiques sur chaque pas de temps
    mse_per_timestep = K.mean(squared_difference, axis=-1)
    # Calcul de la moyenne des MSE sur toute la séquence
    mse = K.mean(mse_per_timestep, axis=-1)
    return mse


# Diviser X_single_mmsi en deux parties
part_length = sequence_length // 4
parts = []
for i in range(4):
    part = X_single_mmsi[:, i * part_length:(i + 1) * part_length, :]
    parts.append(part)


# Liste pour stocker les prédictions
all_predictions = []

# Faire une prédiction pour chaque partie
for part in parts:
    # Prédire les coordonnées futures pour la partie actuelle
    predicted_trajectory = model.predict(part)
    
    # Ajouter la prédiction à la liste des prédictions
    all_predictions.append(predicted_trajectory.flatten())

# Convertir la liste des prédictions en tableau numpy
all_predictions = np.array(all_predictions)

# Afficher toutes les prédictions
print("Prédictions pour le MMSI {} :".format(selected_mmsi))
print(all_predictions)



plt.plot(selected_sequence[:, 1], selected_sequence[:, 0], label='Trajet du bateau', color='blue')

# Tracer les prédictions du modèle hybride en rouge
plt.plot(all_predictions[:,1], all_predictions[:,0], marker='o', markersize=5, color='red')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Trajet du bateau avec prédictions du modèle hybride')
plt.legend()
plt.show()








































# # Nombre de prédictions à faire
# num_predictions = 10

# # Liste pour stocker les prédictions
# all_predictions = []

# #predicted_trajectory = model.predict(X_single_mmsi)

# # Diviser X_single_mmsi en deux parties
# part_length = sequence_length // 4
# parts = []
# for i in range(4):
#     part = X_single_mmsi[:, i * part_length:(i + 1) * part_length, :]
#     parts.append(part)


# # Liste pour stocker les prédictions
# all_predictions = []

# # Faire une prédiction pour chaque partie
# for part in parts:
#     # Prédire les coordonnées futures pour la partie actuelle
#     predicted_trajectory = model.predict(part)
    
#     # Ajouter la prédiction à la liste des prédictions
#     all_predictions.append(predicted_trajectory.flatten())

# # Convertir la liste des prédictions en tableau numpy
# all_predictions = np.array(all_predictions)

# # Afficher toutes les prédictions
# print("Prédictions pour le MMSI {} :".format(selected_mmsi))
# print(all_predictions)

# # Afficher toutes les prédictions
# print("Prédictions pour le MMSI {} :".format(selected_mmsi))
# print(predicted_trajectory)

# plt.plot(selected_sequence[:, 1], selected_sequence[:, 0], label='Trajet du bateau', color='blue')

# # Tracer les prédictions en rouge
# for prediction in all_predictions:
#     plt.plot(prediction[1], prediction[0], marker='o', markersize=5, color='red')

# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Trajet du bateau avec prédictions')
# plt.legend()
# plt.show()













