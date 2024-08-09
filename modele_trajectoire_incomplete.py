"""
Created on Tue Aug  6 16:18:34 2024

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

# Chemin du fichier CSV pour l'entraînement et la validation. Un seul suffit mais vous pouvez en rajouter
data_file = './interpol_data_lin/interpol_data_lin/2023/01/01/exactEarth_historical_data_2023-01-01.csv'
# Chemin du fichier CSV pour la prediction. Ici, il n'y en a qu'un seul mais vous pouvez en rajouter
data_file2 = './incomplet500/exactEarth_historical_data_2023-11-20.csv'

# Lire le fichier CSV pour l'entraînement et la validation
data = pd.read_csv(data_file)

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

# Vérification de la longueur des trajectoires
print(f"Nombre de trajectoires: {len(trajectories)}")

# Conversion de la liste en array numpy
trajectories = [(np.array(t[0]), t[1]) for t in trajectories]

# Vérification des premières trajectoires
for i, (traj, scaler) in enumerate(trajectories[:3]):
    print(f"Trajectoire {i}: {traj}")

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

# Vérification de la longueur des ensembles d'entraînement et de test
print(f"Nombre de trajectoires d'entraînement: {len(X_train)}")
print(f"Nombre de trajectoires de test: {len(X_test)}")

# Conversion des listes en arrays numpy
if X_train:
    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)
else:
    print("Aucune trajectoire valide trouvée pour l'entraînement.")

# Vérification des dimensions des ensembles
print(f"Dimensions X_train: {X_train.shape}")
print(f"Dimensions X_test: {X_test.shape}")
print(f"Dimensions y_train: {y_train.shape}")
print(f"Dimensions y_test: {y_test.shape}")

# Reshape pour LSTM
if X_train.size > 0:
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
else:
    print("Les ensembles d'entraînement sont vides.")

# Création du modèle Bi-LSTM
model = Sequential()
model.add(Bidirectional(LSTM(units=200, return_sequences=True, kernel_initializer='orthogonal'), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=100, kernel_initializer='orthogonal')))

model.add(Dense(2))  # Couche de sortie pour la prédiction (lat, lon)

# Utiliser un taux d'apprentissage plus bas
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mean_absolute_error')

# Entraînement du modèle avec un nombre d'époques augmenté
if X_train.size > 0:
    history = model.fit(X_train, y_train, epochs=1, batch_size=64, validation_split=0.2)

    # Tracer les courbes de perte
    plt.plot(history.history['loss'], label='Loss d\'entraînement')
    plt.plot(history.history['val_loss'], label='Loss de validation')
    plt.title('Courbes de perte du modèle Bi-LSTM')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.show()
else:
    print("Pas d'entraînement effectué car les ensembles d'entraînement sont vides.")

# Fonction pour créer le dossier s'il n'existe pas
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Fonction de prédiction et d'enregistrement
def predict_and_save(data_file2, plot_limit=80):
    # Charger les données de prédiction
    data = pd.read_csv(data_file2)
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
                trajectories.append((segment_data_normalized, scaler, mmsi))
    
    # Conversion de la liste en array numpy
    trajectories = [(np.array(t[0]), t[1], t[2]) for t in trajectories]
    
    # Limiter le nombre de tracés à un certain nombre
    plot_count = 0
    
    # Dossier de sortie pour les prédictions
    output_dir = './prediction_densifier'
    os.makedirs(output_dir, exist_ok=True)
    
    # DataFrame pour stocker toutes les prédictions
    all_densified_data = pd.DataFrame(columns=data.columns)

    # Sélectionner les 10 premières trajectoires pour tracer les prédictions
    for trajectory, scaler, mmsi in trajectories:
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
        
        # Ajouter des points intermédiaires pour compléter la trajectoire
        dense_predicted_positions = []
        for i in range(len(predicted_positions) - 1):
            dense_predicted_positions.append(predicted_positions[i])
            # Ajouter des points intermédiaires
            num_points_to_add = 15  # Ajuster ce nombre selon le besoin
            for j in range(1, num_points_to_add + 1):
                new_point = predicted_positions[i] + (predicted_positions[i + 1] - predicted_positions[i]) * (j / (num_points_to_add + 1))
                dense_predicted_positions.append(new_point)
        dense_predicted_positions.append(predicted_positions[-1])
        dense_predicted_positions = np.array(dense_predicted_positions)
        
        # Créer un DataFrame pour ces prédictions densifiées
        dense_pred_df = pd.DataFrame(columns=data.columns)
        dense_pred_df['Latitude'] = dense_predicted_positions[:, 0]
        dense_pred_df['Longitude'] = dense_predicted_positions[:, 1]
        dense_pred_df['MMSI'] = mmsi
        dense_pred_df['Time'] = pd.date_range(start=data['Time'].iloc[0], periods=len(dense_predicted_positions), freq='T')
        
        # Ajouter les nouvelles données au DataFrame principal
        all_densified_data = pd.concat([all_densified_data, dense_pred_df], ignore_index=True)
        
        # Tracer les positions prédites en rouge
        plt.scatter(dense_predicted_positions[:, 1], dense_predicted_positions[:, 0], label=f'Prédictions {mmsi}', color='red')
        # Tracer le trajet du bateau en bleu
        plt.scatter(trajectory_original[:, 1], trajectory_original[:, 0], label=f'Trajet {mmsi}', color='blue')
        # Ajouter des labels et une légende
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.title(f'Trajectoire et prédictions pour le MMSI {mmsi}')
        
        # Afficher la figure
        plt.show()
        
        plot_count += 1
        
    # Enregistrer toutes les prédictions dans un seul fichier CSV
    base_filename = os.path.splitext(os.path.basename(data_file2))[0]
    all_densified_data.to_csv(os.path.join(output_dir, f'all_densified_data_{base_filename}.csv'), index=False)

# Appeler la fonction de prédiction et d'enregistrement
predict_and_save(data_file2, plot_limit=80)