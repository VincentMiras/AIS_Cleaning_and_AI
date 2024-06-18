import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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
        y.append(df.iloc[i+sequence_length][['Latitude', 'Longitude', 'Time_norm']].values)
    return np.array(X), np.array(y)

sequence_length = 40 # Longueur de la séquence de coordonnées du trajet
X_train, y_train = prepare_data_with_trajectories(df[['Latitude', 'Longitude', 'Time_norm']], sequence_length)




#Interpolation quadratique
max_length = max(len(seq) for seq in ship_sequences)
interp_sequences = []
for seq in ship_sequences:
    interp_func = interp1d(np.arange(len(seq)), seq, kind='quadratic', axis=0)
    interp_seq = interp_func(np.linspace(0, len(seq) - 1, max_length))
    interp_sequences.append(interp_seq)

X_trainn = np.array(interp_sequences)
y_train = X_trainn[:, :, :2] 
X_train = X_trainn[:, :, 2:]

y_test = X_trainn[:, :, :2] 
X_test = X_trainn[:, :, 2:]

# # Interpolation cubique spline
# max_length = max(len(seq) for seq in ship_sequences)
# interp_sequences = []
# for seq in ship_sequences:
#     cs = CubicSpline(np.arange(len(seq)), seq, axis=0)
#     interp_seq = cs(np.linspace(0, len(seq) - 1, max_length))
#     interp_sequences.append(interp_seq)

# X_trainn = np.array(interp_sequences)
# y_train = X_trainn[:, :, :2] 
# X_train = X_trainn[:, :, 2:]





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




# model = Sequential([
#     LSTM(10, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),  
    
#     Dense(3)  # Deux sorties pour prédire 'Latitude' et 'Longitude' pour chaque pas de temps futur
# ])
# # Compilation du modèle avec la fonction de perte personnalisée
# optimizer = Adam(learning_rate=0.001)

# model.compile(optimizer=optimizer, loss='mse')

# #Création et entraînement du modèle LSTM
# model = Sequential([
#     LSTM(5, activation='relu', input_shape=(X_train.shape[1], 1)),  # 1 pour la caractéristique 'Time_norm'
#     RepeatVector(X_train.shape[1]),  # Répéter les prédictions pour toute la séquence temporelle
#     Dense(2)  # Deux sorties pour prédire 'Latitude' et 'Longitude'
# ])

# optimizer = Adam(learning_rate=0.0001)  # Réduisez le taux d'apprentissage
# model.compile(optimizer=optimizer, loss='mse')
# history = model.fit(X_train, y_train, epochs=5, batch_size=20)
# # Générer des prédictions pour chaque séquence temporelle future
# predicted_trajectory = model.predict(X_train)



# # optimizer = Adam(learning_rate=0.0001)  # Réduisez le taux d'apprentissage
# # model.compile(optimizer=optimizer, loss='mse')
# history = model.fit(X_train, y_train, epochs=5, batch_size=20)
# print('a')
# # Générer des prédictions pour chaque séquence temporelle future
# # predicted_trajectory = model.predict(X_train)

# # 5 en lstm et  en epochs marche bien





# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.title('Courbe de perte')
# # plt.legend()
# # plt.show()

# # print(predicted_trajectory)



# Sélectionnez un bateau au hasard
random_ship_index = random.randint(0, len(ship_sequences) - 1)
selected_ship_sequence = ship_sequences[random_ship_index]

# Obtenez le trajet de base pour ce bateau
base_trajectory = selected_ship_sequence[:, :2]  # Latitude et longitude

# Préparez les données pour l'entrée du modèle LSTM
X_train_single_ship = pad_sequences([selected_ship_sequence], maxlen=sequence_length, dtype='float32')

# # Prédire le trajet futur pour ce bateau
# predicted_trajectorys = model.predict(X_train_single_ship)

# # Tracer les trajets de base et prédits
# plt.plot(base_trajectory[:, 1], base_trajectory[:, 0], label='Trajet de base', color='blue')
# plt.plot(predicted_trajectorys[:, 0], predicted_trajectorys[:, 1], label='Trajet prédit', color='red')

# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Trajet de base et trajet prédit pour un bateau au hasard')
# plt.legend()
# plt.show()

# print('b')
# # Fonction pour prédire un point entre deux points donnés
# def predict_intermediate_point(point1, point2):
#     # Préparez les données pour l'entrée du modèle LSTM
#     X_input = np.array([np.concatenate([point1, point2])])
#     # Assurez-vous que les données sont dans le bon format pour le modèle LSTM
#     X_input = X_input.reshape(1, 1, X_input.shape[1])  # Mettez les données dans le format (1, 1, 4)
#     # Prédire le point intermédiaire
#     predicted_point = model.predict(X_input)
#     return predicted_point

# # Liste pour stocker le nouveau trajet prédit
# predicted_trajectory = []

# # Parcourir le trajet initial
# for i in range(len(selected_ship_sequence) - 1):
#     point1 = selected_ship_sequence[i]
#     point2 = selected_ship_sequence[i + 1]
#     # Vérifiez la différence de Time_norm
#     time_norm_diff = point2[-1] - point1[-1]
#     if time_norm_diff > 0.005:
#         # Prédire un nouveau point intermédiaire
#         predicted_point = predict_intermediate_point(point1, point2)
#         # Ajouter les coordonnées prédites individuelles à la trajectoire
#         for coord in predicted_point[0]:
#             predicted_trajectory.append(coord)  # Ajoutez chaque coordonnée prédite individuellement
#     else:
#         # Ajouter les points existants à la trajectoire prédite
#         predicted_trajectory.extend(point1)  # Ajoutez chaque coordonnée existante individuellement


from tensorflow.keras import Input, Model

from tensorflow.keras.layers import Concatenate

# LSTM pour la prédiction de trajectoire
lstm_model = Sequential([
    LSTM(10, activation='relu', input_shape=(40, 3)),  
    Dense(2)  # Sorties pour prédire 'Latitude' et 'Longitude'
])

# Modèle LSTM pour la prédiction de trajectoire
lstm_model = Sequential([
    LSTM(10, activation='relu', input_shape=(X_train.shape[1],1)),  
    Dense(2)  # Sorties pour prédire 'Latitude' et 'Longitude'
])

lstm_model.summary()

# Modèle multitâche avec une architecture différente (à titre d'exemple)
multitask_model = Sequential([
    LSTM(10, activation='relu', input_shape=(X_train.shape[0],X_train.shape[1])),  
    Dense(4, activation='relu')  # Sorties pour prédire 'Latitude', 'Longitude', 'Vitesse', 'Accélération'
])


multitask_model.summary()

def lst_view():
    optimizer = Adam(learning_rate=0.001)
    lstm_model.compile(optimizer=optimizer, loss='mse')
    lstm_model.fit(X_train, y_train, epochs=5, batch_size=20)
    solve=lstm_model.predict(X_train)
    return(solve)

def multi_view():
    optimizer = Adam(learning_rate=0.001)
    multitask_model.compile(optimizer=optimizer, loss='mse')
    multitask_model.fit(X_train, y_train, epochs=5, batch_size=20)
    solve=multitask_model.predict(X_train)
    return(solve)

#%%
# Entrées pour le modèle combiné
input1 = Input(shape=(X_train.shape))
input2 = Input(shape=(X_train.shape))

# Appliquer les modèles aux entrées correspondantes
x1 = lstm_model(input1)
x2 = multitask_model(input2)

# Concaténer les sorties des modèles
x = Concatenate()([x1, x2])

# Couche de sortie pour prédire 'Latitude' et 'Longitude'
output = Dense(2)(x)

# Créer le modèle combiné
combined_model = Model(inputs=[input1, input2], outputs=output)
combined_model.summary()

# Compilation du modèle combiné
optimizer = Adam(learning_rate=0.001)
combined_model.compile(optimizer=optimizer, loss='mse')

# Entraînement du modèle combiné
history = combined_model.fit([X_train, X_train], y_train, epochs=5, batch_size=20)

# Prédiction avec le modèle combiné
predicted_trajectory = combined_model.predict([X_train_single_ship, X_train_single_ship])



# # Ajouter le dernier point à la trajectoire prédite
# predicted_trajectory.extend(selected_ship_sequence[-1])

# # Convertir la trajectoire prédite en un tableau numpy
# predicted_trajectory = np.array(predicted_trajectory)

# Tracer les trajets de base et prédits
plt.plot(base_trajectory[:, 1], base_trajectory[:, 0], label='Trajet de base', color='blue')
plt.plot(predicted_trajectory[:, 1], predicted_trajectory[:, 0], label='Trajet prédit', color='red')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Trajet de base et trajet prédit pour un bateau au hasard')
plt.legend()
plt.show()










