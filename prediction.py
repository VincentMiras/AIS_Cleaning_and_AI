import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from scipy.interpolate import interp1d

# Charger les données AIS avec latitude, longitude, temps et MMSI
csv_filename = './exactEarth_historical_data_2023-01-05_modif2.csv'
df = pd.read_csv(csv_filename)

# Sélectionner un MMSI spécifique pour l'entraînement
selected_mmsi = df['MMSI'].unique()[11]
selected_sequence = df[df['MMSI'] == selected_mmsi][['Latitude', 'Longitude']].values

# Interpolation quadratique des données de trajectoire
x = np.arange(len(selected_sequence))
interp_func_lat = interp1d(x, selected_sequence[:, 0], kind='quadratic', fill_value='extrapolate')
interp_func_lon = interp1d(x, selected_sequence[:, 1], kind='quadratic', fill_value='extrapolate')

# Générer des points interpolés plus fins pour un affichage plus fluide
x_interp = np.linspace(0, len(selected_sequence)-1, 100)
interp_lat = interp_func_lat(x_interp)
interp_lon = interp_func_lon(x_interp)
y_train = np.column_stack((interp_lon[:29], interp_lat[:29]))


# Normalisation des données pour l'entraînement LSTM
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_sequence)

# Longueur de la séquence de données d'entrée pour le modèle LSTM
sequence_length = 28

# Génération des séquences temporelles pour l'entraînement du modèle LSTM
generator = TimeseriesGenerator(scaled_data, scaled_data,
                                length=sequence_length, batch_size=1)

# Créer un modèle LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 2)),
    Dense(2)
])

# Compilation du modèle avec l'optimiseur et la fonction de perte
model.compile(optimizer='adam', loss='mse')

# Entraînement du modèle LSTM
model.fit(generator, epochs=75)  # Increase the number of epochs for better training
print('a')
# Prédiction à partir de la séquence initiale
predicted_positions = []

# Utilisation de la dernière partie de la séquence initiale comme point de départ
last_sequence = scaled_data[-sequence_length:].reshape((1, sequence_length, 2))

# Nombre de points à prédire
num_points = 2

for _ in range(num_points):
    # Prédire le prochain point
    next_position = model.predict(last_sequence)
    
    # Ajouter la prédiction à la liste des positions prédites
    predicted_positions.append(next_position.flatten())
    
    # Mettre à jour la séquence d'entrée avec la nouvelle prédiction
    last_sequence = np.append(last_sequence[:, 1:, :], next_position.reshape(1, 1, 2), axis=1)

# Convertir la liste des positions prédites en array
predicted_positions = np.array(predicted_positions)

# Dénormalisation des positions prédites
predicted_positions = scaler.inverse_transform(predicted_positions)

# Tracer le trajet du bateau en bleu


# Tracer les prédictions du modèle en rouge
plt.plot(predicted_positions[:, 1], predicted_positions[:, 0], marker='o', markersize=5, color='red')
plt.plot(selected_sequence[:, 1], selected_sequence[:, 0], label='Trajet du bateau', color='blue')
plt.plot(interp_lon, interp_lat, label='Interpolation quadratique', color='green')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Trajet du bateau avec prédictions du modèle LSTM')
plt.legend()
plt.show()

