import pandas as pd
import os
from zipfile import ZipFile

# Configuration des colonnes à lire et des répertoires
cols_to_read = ['MMSI', 'Vessel_Name', 'IMO', 'Ship_Type', 'Dimension_to_Bow', 'Dimension_to_stern', 'Dimension_to_port', 'Dimension_to_starboard']
input_directory = r'D:\Vincent\datadl\backfill_2012-2022'  # Remplace par le répertoire contenant les sous-répertoires annuels
save_directory = r'D:\Vincent\Cleandata_valid_transform'  # Remplace par le répertoire de sauvegarde

def process_files_in_directory(directory, save_directory):
    # DataFrame global pour stocker toutes les données
    all_data = pd.DataFrame()
    
    try:
        # Parcours des sous-répertoires par année
        for year in os.listdir(directory):
            file_year = os.path.join(directory, year)
            if not os.path.isdir(file_year):
                continue
                
            # Parcours des sous-répertoires par mois
            for month in os.listdir(file_year):
                file_month = os.path.join(file_year, month)
                if not os.path.isdir(file_month):
                    continue
                    
                # Création du répertoire de sauvegarde pour chaque mois
                save_month = os.path.join(save_directory, year, month)
                os.makedirs(save_month, exist_ok=True)
                
                # Parcours des sous-répertoires par jour
                for day in os.listdir(file_month):
                    file_day = os.path.join(file_month, day)
                    
                    if not os.path.isfile(file_day):
                        continue
                    
                    if file_day.endswith('.zip'):
                        with ZipFile(file_day, 'r') as zip_ref:
                            temp_dir = os.path.join(directory, 'temp')
                            os.makedirs(temp_dir, exist_ok=True)  # Assure que le répertoire temporaire existe
                            zip_ref.extractall(temp_dir)

                            for extracted_file_name in os.listdir(temp_dir):
                                extracted_file_path = os.path.join(temp_dir, extracted_file_name)
                                
                                # Vérifie que le fichier extrait est bien un CSV
                                if not extracted_file_name.endswith('.csv'):
                                    continue
                                
                                # Lire le fichier CSV dans un DataFrame temporaire
                                try:
                                    df = pd.read_csv(extracted_file_path, usecols=cols_to_read)
                                    
                                    # Vérifie si les colonnes nécessaires sont présentes
                                    if not set(cols_to_read).issubset(df.columns):
                                        print(f"Les colonnes requises ne sont pas toutes présentes dans {extracted_file_path}")
                                        continue
                                    
                                    # Filtrer les lignes où la colonne 'IMO' n'est pas vide
                                    filtered_df = df[df['IMO'].notna() & (df['IMO'] != '')]

                                    # Créer de nouvelles colonnes pour les sommes
                                    filtered_df['Lenght'] = filtered_df['Dimension_to_Bow'] + filtered_df['Dimension_to_stern']
                                    filtered_df['Width'] = filtered_df['Dimension_to_port'] + filtered_df['Dimension_to_starboard']

                                    # Sélectionner uniquement les colonnes nécessaires
                                    filtered_df = filtered_df[['MMSI', 'Vessel_Name', 'IMO', 'Ship_Type', 'Lenght', 'Width']]

                                    # Supprimer les doublons en gardant la première occurrence pour chaque MMSI
                                    filtered_df = filtered_df.drop_duplicates(subset='MMSI', keep='first')

                                    # Ajouter les données au DataFrame global
                                    all_data = pd.concat([all_data, filtered_df], ignore_index=True)

                                except Exception as e:
                                    print(f"Erreur lors de la lecture du fichier {extracted_file_path}: {e}")
                                
                                # Nettoyer le répertoire temporaire après utilisation
                                for file in os.listdir(temp_dir):
                                    os.remove(os.path.join(temp_dir, file))
                                os.rmdir(temp_dir)

                    elif file_day.endswith('.csv'):
                        # Traiter les fichiers CSV non compressés
                        try:
                            df = pd.read_csv(file_day, usecols=cols_to_read)

                            # Vérifie si les colonnes nécessaires sont présentes
                            if not set(cols_to_read).issubset(df.columns):
                                print(f"Les colonnes requises ne sont pas toutes présentes dans {file_day}")
                                continue

                            # Filtrer les lignes où la colonne 'IMO' n'est pas vide
                            filtered_df = df[df['IMO'].notna() & (df['IMO'] != '')]

                            # Créer de nouvelles colonnes pour les sommes
                            filtered_df['Lenght'] = filtered_df['Dimension_to_Bow'] + filtered_df['Dimension_to_stern']
                            filtered_df['Width'] = filtered_df['Dimension_to_port'] + filtered_df['Dimension_to_starboard']

                            # Sélectionner uniquement les colonnes nécessaires
                            filtered_df = filtered_df[['MMSI', 'Vessel_Name', 'IMO', 'Ship_Type', 'Lenght', 'Width']]

                            # Supprimer les doublons en gardant la première occurrence pour chaque MMSI
                            filtered_df = filtered_df.drop_duplicates(subset='MMSI', keep='first')

                            # Ajouter les données au DataFrame global
                            all_data = pd.concat([all_data, filtered_df], ignore_index=True)

                        except Exception as e:
                            print(f"Erreur lors de la lecture du fichier {file_day}: {e}")

        # Supprimer les doublons en gardant la première occurrence pour chaque MMSI dans le DataFrame global
        all_data = all_data.drop_duplicates(subset='MMSI', keep='first')
        
        # Trier les données par MMSI
        all_data.sort_values(by=['MMSI'], inplace=True)
        
        # Sauvegarder le DataFrame filtré dans un nouveau fichier CSV
        output_csv = os.path.join(save_directory, 'final_output.csv')
        all_data.to_csv(output_csv, index=False)
        
        print(f"Le fichier filtré a été sauvegardé sous {output_csv}")
        
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

# Exécution du traitement
process_files_in_directory(input_directory, save_directory)
