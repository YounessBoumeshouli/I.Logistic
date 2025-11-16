import pandas as pd
import requests
import time
import os
from csv import writer as csv_writer  # Import pour l'écriture en mode append

# --- CONFIGURATION ---
# La clé API que vous avez fournie
API_KEY = "2074b4e9a6b240f1b5a4afe6036f917b"

INPUT_FILE = "villes_a_geocoder.csv"
OUTPUT_FILE = "geocoded_cities.csv"

OUTPUT_COLUMNS = ["Order City", "Order State", "Dest_Lat", "Dest_Lon"]


# ---------------------

def geocode_city(city, state):
    """Appelle l'API Geoapify pour une ville et un état."""
    if not API_KEY or API_KEY == "VOTRE_CLE_API_GEOAPIFY_ICI":
        print("ERREUR: Veuillez remplacer 'VOTRE_CLE_API_GEOAPIFY_ICI' par votre vraie clé API.")
        return None, None

    try:
        # Formatter la requête
        url = f"https://api.geoapify.com/v1/geocode/search?city={city}&state={state}&format=json&apiKey={API_KEY}"

        response = requests.get(url)
        response.raise_for_status()  # Lève une exception si erreur HTTP
        data = response.json()

        if data['results']:
            # Prend le premier résultat le plus pertinent
            lat = data['results'][0]['lat']
            lon = data['results'][0]['lon']
            print(f"  -> Succès pour {city}, {state}: ({lat}, {lon})")
            return lat, lon
        else:
            print(f"  -> Pas de résultats pour {city}, {state}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"  -> ERREUR API pour {city}, {state}: {e}")
        return None, None


def main():
    try:
        # L'encodage que vous avez ajouté
        villes_a_geocoder = pd.read_csv(INPUT_FILE, encoding="ISO-8859-1")
    except FileNotFoundError:
        print(f"ERREUR: Fichier d'entrée '{INPUT_FILE}' non trouvé.")
        print("Veuillez d'abord lancer l'application Streamlit (main.py) pour générer ce fichier.")
        return

    # Fichier de sortie et gestion de l'en-tête
    write_header = not os.path.exists(OUTPUT_FILE)

    if write_header:
        print(f"Création d'un nouveau fichier de sortie '{OUTPUT_FILE}'...")
        geocoded_cities = pd.DataFrame(columns=OUTPUT_COLUMNS)
        geocoded_cities.to_csv(OUTPUT_FILE, index=False)  # Crée le fichier avec en-tête
    else:
        print(f"Reprise du travail à partir de '{OUTPUT_FILE}'...")
        geocoded_cities = pd.read_csv(OUTPUT_FILE)

    # --- CORRECTION ---
    # On ne considère une ville comme "faite" que si elle a une latitude ET une longitude
    # Si 'Dest_Lat' est nul (NaN), 'dropna' la supprimera.
    villes_succes_df = geocoded_cities.dropna(subset=['Dest_Lat', 'Dest_Lon'])

    # Le set des villes "déjà faites" ne contient maintenant que les SUCCÈS
    villes_deja_faites = set(
        zip(
            villes_succes_df["Order City"].astype(str),
            villes_succes_df["Order State"].astype(str)
        )
    )
    # --- FIN CORRECTION ---

    total_villes = len(villes_a_geocoder)
    print(f"Total de villes uniques à traiter: {total_villes}")
    print(f"Villes déjà géocodées avec succès: {len(villes_deja_faites)}")

    # Ouvrir le fichier en mode "append" (ajout)
    with open(OUTPUT_FILE, 'a', newline='',encoding="ISO-8859-1") as f:
        writer = csv_writer(f)

        # Note: Nous n'écrivons pas l'en-tête ici car il est déjà géré par 'write_header'

        for index, row in villes_a_geocoder.iterrows():
            city = str(row["Order City"]) if pd.notna(row["Order City"]) else "N/A"
            state = str(row["Order State"]) if pd.notna(row["Order State"]) else "N/A"
            ville_tuple = (city, state)

            # Si la ville n'est PAS dans le set des SUCCÈS
            if ville_tuple not in villes_deja_faites:
                print(f"Traitement de la ville ({index + 1}/{total_villes}): {city}, {state}")

                lat, lon = None, None
                if city != "N/A" and state != "N/A":
                    lat, lon = geocode_city(city, state)
                else:
                    print(f"  -> Ignoré (données manquantes): {city}, {state}")

                # Préparer la nouvelle ligne
                new_row = [city, state, lat, lon]

                # ÉCRIRE la nouvelle ligne
                writer.writerow(new_row)
                f.flush()  # Forcer l'écriture sur le disque

                # Mettre à jour le set pour éviter les doublons dans cette session
                # (Même si c'est un échec, on l'ajoute pour ne pas le refaire
                # DANS CETTE MÊME SESSION. Au prochain redémarrage, il sera retraité)
                villes_deja_faites.add(ville_tuple)

                # --- RESPECTER LA LIMITE DE L'API (1 req/sec) ---
                print("...pause de 1.1 seconde pour respecter les limites de l'API...")
                # time.sleep(0.5)

    print("Géocodage terminé (ou mis en pause).")

main()