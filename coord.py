import pandas as pd
import requests
import time
import os
from csv import writer as csv_writer

# --- CONFIGURATION ---
API_KEY = "2074b4e9a6b240f1b5a4afe6036f917b"  # Votre clé
OUTPUT_FILE = "geocoded_failures_output.csv"  # Nouveau fichier de sortie
OUTPUT_COLUMNS = ["Order City", "Order State", "Dest_Lat", "Dest_Lon"]

# --- L'ARRAY DES VILLES EN ÉCHEC (Votre liste) ---
# (J'ai corrigé les fautes de frappe comme Klaip?da -> Klaipėda)
villes_en_echec = [
    ("Jerusalén", "Jerusalén"), ("Bélgorod", "Bélgorod"), ("Cheliábinsk", "Cheliábinsk"),
    ("Eunápolis", "Bahía"), ("Caçador", "Santa Catarina"), ("Camagüey", "Camagüey"),
    ("Zanyán", "Zanyán"), ("San Cristóbal", "Táchira"), ("El Limón", "Aragua"),
    ("Namangán", "Namangán"), ("Mossoró", "Río Grande del Norte"), ("Fayún", "Fayún"),
    ("Jataí", "Goiás"), ("Gómel", "Gómel"), ("Araçatuba", "São Paulo"),
    ("Vínnytsia", "Vínnytsia"), ("Mazyr", "Gómel"), ("Querétaro", "Querétaro"),
    ("Asunción", "Asunción"), ("Córdoba", "Córdoba"), ("Santo André", "São Paulo"),
    ("Consolación del Sur", "Pinar del Río"), ("Astracán", "Astracán"), ("Mérida", "Yucatán"),
    ("San Luis Río Colorado", "Sonora"), ("San Miguelito", "Panamá"), ("Culiacán", "Sinaloa"),
    ("Shanghái", "Shanghái"), ("Kütahya", "Kütahya"), ("Ocotlán", "Jalisco"),
    ("San Pedro de Macorís", "San Pedro de Macorís"), ("Valparaíso", "Valparaíso"),
    ("Arraiján", "Panamá"), ("Camaçari", "Bahía"), ("San Luis Potosí", "San Luis Potosí"),
    ("Apatzingán de la Constitución", "Michoacán"), ("Anápolis", "Goiás"),
    ("Paysandú", "Paysandú"), ("Klaipėda", "Klaipėda"),  # Corrigé de Klaip?da
    ("Teherán", "Teherán"), ("Ilhéus", "Bahía"), ("Metz", "Alsacia-Champaña-Ardenas-Lorena"),
    ("Águas Lindas de Goiás", "Goiás"), ("Medellín", "Antioquía"), ("Tehuacán", "Puebla"),
    ("Járkov", "Járkov"), ("Grajaú", "Marañón"), ("Seúl", "Seúl"),
    ("Açu", "Río Grande del Norte"), ("Amatitlán", "Guatemala"), ("León", "León"),
    ("Vitória", "Espíritu Santo"), ("Colón", "Colón"), ("São Gonçalo", "Río de Janeiro"),
    ("Francisco Beltrão", "Paraná"), ("Lençóis Paulista", "São Paulo"),
    ("São José dos Campos", "São Paulo"), ("Kahramanmaraş", "Kahramanmaraş"),  # Corrigé
    ("Malambo", "Atlántico"), ("Lázaro Cárdenas", "Michoacán"), ("Holguín", "Holguín"),
    ("Melchor Ocampo", "México"), ("Santarém", "Pará"), ("Potosí", "Potosí"),
    ("Nancy", "Alsacia-Champaña-Ardenas-Lorena"), ("Concepción del Uruguay", "Entre Ríos"),
    ("Ibiúna", "São Paulo"), ("San Martín", "Cuscatlán"),
    ("San Francisco de Macorís", "Duarte"), ("Jacareí", "São Paulo"),
    ("Balneário Camboriú", "Santa Catarina"), ("Bolívar", "Bolívar"),
    ("Araucária", "Paraná"), ("Jundiaí", "São Paulo"), ("Ibagué", "Tolima"),
    ("Tetouan", "Tánger-Tetuán"), ("Sant Boi de Llobregat", "Cataluña"),
    ("Chillán", "Bío-Bío"), ("Ternópil", "Ternópil"), ("Valparaíso de Goiás", "Goiás"),
    ("Túnez", "Túnez"), ("Chisináu", "Chisináu"), ("Cuautitlán", "México"),
    ("Garza García", "Nuevo León"), ("Goiânia", "Goiás"), ("José Bonifácio", "São Paulo"),
    ("Neuquén", "Neuquén"), ("Qazvín", "Qazvín"), ("Rolândia", "Paraná"),
    ("Vitória de Santo Antão", "Pernambuco"), ("Cubatão", "São Paulo"),
    ("Bragança Paulista", "São Paulo"), ("Teziutlán", "Puebla"), ("Taubaté", "São Paulo"),
    ("Sumaré", "São Paulo"), ("Vorónezh", "Vorónezh"), ("Brașov", "Brașov"),  # Corrigé
    ("Jersón", "Jersón"), ("San Juan del Río", "Querétaro"), ("Kostanái", "Kostanái"),
    ("Cártama", "Andalucía"), ("Bisáu", "Bisáu"), ("Montbrison", "Auvernia-Ródano-Alpes"),
    ("Vesoul", "Borgoña-Franco Condado"), ("Vitória da Conquista", "Bahía"),
    ("Günzburg", "Bavaria"), ("Quibdó", "Chocó"), ("Zhytómyr", "Zhytómyr"),
    ("Çanakkale", "Çanakkale"), ("Catalão", "Goiás"), ("Apartadó", "Antioquía"),
    ("Taboão da Serra", "São Paulo"), ("Caluire-et-Cuire", "Auvernia-Ródano-Alpes"),
    ("Guimarães", "Braga"), ("Ciénaga", "Magdalena"), ("Mâcon", "Borgoña-Franco Condado"),
    ("Guamúchil", "Sinaloa"), ("Carapicuíba", "São Paulo"),
    ("San José de Guanipa", "Anzoátegui"), ("Semnán", "Semnán"),
    ("Poços de Caldas", "Minas Gerais"), ("Mérida", "Mérida"), ("Gyula", "Békés"),
    ("Bingöl", "Bingöl"), ("El Aaiún", "El Aaiún"), ("Montería", "Córdoba")
]


# ---------------------

def geocode_city(city, state):
    """Appelle l'API Geoapify pour une ville et un état."""
    if not API_KEY or API_KEY == "VOTRE_CLE_API_GEOAPIFY_ICI":
        print("ERREUR: Veuillez remplacer 'VOTRE_CLE_API_GEOAPIFY_ICI' par votre vraie clé API.")
        return None, None

    try:
        url = f"https://api.geoapify.com/v1/geocode/search?city={city}&state={state}&format=json&apiKey={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()  # Lève une exception si erreur HTTP
        data = response.json()

        if data['results']:
            best_result = None
            for result in data['results']:
                if result.get('result_type') == 'city':
                    best_result = result
                    break
            if not best_result:
                best_result = data['results'][0]

            lat = best_result.get('lat')
            lon = best_result.get('lon')

            if lat and lon:
                print(f"  -> Succès pour {city}, {state}: ({lat}, {lon})")
                return lat, lon
            else:
                print(f"  -> Résultat trouvé mais sans lat/lon pour {city}, {state}")
                return None, None
        else:
            print(f"  -> Pas de résultats pour {city}, {state}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"  -> ERREUR API pour {city}, {state}: {e}")
        return None, None


def main():
    # Gérer le fichier de sortie
    write_header = not os.path.exists(OUTPUT_FILE)

    if write_header:
        print(f"Création d'un nouveau fichier de sortie '{OUTPUT_FILE}'...")
        # Écrire l'en-tête
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv_writer(f)
            writer.writerow(OUTPUT_COLUMNS)
    else:
        print(f"Le fichier de sortie '{OUTPUT_FILE}' existe déjà. Ajout des nouveaux résultats...")

    # Lire les villes déjà traitées dans CE fichier (pour la reprise)
    try:
        df_existant = pd.read_csv(OUTPUT_FILE)
        villes_deja_faites = set(zip(df_existant["Order City"], df_existant["Order State"]))
    except pd.errors.EmptyDataError:
        villes_deja_faites = set()

    print(f"Déjà traité dans ce fichier: {len(villes_deja_faites)}")

    # Ouvrir le fichier en mode "append" (ajout)
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv_writer(f)

        # Itérer sur la liste (array) en dur
        for city_state_tuple in villes_en_echec:
            city, state = city_state_tuple

            # Vérifier si on a déjà traité cette ville DANS CE FICHIER
            if (city, state) not in villes_deja_faites:
                print(f"Traitement de la ville : {city}, {state}")

                lat, lon = geocode_city(city, state)

                new_row = [city, state, lat, lon]
                writer.writerow(new_row)
                f.flush()

                print("...pause de 1.1 seconde pour respecter les limites de l'API...")
                time.sleep(1.1)
            else:
                print(f"Ignoré (déjà dans {OUTPUT_FILE}): {city}, {state}")

    print("Script de géocodage manuel terminé.")


if __name__ == "__main__":
    main()