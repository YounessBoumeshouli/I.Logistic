import pandas as pd
import random
import os

# Paths
INPUT_FILE = "/app/villes_a_geocoder.csv"
OUTPUT_FILE = "/app/geocoded_cities.csv"


def generate_random_coordinates():
    try:
        # Read the cities file
        # Ensure we read it with the correct encoding if it was saved with ISO-8859-1
        # If main.py saved it, it might be UTF-8 or ISO-8859-1 depending on your latest run.
        # We'll try reading with default (UTF-8) first, then fallback.
        try:
            df = pd.read_csv(INPUT_FILE)
        except UnicodeDecodeError:
            df = pd.read_csv(INPUT_FILE, encoding="ISO-8859-1")

        print(f"Read {len(df)} cities from {INPUT_FILE}")

        # Generate random coordinates
        # Latitude: -90 to 90
        # Longitude: -180 to 180

        # We use a lambda function to generate a new random number for each row
        df['Dest_Lat'] = df.apply(lambda row: round(random.uniform(-90, 90), 6), axis=1)
        df['Dest_Lon'] = df.apply(lambda row: round(random.uniform(-180, 180), 6), axis=1)

        # Save to the output file
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully saved random coordinates to {OUTPUT_FILE}")
        print(df.head())

    except FileNotFoundError:
        print(f"Error: Input file {INPUT_FILE} not found. Please run the Streamlit app first to generate it.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    generate_random_coordinates()