# scripts/run_training_job.py
import os
import sys

# Ajouter le dossier parent au path pour importer vos modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from train_pipeline import build_and_train
from clean_data import remove_nulls


def run_job():
    print("--- Démarrage du Job d'Entraînement Automatisé ---")

    # 1. Initialiser Spark
    spark = SparkSession.builder \
        .appName("AirflowBatchTraining") \
        .master("local[*]") \
        .getOrCreate()

    # 2. Charger les données
    # Assurez-vous que le chemin est accessible depuis le conteneur Airflow
    df = spark.read.csv("/app/dataset_clean.csv", header=True, inferSchema=True)

    # 3. Nettoyage (Copie de la logique de main.py)
    df_clean = remove_nulls(df)

    # 4. Entraînement
    # On force ici le modèle 'Random Forest' pour l'automatisation
    model_name = "Random Forest"
    print(f"Entraînement du modèle : {model_name}")

    cvModel, _, _ = build_and_train(df_clean, model_name)

    # La sauvegarde est déjà gérée dans build_and_train,
    # mais on peut ajouter des logs ici.
    print("Job terminé avec succès.")
    spark.stop()


if __name__ == "__main__":
    run_job()