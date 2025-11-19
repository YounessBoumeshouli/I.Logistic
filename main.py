import os
import streamlit as st
import pandas as pd
import time
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F
# Supprimer l'import redondant et ne garder que le plus complet
from pyspark.sql.types import StringType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, Imputer, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics # <-- Utilisé pour les métriques de validation
# from sklearn.metrics import confusion_matrix # <-- Inutile, on utilise la version PySpark

# Imports pour l'EDA (Visualisation)
import matplotlib.pyplot as plt
import seaborn as sns

# --- NOUVEL IMPORT ---
# Importer la fonction depuis le nouveau fichier
try:
    from train_pipeline import build_and_train

    st.write("Module `train_pipeline.py` chargé.")
except ImportError:
    st.error("ERREUR: Fichier `train_pipeline.py` non trouvé.")


@st.cache_resource()
def get_spark_session() -> SparkSession:
    """
    Crée (une seule fois) et retourne une session Spark optimisée.
    """
    print("--- CRÉATION D'UNE NOUVELLE SESSION SPARK ---")

    # URL du master (local[*] pour le dev, ou URL du master Spark)
    spark_master_url = os.environ.get("SPARK_MASTER_URL", "local[*]")

    return (
        SparkSession.builder
        .appName("AttritionPrediction")
        .master(spark_master_url)
        .getOrCreate()
    )


spark = get_spark_session()

st.title("Prédiction des Retards de Livraison (PySpark MLlib)")

st.header("Étape 1 : Chargement des Données")
st.write(f"Session Spark démarrée. Version : **{spark.version}**")

DATA_FILE = "dataset_clean.csv"


def load_data(file_path):
    """Charge les données CSV dans un DataFrame Spark."""
    try:
        # --- CORRECTION DE L'ENCODAGE ---
        # Ajouter .option("encoding", "ISO-8859-1")
        return spark.read.csv(
            file_path,
            header=True,
            inferSchema=True,
            sep=',',
            encoding="ISO-8859-1"  # ou "latin1"
        )
        # --- FIN DE LA CORRECTION ---
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier {file_path}: {e}")
        return None


new_df = load_data("dataset_clean.csv")

if new_df:
    st.write("✅ Données brutes chargées avec succès.")

    st.header("Étape 2: Analyse Exploratoire des Données (EDA)")

    st.subheader("Distribution des Modes de Livraison (Shipping Mode)")
    shipping_dist_spark = (new_df
                           .groupBy("Shipping Mode")
                           .count()
                           .orderBy(F.desc("count"))
                           )
    shipping_dist_pd = shipping_dist_spark.toPandas()
    st.dataframe(shipping_dist_pd)
    st.bar_chart(shipping_dist_pd.set_index("Shipping Mode"))

    # -----------------------------------------------------------------
    # --- PLAN DE GÉOCODAGE ÉTAPE 1: EXPORTER LES VILLES UNIQUES ---
    # -----------------------------------------------------------------
    # st.header("Étape 3: Préparation pour le Géocodage (Exporter les villes)")
    #
    # GEOCODING_INPUT_FILE = "/app/villes_a_geocoder.csv"
    #
    # villes_uniques_df = new_df.select("Order City", "Order State").distinct()
    #
    # st.write(f"Nombre de lignes totales : {new_df.count()}")
    # st.write(f"Nombre de combinaisons (Ville, État) uniques à geocoder : {villes_uniques_df.count()}")
    #
    # # Sauvegarder ce fichier pour que le script (Etape 2) puisse l'utiliser
    # # Nous le faisons à chaque fois pour garantir que le fichier existe.
    # try:
    #     # Utiliser l'encodage ISO-8859-1 pour sauvegarder les caractères spéciaux
    #     villes_uniques_df.toPandas().to_csv(GEOCODING_INPUT_FILE, index=False, encoding="ISO-8859-1")
    #     if not os.path.exists(GEOCODING_INPUT_FILE):
    #         st.success(f"Fichier `{GEOCODING_INPUT_FILE}` créé.")
    #         st.info("Vous pouvez maintenant exécuter le script `coord.py` dans un autre terminal.")
    #     else:
    #         st.write(f"Le fichier `{GEOCODING_INPUT_FILE}` a été mis à jour.")
    # except Exception as e:
    #     st.error(f"Erreur lors de la sauvegarde du fichier des villes: {e}")

    # -----------------------------------------------------------------
    # --- PLAN DE GÉOCODAGE ÉTAPE 3: IMPORTER ET JOINDRE LES COORDONNÉES ---
    # -----------------------------------------------------------------
    st.header("Étape 4: Enrichissement des Données (Jointure Géocodée)")

    # GEOCODED_FILE = "/app/geocoded_cities.csv"
    #
    # try:
    #     # Charger les données géocodées (le résultat de l'Etape 2)
    #     geocoded_cities_df = spark.read.csv(GEOCODED_FILE, header=True, inferSchema=True)
    #
    #     st.write(f"✅ Données géocodées de `{GEOCODED_FILE}` chargées avec succès.")
    #     st.dataframe(geocoded_cities_df.filter(F.col("Dest_Lat").isNotNull()).limit(5).toPandas())
    #
    #     # Faire la jointure
    #     df_enrichi = new_df.join(
    #         geocoded_cities_df,
    #         # Assurer une jointure propre même avec des nuls
    #         (new_df["Order City"].eqNullSafe(geocoded_cities_df["Order City"])) &
    #         (new_df["Order State"].eqNullSafe(geocoded_cities_df["Order State"])),
    #         how="left"
    #     ).drop(geocoded_cities_df["Order City"]).drop(
    #         geocoded_cities_df["Order State"])  # Éviter les colonnes dupliquées
    #
    #     st.write("✅ Jointure effectuée! `new_df` est maintenant enrichi avec `Dest_Lat` et `Dest_Lon`.")
    #
    #     # Afficher le nouveau DataFrame enrichi
    #     st.dataframe(df_enrichi.limit(5).toPandas())
    #
    #     # Remplacer new_df pour les analyses futures
    #     new_df = df_enrichi
    #
    #     # (Vous pouvez maintenant calculer la distance)
    #
    # except Exception as e:
    #     st.warning(f"Impossible de charger le fichier '{GEOCODED_FILE}'.")
    #     st.info("Avez-vous exécuté le script `coord.py` ? Si oui, est-il terminé ?")
    #     # st.error(e) # Décommentez pour le débogage

    # -----------------------------------------------------------------
    # --- PROCHAINES ÉTAPES: PRÉTRAITEMENT & MODÉLISATION ---
    # -----------------------------------------------------------------
    st.header("Étape 5: Prétraitement et Pipeline ML")

    # --- CORRECTION: AJOUT DU SELECTBOX ---
    st.subheader("1. Choisissez votre modèle")
    model_choice = st.selectbox(
        "Quel modèle souhaitez-vous entraîner et évaluer ?",
        (
            "Logistic Regression",
            "Random Forest",
            "Gradient-Boosted Trees (GBT)"
        )
    )
    st.info(f"Vous avez sélectionné : **{model_choice}**")

    if st.button(f"Lancer l'entraînement pour {model_choice} (peut prendre 10-20 mins)"):

        if 'new_df' in locals() and new_df.columns:

            try:
                # --- CORRECTION: MISE À JOUR DU SPINNER ---
                with st.spinner(
                        f"Entraînement de {model_choice} en cours... Application des optimisations (Repartition et Parallelism=1)..."):
                    # --- CORRECTION: PASSER LE 2ÈME ARGUMENT ---
                    cvModel, testData, evaluator = build_and_train(new_df, model_choice)

                st.success(f"✅ Entraînement de {model_choice} terminé avec succès !")

                # --- Évaluation du Modèle ---
                st.subheader(f"Évaluation du Modèle : {model_choice}")
                with st.spinner("Évaluation des performances sur les données de test..."):
                    predictions = cvModel.transform(testData)

                    # Calculer l'AUC (Area Under Curve)
                    auc = evaluator.evaluate(predictions)
                    st.metric(label="**Area Under ROC (AUC)** sur données de test", value=f"{auc:.4f}")

                    st.write("Plus ce score est proche de 1.0, meilleur est le modèle.")

                    # -----------------------------------------------------------------
                    # --- AJOUT DES METRIQUES DE VALIDATION ET MATRICE DE CONFUSION ---
                    # -----------------------------------------------------------------

                    st.subheader("Mesures de Validation Détaillées (Accuracy, F1, Précision, Rappel)")

                    # Préparer les données pour MulticlassMetrics
                    # MulticlassMetrics nécessite un RDD de (prediction, label)
                    predictionAndLabels = (
                        predictions.select("prediction", "Late_delivery_risk")
                        .rdd.map(lambda row: (float(row.prediction), float(row.Late_delivery_risk)))
                    )

                    # Instancier l'objet metrics
                    metrics = MulticlassMetrics(predictionAndLabels)

                    # Statistiques générales
                    accuracy = metrics.accuracy
                    weightedF1 = metrics.weightedFMeasure()
                    weightedPrecision = metrics.weightedPrecision
                    weightedRecall = metrics.weightedRecall

                    # Matrice de Confusion (convertir le RDD en Array, puis en DataFrame Pandas pour l'affichage)
                    conf_matrix_rdd = metrics.confusionMatrix().toArray()
                    conf_matrix_df = pd.DataFrame(
                        conf_matrix_rdd,
                        index=['Vrai Négatif (0: Non Retard)', 'Vrai Positif (1: Retard)'],
                        columns=['Prédit Négatif (0)', 'Prédit Positif (1)']
                    ).astype(int) # Afficher les nombres entiers

                    # Afficher les métriques
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy (Précision Globale)", f"{accuracy:.4f}")
                    col2.metric("F1-Score Pondéré", f"{weightedF1:.4f}")
                    col3.metric("Précision Pondérée", f"{weightedPrecision:.4f}")
                    col4.metric("Rappel Pondéré (Recall)", f"{weightedRecall:.4f}")

                    st.markdown("---")

                    # Afficher la Matrice de Confusion
                    st.subheader("Matrice de Confusion")
                    st.dataframe(conf_matrix_df)
                    st.write("Le F1-Score et la Précision/Rappel pondérés tiennent compte du déséquilibre des classes.")

                    # --- FIN DE L'AJOUT DES METRIQUES ---
                    # -----------------------------------------------------------------

                    # --- 2. CORRECTION DE L'AFFICHAGE (Déjà présent) ---

                    # Définir une UDF (User Defined Function) pour convertir le vecteur en string
                    vec_to_string_udf = F.udf(lambda v: str(v), StringType())

                    # UDF pour extraire la probabilité de la classe 1 (le 2ème élément)
                    # v[1] est la probabilité d'être en retard
                    extract_prob_udf = F.udf(lambda v: float(v[1]), FloatType())

                    # Appliquer les DEUX UDFs
                    predictions_display = predictions.withColumn(
                        "features_str", vec_to_string_udf(F.col("features"))
                    ).withColumn(
                        "Prob_En_Retard", extract_prob_udf(F.col("probability"))  # Créer une colonne propre
                    )

                    st.subheader("Aperçu des Prédictions")
                    # Afficher le nouveau DataFrame avec les colonnes converties
                    st.dataframe(
                        predictions_display.select("features_str", "Prob_En_Retard", "prediction", "Late_delivery_risk")
                        .toPandas()
                    )

            except Exception as e:
                st.error("Erreur lors de l'entraînement du modèle:")
                st.exception(e)  # Affiche l'erreur complète (OutOfMemory, etc.)
        else:
            st.error("Le DataFrame `new_df` n'a pas pu être chargé ou enrichi. L'entraînement est annulé.")