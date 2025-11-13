import os
import streamlit as st
import pandas as pd
import time
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, Imputer, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.metrics import confusion_matrix

@st.cache_resource()
def get_spark_session() -> SparkSession:
    """
    Crée (une seule fois) et retourne une session Spark optimisée.
    """
    print("--- CRÉATION D'UNE NOUVELLE SESSION SPARK ---")


    # URL du master (local[*] pour le dev, ou URL du master Spark)
    spark_master_url = os.environ.get("SPARK_MASTER_URL", "local[*]")

    # Configuration MongoDB
    mongo_url = "mongodb://mongo:27017/bank_attrition"

    return (
        SparkSession.builder
        .appName("AttritionPrediction")
        .master(spark_master_url)
        .getOrCreate()
    )
spark = get_spark_session()
st.title("Prjet de Prédiction d'Attrition Client (PySpark, MLlib, MongoDB)")

st.header("Étape 1 : Configuration et Initialisation de Spark")
st.write(f"Session Spark démarrée. Version : **{spark.version}**")

DATA_FILE = "/app/datasets/DataCoSupplyChainDataset.csv"
def load_data(file_path):
    """Charge les données CSV dans un DataFrame Spark."""
    try:
        return spark.read.csv(file_path, header=True, inferSchema=True, sep=',')
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier {file_path}: {e}")
        st.error("Assurez-vous que le fichier 'data-set.csv' est présent dans le même répertoire.")
        return None


df = load_data(DATA_FILE)

if df:
    st.write("✅ Données brutes chargées avec succès.")

    st.write("Aperçu des 5 premières lignes :")
    st.dataframe(df.limit(5).toPandas())
    cols = [
        # Variable Cible
        'Late_delivery_risk',

        # Features (Variables Explicatives)
        'Days for shipment (scheduled)',
        'order date (DateOrders)',
        'Shipping Mode',
        'Order Region',
        'Latitude',
        'Longitude',
        'Category Name',
        'Department Name',
        'Order Item Quantity',
        'Product Status',
        'Customer Segment',
        'Order Item Total',
        'Order Profit Per Order',
        'Order Item Discount Rate'
    ]
    new_df = df.select(cols)
    st.dataframe(new_df.limit(5).toPandas())

    st.dataframe(new_df.select("Late_delivery_risk").distinct().toPandas())
    st.dataframe(new_df.select("Category Name").distinct().toPandas())
    st.dataframe(new_df.select("Customer Segment").distinct().toPandas())
    st.dataframe(new_df.select("Market").distinct().toPandas())
    st.dataframe(new_df.select("Shipping Mode").distinct().toPandas())
    st.dataframe(new_df.select("Latitude" , "Longitude").distinct().toPandas())
    st.dataframe(new_df.select("Market","Order Region").distinct().orderBy("Market").toPandas())


    st.dataframe(new_df.distinct().toPandas())
    st.text('duplicated rows :')
    st.text(len(new_df.distinct().toPandas()) - len(new_df.toPandas()))
    st.write("Schéma des colonnes :")
    # Capture du printSchema pour l'afficher dans Streamlit
    from io import StringIO
    import sys

    old_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output
    new_df.printSchema()
    sys.stdout = old_stdout

    st.text(redirected_output.getvalue())
    st.text("missing_counts")

    numeric_cols = [c.split(':')[0] for c in redirected_output]
    for n in numeric_cols:
        st.text(n)
    missing_counts = new_df.select([
        F.count(
            F.when(
                F.col(c).isNull | F.isnan(F.col(c)), 1
            )
        ).alias(c)
        for c in new_df.columns
    ])
    st.text(missing_counts)