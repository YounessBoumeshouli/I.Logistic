from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window
from pyspark.sql.types import StructType, StringType, IntegerType, FloatType, TimestampType
from pyspark.ml import PipelineModel

# 1. Initialisation
spark = SparkSession.builder \
    .appName("LogisticRealTimePrediction") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. Chargement du Modèle Entraîné
# Assurez-vous d'utiliser le bon chemin (celui que nous avons vu précédemment)
MODEL_PATH = "/app/spark_models/random_forest" # ou logistic_regression
try:
    loaded_model = PipelineModel.load(MODEL_PATH)
    print("✅ Modèle chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur de chargement du modèle: {e}")
    # En production, on arrêterait le script ici.

# 3. Lecture du flux TCP
lines = spark.readStream \
    .format("socket") \
    .option("host", "tcp-bridge") \
    .option("port", 9999) \
    .load()

# 4. Définition du Schéma (DOIT correspondre aux données de l'API)
json_schema = StructType() \
    .add("order_date", StringType()) \
    .add("Order City", StringType()) \
    .add("Customer Segment", StringType()) \
    .add("Shipping Mode", StringType()) \
    .add("Order Item Quantity", IntegerType()) \
    .add("Order Item Total", FloatType()) \
    .add("Order Profit Per Order", FloatType()) \
    .add("Latitude", FloatType()) \
    .add("Longitude", FloatType())
    # Ajoutez les autres champs ici...

# 5. Parsing et Transformation
orders_df = lines.select(from_json(col("value"), json_schema).alias("data")).select("data.*")

# Convertir la date en Timestamp pour le windowing
orders_df = orders_df.withColumn("timestamp", col("order_date").cast(TimestampType()))

# --- APPLICATION DU MODÈLE ---
# Le modèle attend un DataFrame avec les colonnes brutes.
# Il appliquera lui-même le StringIndexer, VectorAssembler, etc.
predictions_df = loaded_model.transform(orders_df)

# Sélectionner les résultats intéressants
results_df = predictions_df.select(
    "timestamp",
    "Order City",
    "prediction", # 0 ou 1
    "probability"
)

# 6. Agrégations (Insights) avec Windowing
# Compter les retards prédits par fenêtre de 1 minute
insights_df = results_df \
    .withWatermark("timestamp", "1 minute") \
    .groupBy(window(col("timestamp"), "1 minute"), "Order City") \
    .agg({"prediction": "sum"}) \
    .withColumnRenamed("sum(prediction)", "predicted_late_orders")

# 7. Écriture (Sink)
# Pour l'instant, affichons dans la console pour tester.
# Plus tard, nous écrirons dans Mongo/Postgres via `foreachBatch`.

query = results_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()