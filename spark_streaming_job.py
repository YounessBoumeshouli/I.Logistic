from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, current_timestamp
from pyspark.sql.types import StructType, StringType, IntegerType, FloatType, TimestampType
from pyspark.ml import PipelineModel
import os

# 1. Initialize Spark Session
# We need to include the MongoDB and Postgres drivers packages here if not already in the base image
# But since we configured PYTHONPATH in compose.yaml, we assume drivers are available or we use simple python libs.
# For simplicity in this setup, we will use standard Python libraries (pymongo, psycopg2) inside foreachBatch
# because adding JARs to Spark Submit in Docker can be tricky without internet.

spark = SparkSession.builder \
    .appName("LogisticRealTimePrediction") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. Load Resources (Model & Geo Data)
MODEL_PATH = "/app/spark_models/random_forest"
GEO_FILE = "/app/geocoded_cities.csv"

try:
    if os.path.exists(MODEL_PATH):
        loaded_model = PipelineModel.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        loaded_model = None
except Exception:
    loaded_model = None

try:
    if os.path.exists(GEO_FILE):
        geo_df = spark.read.csv(GEO_FILE, header=True, inferSchema=True)
        geo_df = geo_df.select("Order City", "Order State", "Dest_Lat", "Dest_Lon")
        print(f"✅ Geo data loaded.")
    else:
        geo_df = None
except Exception:
    geo_df = None

# 3. Read Stream
lines = spark.readStream \
    .format("socket") \
    .option("host", "tcp-bridge") \
    .option("port", 9999) \
    .load()

# 4. Schema
json_schema = StructType() \
    .add("order_date", StringType()) \
    .add("Order City", StringType()) \
    .add("Order State", StringType()) \
    .add("Customer Segment", StringType()) \
    .add("Shipping Mode", StringType()) \
    .add("Market", StringType()) \
    .add("Order Region", StringType()) \
    .add("Category Name", StringType()) \
    .add("Department Name", StringType()) \
    .add("Product Status", StringType()) \
    .add("Order Item Quantity", IntegerType()) \
    .add("Order Item Total", FloatType()) \
    .add("Order Profit Per Order", FloatType()) \
    .add("Order Item Discount Rate", FloatType()) \
    .add("Latitude", FloatType()) \
    .add("Longitude", FloatType()) \
    .add("Days for shipment (scheduled)", IntegerType())

# 5. Processing
orders_df = lines.select(from_json(col("value"), json_schema).alias("data")).select("data.*")
orders_df = orders_df.withColumn("timestamp", current_timestamp())

if geo_df:
    orders_enriched_df = orders_df.join(geo_df, ["Order City", "Order State"], "left")
else:
    orders_enriched_df = orders_df

if loaded_model:
    try:
        orders_final = orders_enriched_df.na.fill(0, subset=["Dest_Lat", "Dest_Lon"])
        predictions_df = loaded_model.transform(orders_final)
    except Exception:
        predictions_df = orders_enriched_df.withColumn("prediction", col("Order Item Quantity") * 0)  # Dummy
else:
    predictions_df = orders_enriched_df.withColumn("prediction", col("Order Item Quantity") * 0)  # Dummy

# Select final columns for storage
final_stream_df = predictions_df.select(
    "timestamp", "Order City", "Order Region",
    "Order Item Total", "prediction"
)


# --- 6. DEFINE WRITING LOGIC (ForeachBatch) ---

def write_to_databases(batch_df, batch_id):
    """
    This function runs on the Driver for every batch.
    We collect the data to the driver and use standard Python libs to write.
    For big data, we would use JDBC/Mongo Spark Connector, but this is safer for your setup.
    """
    # Convert to Pandas for easy insertion
    pdf = batch_df.toPandas()

    if pdf.empty:
        return

    print(f"Batch {batch_id}: Writing {len(pdf)} rows to DBs...")

    # --- A. Write to PostgreSQL (Transactional) ---
    try:
        import psycopg2
        # Connect to the 'postgres' service defined in compose.yaml
        conn = psycopg2.connect(
            host="postgres",
            database="logistic_db",
            user="user",
            password="password"
        )
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                city VARCHAR(255),
                region VARCHAR(255),
                amount FLOAT,
                prediction FLOAT
            );
        """)

        # Insert rows
        for index, row in pdf.iterrows():
            cursor.execute(
                "INSERT INTO order_predictions (timestamp, city, region, amount, prediction) VALUES (%s, %s, %s, %s, %s)",
                (row['timestamp'], row['Order City'], row['Order Region'], row['Order Item Total'], row['prediction'])
            )

        conn.commit()
        cursor.close()
        conn.close()
        print("  -> Written to PostgreSQL.")
    except Exception as e:
        print(f"  -> Error writing to Postgres: {e}")

    # --- B. Write to MongoDB (Aggregated/Document) ---
    try:
        import pymongo
        # Connect to the 'mongo' service
        client = pymongo.MongoClient("mongodb://mongo:27017/")
        db = client["logistic_db"]
        collection = db["realtime_orders"]

        # Convert DataFrame to list of dicts
        records = pdf.to_dict("records")

        # Insert
        if records:
            collection.insert_many(records)
            print("  -> Written to MongoDB.")

        client.close()
    except Exception as e:
        print(f"  -> Error writing to Mongo: {e}")


# --- 7. START STREAMING ---
query = final_stream_df.writeStream \
    .outputMode("append") \
    .foreachBatch(write_to_databases) \
    .start()

query.awaitTermination()