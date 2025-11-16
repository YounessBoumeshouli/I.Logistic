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
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
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
        return spark.read.csv(file_path, header=True,encoding="ISO-8859-1" ,inferSchema=True, sep=',')
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
        'Order Country',
        'Order City',
        'Order State',
        'Order Profit Per Order',
        'Order Item Discount Rate'
    ]
    new_df = df.select(cols)
    st.dataframe(new_df.limit(5).toPandas())
    st.dataframe(new_df.select("Late_delivery_risk").distinct().toPandas())
    st.dataframe(new_df.select("Category Name").distinct().toPandas())
    st.dataframe(new_df.select("Department Name").distinct().toPandas())
    st.dataframe(new_df.select("Customer Segment").distinct().toPandas())
    st.dataframe(new_df.select("Shipping Mode").distinct().toPandas())
    distinct_rounded_pd = new_df.select(
        F.round(F.col("Latitude"), 2).alias("Latitude"),
        F.round(F.col("Longitude"), 2).alias("Longitude")
    ).distinct().toPandas()
    st.dataframe(distinct_rounded_pd)

    st.dataframe(new_df.select("Order City" , "Order State").distinct().toPandas())


    st.dataframe(new_df.distinct().toPandas())
    st.text('duplicated rows :')
    st.text(len(new_df.distinct().toPandas()) - len(new_df.toPandas()))
    st.write("Schéma des colonnes :")
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
                F.col(c).isNull() | F.isnan(F.col(c)), 1
            )
        ).alias(c)
        for c in new_df.columns
    ])
    st.text(missing_counts)

    numeric_cols = [
        'Order Item Quantity',
        'Order Item Total',
        'Order Profit Per Order',
        'Order Item Discount Rate'
    ]
    st.text("select min order profit per order")
    st.dataframe(new_df.select(F.min("Order Profit Per Order")))
    # Demandez à Spark de calculer les statistiques
    st.write("Résumé statistique des colonnes numériques :")
    st.dataframe(new_df.select(numeric_cols).summary().toPandas())
    assembler = VectorAssembler(
        inputCols=numeric_cols,
        outputCol="features",
        handleInvalid="skip"  # Saute les lignes avec des nuls
    )

    # 3. Transformer le DataFrame
    df_vector = assembler.transform(new_df.na.drop(subset=numeric_cols)).select("features")

    # 4. Calculer la matrice de corrélation (Pearson)
    correlation_matrix = Correlation.corr(df_vector, "features").head()

    # 5. Extraire et afficher la matrice (un peu complexe, mais voici)
    matrix = correlation_matrix[0].toArray()
    corr_df = pd.DataFrame(matrix, columns=numeric_cols, index=numeric_cols)

    st.write("Matrice de Corrélation Numérique :")
    st.dataframe(corr_df)
    st.write("Taux de retard moyen par 'Shipping Mode':")
    st.dataframe(
        new_df.groupBy("Shipping Mode")
        .avg("Late_delivery_risk")  # Calcule le taux de retard moyen
        .orderBy(F.desc("avg(Late_delivery_risk)"))
        .toPandas()
    )
    st.subheader("VRAI Taux de Retard par Shipping Mode")

    # Calculez la MOYENNE (le Taux)
    rate_by_segment_spark = (new_df
                             .groupBy("Shipping Mode")
                             .avg("Late_delivery_risk")
                             .withColumnRenamed("avg(Late_delivery_risk)", "Taux de Retard")
                             .orderBy(F.desc("Taux de Retard"))
                             )

    rate_by_segment_pd = rate_by_segment_spark.toPandas()

    st.dataframe(rate_by_segment_pd)
    st.bar_chart(rate_by_segment_pd.set_index("Shipping Mode"))
    # st.text("this feature not effecting the data so i will delete it")
    # new_df = new_df.drop('Customer Segment')
    st.write("Taux de retard moyen par 'Category Name':")
    st.dataframe(
        new_df.groupBy("Category Name")
        .avg("Late_delivery_risk")
        .orderBy(F.desc("avg(Late_delivery_risk)"))
        .toPandas()
    )

    # REGARDEZ:
    # Si "Standard Class" a un "avg(Late_delivery_risk)" de 0.6 (60%)
    # et "Same Day" a 0.05 (5%), vous savez que 'Shipping Mode' est
    # une feature EXTRÊMEMENT prédictive.
    df_with_day = new_df.withColumn(
        "jour_semaine",
        F.dayofweek(F.to_timestamp("order date (DateOrders)"))
    )

    st.write("Taux de retard moyen par Jour de la Semaine :")
    st.dataframe(
        df_with_day.groupBy("jour_semaine")
        .avg("Late_delivery_risk")
        .orderBy("jour_semaine")
        .toPandas()
    )
    st.header("Analyse de Distribution (EDA)")

    st.subheader("Distribution des Modes de Livraison (Shipping Mode)")

    shipping_dist_spark = (new_df
                           .groupBy("Shipping Mode")
                           .count()
                           .orderBy(F.desc("count"))
                           )
    shipping_dist_pd = shipping_dist_spark.toPandas()
    st.dataframe(shipping_dist_pd)
    st.bar_chart(shipping_dist_pd.set_index("Shipping Mode"))
    st.subheader("VRAI Taux de Retard par Order profit per Order")
    df_with_profit_cluster = new_df.withColumn("Tranche de Profit",
                                               F.when(F.col("Order Profit Per Order") < -100, "Perte Extrême (< -100)")
                                               .when(F.col("Order Profit Per Order") < 0, "Perte Normale (-100 à 0)")
                                               .when(F.col("Order Profit Per Order") == 0, "Nul (0)")
                                               .when(F.col("Order Profit Per Order") < 50, "Profit Faible (0 à 50)")
                                               .when(F.col("Order Profit Per Order") < 150, "Profit Moyen (50 à 150)")
                                               .otherwise("Profit Élevé (> 150)")
                                               )

    rate_by_profit_spark = (df_with_profit_cluster
                            .groupBy("Tranche de Profit")
                            .avg("Late_delivery_risk")
                            .withColumnRenamed("avg(Late_delivery_risk)", "Taux de Retard")
                            .orderBy(F.desc("Taux de Retard"))
                            )
    shipping_dist_pd = rate_by_profit_spark.toPandas()
    st.dataframe(shipping_dist_pd)
    st.bar_chart(shipping_dist_pd.set_index("Tranche de Profit"))

    st.subheader("Distribution de Order profit per Order")

    order_item_discount_rate = (new_df
                                .groupBy("Order profit per Order")
                                .count()
                                .orderBy(F.desc("count"))
                                )
    order_item_discount_rate_pd = order_item_discount_rate.toPandas()
    st.dataframe(order_item_discount_rate_pd)
    st.bar_chart(order_item_discount_rate_pd.set_index("Order profit per Order"))


    st.subheader("VRAI Taux de Retard par Order profit per Order")
    df_with_profit_cluster = new_df.withColumn("Tranche de Profit",
                                               F.when(F.col("Order Profit Per Order") < -100, "Perte Extrême (< -100)")
                                               .when(F.col("Order Profit Per Order") < 0, "Perte Normale (-100 à 0)")
                                               .when(F.col("Order Profit Per Order") == 0, "Nul (0)")
                                               .when(F.col("Order Profit Per Order") < 50, "Profit Faible (0 à 50)")
                                               .when(F.col("Order Profit Per Order") < 150, "Profit Moyen (50 à 150)")
                                               .otherwise("Profit Élevé (> 150)")
                                               )

    rate_by_profit_spark = (df_with_profit_cluster
                            .groupBy("Tranche de Profit")
                            .avg("Late_delivery_risk")
                            .withColumnRenamed("avg(Late_delivery_risk)", "Taux de Retard")
                            .orderBy(F.desc("Taux de Retard"))
                            )
    shipping_dist_pd = rate_by_profit_spark.toPandas()
    st.dataframe(shipping_dist_pd)
    st.bar_chart(shipping_dist_pd.set_index("Tranche de Profit"))

    st.subheader("Distribution de Order Item Discount Rate")
    st.text("this feature not effecting the data so i will delete it")
    new_df = new_df.drop('Order Profit Per Order')
    order_item_discount_rate = (new_df
                           .groupBy("Order Item Discount Rate")
                           .count()
                           .orderBy(F.desc("count"))
                           )
    order_item_discount_rate_pd = order_item_discount_rate.toPandas()
    st.dataframe(order_item_discount_rate_pd)
    st.bar_chart(order_item_discount_rate_pd.set_index("Order Item Discount Rate"))
    st.text('we seeing that the distribution si equilbre this features not effecting the data')
    st.dataframe(new_df.toPandas())

    st.subheader("VRAI Taux de Retard par Order Item Discount Rate")

    # Calculez la MOYENNE (le Taux)
    rate_by_segment_spark = (new_df
                             .groupBy("Order Item Discount Rate")
                             .avg("Late_delivery_risk")
                             .withColumnRenamed("avg(Late_delivery_risk)", "Taux de Retard")
                             .orderBy(F.desc("Taux de Retard"))
                             )

    rate_by_segment_pd = rate_by_segment_spark.toPandas()

    st.dataframe(rate_by_segment_pd)
    new_df = new_df.drop('Order Item Discount Rate')

    st.bar_chart(rate_by_segment_pd.set_index("Order Item Discount Rate"))

    st.dataframe(new_df.toPandas())

    st.subheader("Distribution de retard avec le type de client")
    Late_delivery_risk_by_client_rate = (new_df.select("*").where(F.col("Late_delivery_risk") == 1)
                                .groupBy("Late_delivery_risk","Customer Segment")
                                .count()
                                .orderBy(F.desc("count"))
                                )
    Late_delivery_risk_by_client_rate_pd = Late_delivery_risk_by_client_rate.toPandas()
    st.dataframe(Late_delivery_risk_by_client_rate_pd)
    st.bar_chart(Late_delivery_risk_by_client_rate_pd.set_index("Customer Segment"))
    st.subheader("VRAI Taux de Retard par Segment Client")

    # Calculez la MOYENNE (le Taux)
    rate_by_segment_spark = (new_df
                             .groupBy("Customer Segment")
                             .avg("Late_delivery_risk")
                             .withColumnRenamed("avg(Late_delivery_risk)", "Taux de Retard")
                             .orderBy(F.desc("Taux de Retard"))
                             )

    rate_by_segment_pd = rate_by_segment_spark.toPandas()

    st.dataframe(rate_by_segment_pd)
    st.bar_chart(rate_by_segment_pd.set_index("Customer Segment"))
    st.text("this feature not effecting the data so i will delete it")
    new_df = new_df.drop('Customer Segment')
    st.subheader("Distribution de retard avec le type de livraison")
    Late_delivery_risk_by_shipping_mode_rate = (new_df.select("*").where(F.col("Late_delivery_risk") == 1)
                                .groupBy("Late_delivery_risk","Shipping Mode")
                                .count()
                                .orderBy(F.desc("count"))
                                )
    Late_delivery_risk_by_shipping_mode_pd = Late_delivery_risk_by_shipping_mode_rate.toPandas()
    st.dataframe(Late_delivery_risk_by_shipping_mode_pd)
    st.bar_chart(Late_delivery_risk_by_shipping_mode_pd.set_index("Shipping Mode"))
    st.subheader("VRAI Taux de Retard par Mode de Livraison")

    # Calculez la MOYENNE (le Taux)
    rate_by_shipping_spark = (new_df
                              .groupBy("Shipping Mode")
                              .avg("Late_delivery_risk")
                              .withColumnRenamed("avg(Late_delivery_risk)", "Taux de Retard")
                              .orderBy(F.desc("Taux de Retard"))
                              )

    rate_by_shipping_pd = rate_by_shipping_spark.toPandas()

    st.dataframe(rate_by_shipping_pd)
    st.bar_chart(rate_by_shipping_pd.set_index("Shipping Mode"))
    st.subheader("VRAI Taux de Retard par Department Name")

    rate_by_shipping_spark = (new_df
                              .groupBy("Department Name")
                              .avg("Late_delivery_risk")
                              .withColumnRenamed("avg(Late_delivery_risk)", "Taux de Retard")
                              .orderBy(F.desc("Taux de Retard"))
                              )

    rate_by_shipping_pd = rate_by_shipping_spark.toPandas()

    st.dataframe(rate_by_shipping_pd)
    st.bar_chart(rate_by_shipping_pd.set_index("Department Name"))

    st.text("again this feature not effecting the data so i will drop it")
    new_df = new_df.drop("Department Name")

    st.subheader("VRAI Taux de Retard par Order Item Total")

    df_with_profit_cluster = new_df.withColumn("Tranche de Profit",
                                               F.when(F.col("Order Item Total") == 0, "Total Nul (0)")
                                               .when(F.col("Order Item Total") < 50, "Total Faible (0 à 50)")
                                               .when(F.col("Order Item Total") < 100, "Total Moyen (50 à 100)")
                                               .when(F.col("Order Item Total") < 150, "Total Moyen (100 à 150)")
                                               .when(F.col("Order Item Total") < 200, "Total Moyen (150 à 200)")
                                               .otherwise("Profit Élevé (> 200)")
                                               )

    rate_by_profit_spark = (df_with_profit_cluster
                            .groupBy("Tranche de Profit")
                            .avg("Late_delivery_risk")
                            .withColumnRenamed("avg(Late_delivery_risk)", "Taux de Retard")
                            .orderBy(F.desc("Taux de Retard"))
                            )
    shipping_dist_pd = rate_by_profit_spark.toPandas()
    st.dataframe(shipping_dist_pd)
    st.bar_chart(shipping_dist_pd.set_index("Tranche de Profit"))
    st.subheader("VRAI Taux de Retard par Order Item Quantity")

    df_with_profit_cluster = new_df.withColumn("Tranche de Quantity",
                                               F.when(F.col("Order Item Quantity") == 1, "Total 1")
                                               .when(F.col("Order Item Quantity") == 2, "Total 2")
                                               .when(F.col("Order Item Quantity") == 3, "Total 3")
                                               .when(F.col("Order Item Quantity")  == 4, "Total 4")
                                               .when(F.col("Order Item Quantity")  ==5, "Total 5")
                                               .otherwise("Profit Élevé (> 5)")
                                               )

    rate_by_profit_spark = (df_with_profit_cluster
                            .groupBy("Tranche de Quantity")
                            .avg("Late_delivery_risk")
                            .withColumnRenamed("avg(Late_delivery_risk)", "Taux de Retard")
                            .orderBy(F.desc("Taux de Retard"))
                            )
    shipping_dist_pd = rate_by_profit_spark.toPandas()
    st.dataframe(shipping_dist_pd)
    st.bar_chart(shipping_dist_pd.set_index("Tranche de Quantity"))
    st.subheader("VRAI Taux de Retard par Order Country")

    rate_by_shipping_spark = (new_df
                              .groupBy("Order Country")
                              .avg("Late_delivery_risk")
                              .withColumnRenamed("avg(Late_delivery_risk)", "Taux de Retard")
                              .orderBy(F.desc("Taux de Retard"))
                              )

    rate_by_shipping_pd = rate_by_shipping_spark.toPandas()

    st.dataframe(rate_by_shipping_pd)
    st.bar_chart(rate_by_shipping_pd.set_index("Order Country"))
    st.subheader("VRAI Taux de Retard par Order City")

    rate_by_shipping_spark = (new_df
                              .groupBy("Order City")
                              .avg("Late_delivery_risk")
                              .withColumnRenamed("avg(Late_delivery_risk)", "Taux de Retard")
                              .orderBy(F.desc("Taux de Retard"))
                              )

    rate_by_shipping_pd = rate_by_shipping_spark.toPandas()

    st.dataframe(rate_by_shipping_pd)
    st.bar_chart(rate_by_shipping_pd.set_index("Order City"))
    st.subheader("VRAI Taux de Retard par Order State")

    rate_by_shipping_spark = (new_df
                              .groupBy("Order State")
                              .avg("Late_delivery_risk")
                              .withColumnRenamed("avg(Late_delivery_risk)", "Taux de Retard")
                              .orderBy(F.desc("Taux de Retard"))
                              )

    rate_by_shipping_pd = rate_by_shipping_spark.toPandas()

    st.dataframe(rate_by_shipping_pd)
    st.bar_chart(rate_by_shipping_pd.set_index("Order State"))

    st.dataframe(new_df.toPandas())
    st.header('encodage des variables categorielles')
    new_df =new_df.drop("order date (DateOrders)")
    cols = [
        "Shipping Mode",
        "Category Name",
        "Order City",
        "Order Country",
        "Order State",
        "Order Region",
        "Latitude",
        "Longitude",
        "Order Item Quantity",
        "Order Item Total",
        "Late_delivery_risk"
    ]
    new_df = new_df.select(cols)
    categorical_cols = ["Shipping Mode", "Category Name","Order City","Order Country","Order State","Order Region"]
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_Index", handleInvalid="keep") for c in categorical_cols]
    ohe = OneHotEncoder(inputCols=[f"{c}_Index" for c in categorical_cols],
                        outputCols=[f"{c}_Vec" for c in categorical_cols])
    label_col = "Late_delivery_risk"

    numeric_cols = [
        'Latitude',
        'Longitude',
        'Order Item Quantity',
        'Order Item Total',
    ]
    imputer_cols = numeric_cols
    imputer_output_cols = [f"{c}_imputed" for c in imputer_cols]
    imputer = Imputer(
        inputCols=imputer_cols,
        outputCols=imputer_output_cols
    ).setStrategy("mean")
    temp_pipeline = Pipeline(stages=[imputer] + indexers)
    df_preprocessed_model = temp_pipeline.fit(new_df)
    total_count = new_df.count()
    count_class_1 = new_df.filter(F.col(label_col) == 1).count()
    count_class_0 = total_count - count_class_1

    # Poids = Total / (2 * Nb_Classe)
    weight_class_0 = total_count / (2.0 * count_class_0)
    weight_class_1 = total_count / (2.0 * count_class_1)

    df_weighted = new_df.withColumn("weight",
                                        F.when(F.col(label_col) == 1, weight_class_1).otherwise(weight_class_0)
                                        )
    st.write(f"Gestion du déséquilibre : Poids Classe 0 = {weight_class_0:.2f}, Poids Classe 1 = {weight_class_1:.2f}")

    # 6.2. Assemblage des 'features'
    # Colonnes en entrée de l'assembleur:
    imputed_cols = [f"{c}_imputed" for c in numeric_cols]
    ohe_cols = [f"{c}_Vec" for c in categorical_cols]

    assembler_inputs = imputed_cols  + ohe_cols

    vector_assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features_unscaled"
    )

    scaler = StandardScaler(
        inputCol="features_unscaled",
        outputCol="features",
        withStd=True,
        withMean=True
    )
    st.write("Étapes d'assemblage (VectorAssembler) et de normalisation (StandardScaler) définies.")

    (training_data, test_data) = df_weighted.randomSplit([0.8, 0.2], seed=42)

    training_data.cache()
    test_data.cache()
    st.write(
        f"Données séparées : {training_data.count()} en entraînement, {test_data.count()} en test (et mises en cache).")

    # 6.5. Choix du modèle
    lr = LogisticRegression(featuresCol="features", labelCol=label_col, weightCol="weight")

    # 6.6. Construction du Pipeline ML complet
    # (Imputer -> Indexers -> OHE -> Assembler -> Scaler -> Modèle)
    ml_pipeline = Pipeline(stages=[imputer] + indexers + [ohe, vector_assembler, scaler, lr])
    st.success("✅ Pipeline ML complet (preprocessing + modèle) construit.")
    st.header("Étape 7 : Entraînement et Validation Croisée")

    # Modèle global pour qu'il soit accessible par l'étape 8 et 9
    best_pipeline_model = None

    if df:
        # Construction de la grille de paramètres
        paramGrid = (ParamGridBuilder()
                     .addGrid(lr.regParam, [0.1, 0.01])  # Paramètre de régularisation
                     .addGrid(lr.elasticNetParam, [0.0, 0.5])  # 0.0 = L2, 1.0 = L1
                     .build())

        # Évaluateur (AUC-ROC)
        evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction",
                                                  metricName="areaUnderROC")

        # Cross-Validator
        crossval = CrossValidator(
            estimator=ml_pipeline,  # Le pipeline complet est l'estimateur
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=3  # 3 folds pour la rapidité de la démo
        )

        st.write("Démarrage de la validation croisée (CrossValidator)... (Cette étape peut être longue)")
        start_time = time.time()

        try:
            # Entraînement du CrossValidator
            cv_model = crossval.fit(training_data)

            end_time = time.time()
            st.write(f"Entraînement terminé en {end_time - start_time:.2f} secondes.")

            # Le 'bestModel' est le pipeline complet optimisé
            best_pipeline_model = cv_model.bestModel

            # Affichage des meilleurs hyperparamètres
            best_lr_model = best_pipeline_model.stages[-1]  # Le modèle LR est la dernière étape
            st.success("Meilleur modèle (pipeline) trouvé :")
            st.write(f" - Meilleur RegParam : {best_lr_model.getRegParam()}")
            st.write(f" - Meilleur ElasticNetParam : {best_lr_model.getElasticNetParam()}")

        except Exception as e:
            st.error(f"Erreur lors de l'entraînement CrossValidator : {e}")
            best_pipeline_model = None  # Assure que la suite ne s'exécute pas

    # ------------------------------------------------------------------
    # Étape 8 : Évaluation du Modèle
    # ------------------------------------------------------------------
    st.header("Étape 8 : Évaluation du Modèle")

    if best_pipeline_model:
        st.write("Évaluation des performances sur l'ensemble de test...")
        predictions = best_pipeline_model.transform(test_data)

        # 1. AUC-ROC (via l'évaluateur)
        auc = evaluator.evaluate(predictions)
        st.metric(label="**Aire sous la courbe (AUC-ROC)**", value=f"{auc:.4f}")

        # 2. Autres métriques (Accuracy, Precision, Recall, F1)
        preds_and_labels = predictions.select("prediction", label_col).withColumn("label",
                                                                                  F.col(label_col).cast(
                                                                                      "double")).select(
            "prediction", "label")

        # Conversion pour MulticlassMetrics
        metrics_rdd = preds_and_labels.rdd.map(lambda row: (float(row.prediction), row.label))
        multi_metrics = MulticlassMetrics(metrics_rdd)

        accuracy = multi_metrics.accuracy
        precision = multi_metrics.precision(1.0)  # Précision pour la classe '1' (Exited)
        recall = multi_metrics.recall(1.0)  # Rappel pour la classe '1' (Exited)
        f1_score = multi_metrics.fMeasure(1.0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="**Accuracy**", value=f"{accuracy:.4f}")
        col2.metric(label="**Precision (Classe 1)**", value=f"{precision:.4f}")
        col3.metric(label="**Recall (Classe 1)**", value=f"{recall:.4f}")
        col4.metric(label="**F1-Score (Classe 1)**", value=f"{f1_score:.4f}")

        # 3. Matrice de Confusion (via scikit-learn sur les résultats Pandas)
        st.subheader("Matrice de confusion")
        try:
            y_true_pd = predictions.select(label_col).toPandas()
            y_pred_pd = predictions.select("prediction").toPandas()

            cm = confusion_matrix(y_true_pd[label_col], y_pred_pd["prediction"])

            # Affichage de la matrice
            cm_df = pd.DataFrame(cm,
                                 index=[f"Vrai_{i}" for i in [0, 1]],
                                 columns=[f"Prédit_{i}" for i in [0, 1]])
            st.dataframe(cm_df)

            st.write(f"Vrais Négatifs (VN) : {cm[0][0]} (Client 'Fidèle' prédit 'Fidèle')")
            st.write(f"Faux Positifs (FP) : {cm[0][1]} (Client 'Fidèle' prédit 'Attrition')")
            st.write(f"Faux Négatifs (FN) : {cm[1][0]} (Client 'Attrition' prédit 'Fidèle') - **Erreur critique !**")
            st.write(f"Vrais Positifs (VP) : {cm[1][1]} (Client 'Attrition' prédit 'Attrition')")

        except Exception as e:
            st.error(f"Erreur lors du calcul de la matrice de confusion : {e}")

        # Dé-cache des données
        training_data.unpersist()
        test_data.unpersist()
        st.write("Données d'entraînement et de test libérées du cache.")

    # ------------------------------------------------------------------
    # Étape 9 : Sauvegarde et Déploiement
    # ------------------------------------------------------------------
    st.header("Étape 9 : Sauvegarde et Déploiement")

    if best_pipeline_model:
        # 1. Sauvegarde du modèle optimisé
        model_path = "/tmp/spark-ml-model-attrition"  # Path dans le conteneur Docker
        try:
            best_pipeline_model.write().overwrite().save(model_path)
            st.success(f"✅ Modèle (Pipeline) sauvegardé dans : {model_path}")
        except Exception as e:
            st.error(f"Erreur lors de la sauvegarde du modèle : {e}")

        # 2. Interface de prédiction en temps réel avec Streamlit
        st.subheader("Interface de Prédiction en Temps Réel")
        st.write("Entrez les informations d'un client pour prédire son risque d'attrition :")

        # Création des inputs utilisateur
        col1, col2, col3 = st.columns(3)

        with col1:
            geo = st.selectbox("Géographie", ("France", "Spain", "Germany"), key="pred_geo")
            gender = st.selectbox("Genre", ("Male", "Female"), key="pred_gender")
            age = st.slider("Âge", 18, 100, 35, key="pred_age")

        with col2:
            credit_score = st.slider("Score de Crédit", 300, 850, 600, key="pred_score")
            tenure = st.slider("Ancienneté (années)", 0, 10, 5, key="pred_tenure")
            balance = st.number_input("Solde (Balance)", 0.0, 250000.0, 50000.0, key="pred_balance")

        with col3:
            num_products = st.selectbox("Nb. de Produits", (1, 2, 3, 4), key="pred_prod")
            has_cr_card = st.selectbox("Possède une Carte de Crédit ?", (1, 0),
                                       format_func=lambda x: "Oui" if x == 1 else "Non", key="pred_card")
            is_active_member = st.selectbox("Membre Actif ?", (1, 0), format_func=lambda x: "Oui" if x == 1 else "Non",
                                            key="pred_active")
            estimated_salary = st.number_input("Salaire Estimé", 0.0, 200000.0, 75000.0, key="pred_salary")

        if st.button("Lancer la Prédiction"):
            # Créer un DataFrame Spark à partir des inputs
            # Doit avoir EXACTEMENT le même schéma que les données d'entraînement (df_cleaned)
            schema = new_df.schema

            input_data = [(
                credit_score, geo, gender, age, tenure, balance,
                num_products, has_cr_card, is_active_member, estimated_salary,
                0  # 'Exited' (valeur fictive, non utilisée pour la prédiction)
            )]

            try:
                # Créer le DataFrame
                predict_df = spark.createDataFrame(input_data, schema)

                # Appliquer le pipeline entraîné
                prediction_result_df = best_pipeline_model.transform(predict_df)

                # Obtenir le résultat
                result = prediction_result_df.select("prediction", "probability").first()
                prediction = result["prediction"]
                probability_of_exit = result["probability"][1]  # Probabilité d'être classe '1' (Exited)

                st.write("---")
                if prediction == 1.0:
                    st.error(f"### Résultat : RISQUE ÉLEVÉ D'ATTRITION")
                    st.metric("Probabilité d'attrition", f"{probability_of_exit:.2%}")
                else:
                    st.success(f"### Résultat : Faible risque d'attrition")
                    st.metric("Probabilité d'attrition", f"{probability_of_exit:.2%}")

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                st.error("Assurez-vous que le modèle a été entraîné (rafraîchissez la page si nécessaire).")
    else:
        st.warning(
            "Le DataFrame n'a pas pu être chargé ou le modèle n'a pas été entraîné. L'évaluation et le déploiement sont désactivés.")
