import streamlit as st
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
)
# --- NOUVEAUX IMPORTS DE MODÈLES ---
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier
)
# -----------------------------------
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


def build_and_train(input_df: DataFrame, model_name: str):
    """
    Prend le DataFrame enrichi, construit le pipeline de prétraitement
    et entraîne le modèle de classification (choisi par l'utilisateur)
    avec CrossValidator.
    """

    # --- 1. Définition Dynamique des Features ---
    # Nous vérifions d'abord si les colonnes de géocodage (longues à obtenir) existent.

    categorical_cols = [
        "Shipping Mode","Order Region","Order State"

    ]

    base_numerical_cols = [
     "Latitude", "Longitude","distance"

    ]

    # Vérifier si le géocodage a été fait et si les colonnes existent
    if "Dest_Lat" in input_df.columns and "Dest_Lon" in input_df.columns:
        st.write("Info: Colonnes de géocodage (Dest_Lat, Dest_Lon) détectées et incluses.")
        numerical_cols = base_numerical_cols
    else:
        st.warning("Avertissement: Colonnes Dest_Lat/Dest_Lon non trouvées. Entraînement sans les données de distance.")
        numerical_cols = base_numerical_cols

    label_col = "Late_delivery_risk"

    # --- 2. Création des Étapes de Prétraitement ---

    stages = []

    # Gérer les valeurs nulles (Ex: pour Dest_Lat/Dest_Lon si le géocodage a échoué)
    numerical_imputed_cols = [c + "_imputed" for c in numerical_cols]
    imputer = Imputer(
        inputCols=numerical_cols,
        outputCols=numerical_imputed_cols,
        strategy="mean"
    )
    stages.append(imputer)

    # Indexer les colonnes catégorielles
    categorical_indexed_cols = [c + "_index" for c in categorical_cols]
    string_indexer = StringIndexer(
        inputCols=categorical_cols,
        outputCols=categorical_indexed_cols,
        handleInvalid="keep"
    )
    stages.append(string_indexer)

    # Encoder en One-Hot les colonnes indexées
    categorical_onehot_cols = [c + "_onehot" for c in categorical_cols]
    one_hot_encoder = OneHotEncoder(
        inputCols=categorical_indexed_cols,
        outputCols=categorical_onehot_cols
    )
    stages.append(one_hot_encoder)

    # Assembler toutes les features
    feature_cols = numerical_imputed_cols + categorical_onehot_cols
    vector_assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_unscaled"
    )
    stages.append(vector_assembler)

    # Mettre à l'échelle (Standardiser) le vecteur de features
    scaler = StandardScaler(
        inputCol="features_unscaled",
        outputCol="features",
        withStd=True,
        withMean=True
    )
    stages.append(scaler)

    # --- 3. SÉLECTION DU MODÈLE ET GRILLE DE PARAMÈTRES ---

    st.info(f"Configuration du pipeline pour : **{model_name}**")

    if model_name == "Logistic Regression":
        model = LogisticRegression(featuresCol="features", labelCol=label_col)
        # Grille pour tester la "régularisation" (évite l'overfitting)
        paramGrid = ParamGridBuilder().addGrid(model.regParam, [0.01]).build()


    elif model_name == "Random Forest":
        model = RandomForestClassifier(featuresCol="features", labelCol=label_col)
        # Grille pour tester différentes tailles de forêt
        # paramGrid = (ParamGridBuilder()
        #              .addGrid(model.numTrees, [50, 100])  # Nombre d'arbres
        #              .addGrid(model.maxDepth, [5, 10])  # Profondeur de l'arbre
        #              .build())
        paramGrid = ParamGridBuilder().addGrid(model.numTrees, [20]).build()
    elif model_name == "Gradient-Boosted Trees (GBT)":
        model = GBTClassifier(featuresCol="features", labelCol=label_col)
        # Grille pour tester différentes complexités
        paramGrid = (ParamGridBuilder()
                     .addGrid(model.maxIter, [20, 50])  # Nombre d'itérations
                     .addGrid(model.maxDepth, [3, 5])  # Profondeur
                     .build())
    else:
        st.error(f"Nom de modèle '{model_name}' non reconnu.")
        return None, None, None

    stages.append(model)

    # --- 4. Création du Pipeline Final ---
    pipeline = Pipeline(stages=stages)

    # --- 5. Préparation des Données pour l'Entraînement ---

    cols_to_keep = numerical_cols + categorical_cols + [label_col]
    data_for_ml = input_df.select(cols_to_keep).na.drop(subset=[label_col])

    # --- OPTIMISATION 1 (Solution 2) ---
    st.write("Optimisation 1: Repartitionnement des données...")
    data_for_ml = data_for_ml.repartition(200)
    data_for_ml.cache()

    st.write(f"Données prêtes : {data_for_ml.count()} lignes.")

    # Diviser les données en ensembles d'entraînement et de test
    (trainingData, testData) = data_for_ml.randomSplit([0.8, 0.2], seed=42)

    # --- 6. Configuration du CrossValidator ---

    evaluator = BinaryClassificationEvaluator(labelCol=label_col)

    # --- OPTIMISATION 2 (Solution 3) ---
    st.write("Optimisation 2: Configuration du CrossValidator (parallélisme=1)...")
    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=2,  # <-- Réduire de 3 à 2 pour économiser des ressources
        parallelism=1
    )

    # --- 7. Entraînement ---
    st.write(f"Démarrage de l'entraînement CrossValidator pour {model_name}. Cela peut être long...")
    cvModel = crossval.fit(trainingData)
    model_folder_name = model_name.replace(' ', '_').replace('-', '_').lower()

    # 2. Crée le chemin complet
    model_save_path = f"/app/spark_models/{model_folder_name}"

    # 3. Sauvegarde le modèle (en tant que dossier)
    cvModel.bestModel.write().overwrite().save(model_save_path)
    st.write("Entraînement terminé.")

    data_for_ml.unpersist()

    return cvModel, testData, evaluator