import streamlit as st
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_outliers(df: DataFrame):
    """
    Affiche des boxplots pour visualiser les outliers dans les colonnes numériques.
    """
    numeric_cols = ["Days for shipment (scheduled)", "Latitude", "Longitude",
        "Order Item Quantity", "Order Item Total", "Order Profit Per Order",
        "Order Item Discount Rate","Dest_Lat","Dest_Lon"]
    st.subheader("1. Visualisation des Outliers (Boxplots)")

    # Échantillonnage pour la visualisation (car matplotlib ne gère pas le Big Data)
    # On prend 10% des données ou max 20 000 lignes
    sample_df = df.select(numeric_cols).sample(withReplacement=False, fraction=0.1, seed=42).toPandas()

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.boxplot(x=sample_df[col], ax=ax)
        ax.set_title(f"Distribution de : {col}")
        st.pyplot(fig)


def remove_nulls(df: DataFrame):
    """
    Supprime toute ligne contenant au moins une valeur nulle ou NaN.
    """
    st.subheader("2. Suppression des Valeurs Nulles")

    original_count = df.count()

    # dropna() supprime les lignes avec des nulls/NaN
    cleaned_df = df.dropna()

    new_count = cleaned_df.count()
    deleted_count = original_count - new_count

    st.write(f"Lignes avant nettoyage : **{original_count}**")
    st.write(f"Lignes après nettoyage : **{new_count}**")
    st.warning(f"Lignes supprimées (contenaient des nulls) : **{deleted_count}**")

    return cleaned_df


def winsorize_column(df: DataFrame, col_name: str, lower_percentile=0.01, upper_percentile=0.99):
    """
    Applique la Winsorisation sur une colonne.
    Remplace les valeurs < 1er percentile par la valeur du 1er percentile.
    Remplace les valeurs > 99ème percentile par la valeur du 99ème percentile.
    """
    # Calculer les seuils (quantiles)
    quantiles = df.stat.approxQuantile(col_name, [lower_percentile, upper_percentile], 0.001)
    lower_bound = quantiles[0]
    upper_bound = quantiles[1]

    # Appliquer la logique de remplacement avec F.when
    return df.withColumn(col_name,
                         F.when(F.col(col_name) < lower_bound, lower_bound)
                         .when(F.col(col_name) > upper_bound, upper_bound)
                         .otherwise(F.col(col_name))
                         )


def apply_winsorization(df: DataFrame, cols_to_winsorize: list):
    """
    Applique la winsorisation sur une liste de colonnes.
    """
    st.subheader("3. Winsorisation des Outliers Extrêmes")
    st.info("Les valeurs inférieures au 1% et supérieures au 99% sont plafonnées.")

    winsorized_df = df
    for col in cols_to_winsorize:
        winsorized_df = winsorize_column(winsorized_df, col)
        st.write(f"- Colonne `{col}` traitée.")

    return winsorized_df