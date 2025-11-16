import pandas as pd

df = pd.read_csv("datasets/DataCoSupplyChainDataset.csv")
lag_long = pd.read_csv("geocoded_cities.csv")
new_df = df.concat()