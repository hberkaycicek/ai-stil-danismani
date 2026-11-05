import pandas as pd
df = pd.read_feather(r"C:\...\feathers\basv_data01_10_204_vs.feather")
print(df.shape, df["DONEM"].min(), df["DONEM"].max())
