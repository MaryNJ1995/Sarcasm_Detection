import pandas as pd

df = pd.read_csv("train.csv")[42133:]
df.to_csv("train_split_data.csv", index=False)
