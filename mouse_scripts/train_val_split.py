import pandas as pd

seed = 42
train_size = 0.85

# Load CSV; adjust the file path if necessary.
df = pd.read_csv("./clips.csv")

# Shuffle the DataFrame for randomness.
df = df.sample(frac=1, random_state=seed)

# Calculate 80% split index.
split_index = int(len(df) * train_size)
train_df = df.iloc[:split_index]
val_df = df.iloc[split_index:]

# Save train and val splits without header and index, with space-separated values.
train_df[['clip_file', 'label']].to_csv("train.csv", header=False, index=False, sep=' ')
val_df[['clip_file', 'label']].to_csv("val.csv", header=False, index=False, sep=' ')
