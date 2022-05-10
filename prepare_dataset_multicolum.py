import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import model_selection

try:
    print("Trying to read data from csv file...")
    texts = pd.read_csv("data/CNN_Articels_clean.csv")
except Exception as e:
    print("Data not found. Downloading data from kaggle...")
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('hadasu92/cnn-articles-after-basic-cleaning', path='data', unzip=True)
    print("Trying to read data from csv file...")
    texts = pd.read_csv("data/CNN_Articels_clean/CNN_Articels_clean.csv")

print("Data read successfully. Splitting data into train and validation sets...")

X = texts[["Description","Headline"]]
y = texts["Category"]

X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# validation set
df = pd.DataFrame({"Description": X_validation, "Category": y_validation})
df.to_csv("data/validation_set.csv")

# training set
df = pd.DataFrame({"Description": X_train, "Category": y_train})
df.to_csv("data/training_set.csv")

print("All done!")