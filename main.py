import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing, model_selection
import xgboost
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def encode_columns(df, columns):
    df[columns] = df[columns].astype(str).fillna("NONE")

    # encode each column
    for col in columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    return df

def encode_columns(df, feature_columns, target_column):
    df[feature_columns] = df[feature_columns].astype(str).fillna("NONE")
    df[target_column] = df[target_column].astype(str).fillna("OTHER")

    X = None
    # encode each feature column
    for col in feature_columns:
        lbl = CountVectorizer(
            tokenizer=word_tokenize,
            token_pattern=None
        )
        lbl.fit(df[col])
        X = lbl.transform(df[col])

    # encode the target column
    lbl = preprocessing.LabelEncoder()
    lbl.fit(df[target_column])
    #df.loc[:, target_column] = lbl.transform(df[target_column])
    y = lbl.transform(df[target_column])
    return X, y

df = pd.read_csv("data/training_set.csv")

X, y = encode_columns(df, ["Description"], "Category")

print("X")
print(X)

print("Y")
print(y)
model = xgboost.XGBClassifier()

#print("Training model")

#scores = model_selection.cross_val_score(model, X, y, cv=10, scoring='f1_macro')

#print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))