import pandas as pd
import numpy as np

from sklearn.utils import shuffle

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, LabelEncoder
from sklearn.impute import SimpleImputer as Imputer # if we also look at numeric features
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

from utils import combine_text_columns, SparseInteractions

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

df = pd.read_csv("data/training_set.csv")

SCORING_METRIC = "roc_auc_ovr"

FEATURES = ["Description"]
LABEL = "Category"
NON_LABELS = [c for c in df.columns if c != LABEL]

# Select k best features in text vectors
chi_k = 300

TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

#dummy_labels = pd.get_dummies(df[LABEL])
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(df[LABEL])

X_train, X_test, y_train, y_test = train_test_split(df[NON_LABELS], encoded_labels, random_state=42)

get_text_data = FunctionTransformer(lambda x: combine_text_columns(x, to_drop=[LABEL]), validate=False)


pl = Pipeline([
        ('feature_preprocessing ', FeatureUnion(
            transformer_list = [
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),
                    #('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                    #                                 alternate_sign=False, norm=None, binary=False,
                    #                                 ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, k=chi_k))
                ]))
             ]
        )),
        #('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression(), n_jobs=-1))
    ])

cv = cross_val_score(pl, X_train, y_train, cv=5, scoring=SCORING_METRIC)

print(f"\nMean cross validated {SCORING_METRIC} score: {np.mean(cv)}\n")

for fold, score in enumerate(cv):
    print(f"Score on fold {fold}: {score}")
