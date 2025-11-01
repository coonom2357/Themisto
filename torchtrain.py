import numpy as np
import pickle

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB, ComplementNB
import pandas as pd
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv("./Data/Parsed/parsed_data.csv")
df = df[["description", "severity"]].dropna()

print(df.head())
print(df['severity'].value_counts())
print(f"Number of samples: {len(df)}")

le = LabelEncoder()
y = le.fit_transform(df['severity'])

print(list(zip(le.classes_, le.transform(le.classes_))))

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=20000,
    ngram_range=(1, 3),
    min_df=3,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["description"]).toarray()  # dense for PyTorch

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_data = TensorDataset(X_train_t, y_train_t)
test_data = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

