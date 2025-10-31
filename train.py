import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB, ComplementNB
import pandas as pd

df = pd.read_csv("./Data/Parsed/parsed_data.csv")
df = df[["description", "severity"]].dropna()

print(df.head())
print(df['severity'].value_counts())
print(f"Number of samples: {len(df)}")

le = LabelEncoder()
y = le.fit_transform(df['severity'])

print(list(zip(le.classes_, le.transform(le.classes_))))

X_train, X_test, y_train, y_test = train_test_split(
    df['description'], y, test_size=0.2, random_state=1, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=50000,
    ngram_range=(1, 5),
    min_df=3,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Number of features: {len(vectorizer.get_feature_names_out())}")

model = svm.LinearSVC(class_weight='balanced', max_iter=1000, verbose=1)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print("SVM Results:")
print(classification_report(y_test, y_pred))

mnb_model = MultinomialNB()
mnb_model.fit(X_train_tfidf, y_train)
y_pred = mnb_model.predict(X_test_tfidf)
print("MultinomialNB Results:")
print(classification_report(y_test, y_pred))

cnb_model = ComplementNB()
cnb_model.fit(X_train_tfidf, y_train)
y_pred = cnb_model.predict(X_test_tfidf)
print("ComplementNB Results:")
print(classification_report(y_test, y_pred))


with open("linearsvc_severity_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("multinomialnb_severity_model.pkl", "wb") as f:
    pickle.dump(mnb_model, f)

with open("complementnb_severity_model.pkl", "wb") as f:
    pickle.dump(cnb_model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
