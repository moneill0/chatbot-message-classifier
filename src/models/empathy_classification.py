from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix
from pathlib import Path
import os
from sklearn.pipeline import Pipeline

# Read encoded dataset
abs_path = Path(".").absolute()
file = str(abs_path) + os.sep + "data/processed/labeled_messages_encoded.csv"
df = pd.read_csv(file)

# Split data into train and test sets
X = df["message"].values
y = csr_matrix(df[df.columns[1:]].values)  # Compressed Sparse Row matrix
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Initialize classifier pipeline with data vectorization and Linear SVC classifier
clf = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', OneVsRestClassifier(LinearSVC(dual="auto"))),
                ])

# Train the model
clf.fit(X_train, y_train)

# Apply classifier to make predictions on test set
y_pred = clf.predict(X_test)

# Evaluate precision, recall, and f1 score
print("micro average precision score: {:.2f}".format(
    precision_score(y_test, y_pred, average="micro")))
print("micro average recall score: {:.2f}".format(
    recall_score(y_test, y_pred, average="micro")))
print("micro average f1 score: {:.2f}".format(
    f1_score(y_test, y_pred, average="micro")))
