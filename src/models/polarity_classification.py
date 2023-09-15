from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import pandas as pd
from pathlib import Path
import os
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from pathlib import Path
import os

# Load dataset
abs_path = Path(".").absolute()
file = str(abs_path) + os.sep + "data/processed/labeled_messages_processed.csv"
df = pd.read_csv(file)

# Split data into train and test sets
X = df["message"].values
y = df["polarity classification"].astype(float).values
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Initialize empty dataframe to hold results
results_df = pd.DataFrame(columns=["Classifiers", "F1 Scores"])
classifiers = []
f1_scores = []

# Classifiers to be tested
models = [["SGD", SGDClassifier(loss='hinge', penalty='l2',
                                alpha=1e-3, random_state=42,
                                max_iter=10, tol=None)],
          ["Multinomial Naive Bayes", MultinomialNB(force_alpha=True)],
          ["Gradient Boosting Classifier", GradientBoostingClassifier(n_estimators=100,
                                                                      learning_rate=1.0,
                                                                      max_depth=1,
                                                                      random_state=0)],
          ["Decision Tree", DecisionTreeClassifier(random_state=0)]]

for model in models:
    # Sequentially apply a list of transforms and a final estimator
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', model[1]),
                    ])

    # Train the model
    clf.fit(X_train, y_train)

    # Get predictions
    y_pred = clf.predict(X_test)

    # Evaluation metrics
    print(f"\n{model[0]} Classification Report:")
    print(classification_report(y_test, y_pred))
    f1 = f1_score(y_test, y_pred, average="micro")

    # Add classifier and its results to lists to be added to results dataframe
    classifiers.append(model[0])
    f1_scores.append(f1)

# Create csv file to hold results
results_df["Classifiers"] = classifiers
results_df["F1 Scores"] = f1_scores
results_df.to_csv((str(abs_path) + os.sep + "data/results/polarity_classification_results.csv"), index=False)
