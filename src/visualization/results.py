import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Load dataframes
abs_path = Path(".").absolute()
polarity_results_file = str(abs_path) + os.sep + \
    "data/results/polarity_classification_results.csv"
polarity_results_df = pd.read_csv(polarity_results_file)

# Sort in descending order
sorted_df = polarity_results_df.sort_values(by=["F1 Scores"], ascending=False)

# Plot the amount of occurences for each empathy label
plt.figure(figsize=(8, 10))
plt.bar(polarity_results_df["Classifiers"], polarity_results_df["F1 Scores"])
plt.xlabel("Classifiers")
plt.ylabel("F1 Score")
plt.xticks(rotation=25)
ax = plt.gca()
ax.set_ylim([0.75, 1])
plt.savefig(str(abs_path) + os.sep + "figures/results/polarity_classification_results.png")
plt.close()
