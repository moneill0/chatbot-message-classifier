import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Load dataframes
abs_path = Path(".").absolute()
msgs_file = str(abs_path) + os.sep + \
    "data/processed/labeled_messages_processed.csv"
msgs_df = pd.read_csv(msgs_file)

empathies_file = str(abs_path) + os.sep + \
    "data/processed/empathies_processed.csv"
empathies_df = pd.read_csv(empathies_file)

# Sort in descending order
sorted_df = empathies_df.sort_values(by=["occurences"], ascending=False)

# Plot the amount of occurences for each empathy label
plt.figure(figsize=(20, 15))
plt.bar(sorted_df["empathy"], sorted_df["occurences"])
plt.xticks(rotation=75)
plt.xlabel("Empathy Types")
plt.ylabel("Number of Occurences")
plt.savefig(str(abs_path) + os.sep + "figures/distributions/empathy_distribution.png")
plt.close()

# Plot the amount of times each polarity classification appears in the messages dataset
polarity_class_freq = msgs_df.groupby(["polarity classification"]).size()

classes = [-1, 0, 1]
plt.bar(classes, polarity_class_freq)
plt.xlabel("Polarity Classes")
plt.ylabel("Number of Occurences")
plt.xticks(np.arange(min(classes), max(classes), 1.0))
plt.savefig(str(abs_path) + os.sep + "figures/distributions/polarity_distribution.png")
