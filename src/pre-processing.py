import re
from textblob import TextBlob
import os
from nltk.stem import WordNetLemmatizer
import pandas as pd
from statistics import mean
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
import nltk
nltk.download("wordnet")  # TO DO: write a workaround

# Find the polarity associated with the given empathy
def get_polarity(empathy, empathies_df):
    empathy_index = empathies_df[empathies_df["empathy"]
                                 == empathy.strip()].index.values
    return empathies_df.iloc[empathy_index]["polarity"], empathy_index


# Find the polarities in the empathy df associated with the empathy labels in the messages df
def get_associated_polarity(empathies):
    # If there are multiple empathy labels given, get the average polarity for that message
    if len(empathies) > 1:
        curr_polarities = []
        for empathy in empathies:
            curr_polarity, empathy_index = get_polarity(empathy, empathies_df)
            if len(empathy_index) != 0:
                curr_polarities.append(float(curr_polarity.values))
        return mean(curr_polarities)
    # Otherwise, get the single polarity associated with the given empathy label
    else:
        curr_polarity, empathy_index = get_polarity(empathies[0], empathies_df)
        if len(empathy_index) != 0:
            return float(curr_polarity.values)


# Set any negative polarity to -1, positive polarity to 1, and neutral polarity stays at 0
def classify(polarity):
    polarity = float(polarity)
    if polarity > 0:
        return 1
    elif polarity < 0:
        return -1
    else:
        return 0


# Replace empathy column with encodings representing each label
def get_one_hot_encoding(df):
    mlb = MultiLabelBinarizer()
    ft = mlb.fit_transform(df.pop("empathy"))
    df = df.join(pd.DataFrame(ft,
                              columns=mlb.classes_,
                              index=df.index))
    return df


lemmatizer = WordNetLemmatizer()


def lemmatize_text(message):
    words = message.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)


def fix_spelling(message):
    spell_checker = TextBlob(message)
    return spell_checker.correct()


def remove_special_chars(message):
    return re.sub("[^A-Za-z0-9- ]+", "", str(message))


def preprocess(msgs_df, empathies_df):
    # Drop unused columns
    msgs_df = msgs_df.drop(["num_seen", "ignore"], axis=1)

    # Convert objects to strings
    msgs_df = msgs_df[["message", "empathy"]].astype(str)

    # Apply lemmetization
    msgs_df["message"] = msgs_df["message"].apply(lemmatize_text)

    # Remove special characters
    msgs_df["message"] = msgs_df["message"].apply(remove_special_chars)

    # Fix spelling
    msgs_df["message"] = msgs_df["message"].apply(fix_spelling)

    # Convert empathy column to arrays of strings without spaces
    msgs_df["empathy"] = msgs_df["empathy"].str.replace(", ", ",")
    msgs_df["empathy"] = msgs_df["empathy"].str.split(",")

    return msgs_df, empathies_df


def add_columns(msgs_df, empathies_df):
    # Append column to empathies dataframe that contains the frequencies of each empathy label
    count_df = msgs_df.explode("empathy")
    count_df = count_df["empathy"].value_counts().rename_axis(
        "empathy").reset_index(name="occurences")
    empathies_df = empathies_df.merge(count_df)

    # Create column with average polarity associated with the empathy labels for each message
    msgs_df["polarity"] = msgs_df["empathy"].apply(get_associated_polarity)

    # More processing - drop any rows containing empty cells
    msgs_df[["polarity", "message"]] = msgs_df[[
        "polarity", "message"]].replace("", np.nan)
    msgs_df.dropna(inplace=True)

    # Add a column with a class label (-1, 0, or 1) to the messages dataframe
    msgs_df["polarity classification"] = msgs_df["polarity"].apply(classify)

    # Add columns with encodings representing the empathy classes
    encoded_df = get_one_hot_encoding(msgs_df)
    encoded_df = encoded_df.drop(
        ["polarity", "polarity classification"], axis=1)

    return msgs_df, empathies_df, encoded_df


if __name__ == "__main__":
    # Load dataframes
    abs_path = Path(".").absolute()
    msgs_file = str(abs_path) + os.sep + "data/original/labeled_messages.csv"
    msgs_df = pd.read_csv(msgs_file)

    empathies_file = str(abs_path) + os.sep + "data/original/empathies.csv"
    empathies_df = pd.read_csv(empathies_file)

    msgs_df, empathies_df = preprocess(msgs_df, empathies_df)
    msgs_df, empathies_df, encoded_df = add_columns(msgs_df, empathies_df)

    # Save dataframes as CSV files in data folder
    msgs_df.to_csv((str(abs_path) + os.sep +
                   "data/processed/labeled_messages_processed.csv"), index=False)
    empathies_df.to_csv(str(abs_path) + os.sep +
                        "data/processed/empathies_processed.csv", index=False)
    encoded_df.to_csv(str(abs_path) + os.sep +
                      "data/processed/labeled_messages_encoded.csv", index=False)
