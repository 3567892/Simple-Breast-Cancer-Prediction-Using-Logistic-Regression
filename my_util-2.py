import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset_csv(csv_file_path):
    # Loads the csv file
    df = pd.read_csv(csv_file_path)

    # Drops the column with too many missing values
    df.drop(columns=['Unnamed: 32'], inplace=True)

    # Extracts features (X)
    X = df.drop(columns=["diagnosis"]).to_numpy()

    # Extracts labels (Y)
    Y = df["diagnosis"].to_numpy()

    # List of classes
    classes = np.array(["B", "M"])

    # Splitting into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

    # Converting labels to binary
    Y_train = np.where(Y_train == 'B', 0, 1)
    Y_test = np.where(Y_test == 'B', 0, 1)

    return X_train, X_test, Y_train, Y_test, classes

    