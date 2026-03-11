import pandas as pd

def load_train_data(path):
    df = pd.read_csv(path)
    return df

def load_test_data(path):
    df = pd.read_csv(path)
    return df