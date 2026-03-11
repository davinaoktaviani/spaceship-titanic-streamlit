import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

def feature_engineering(df):

    df = df.copy()

    df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else 'Unknown')
    df['Cabin_num'] = df['Cabin'].apply(lambda x: x.split('/')[1] if pd.notna(x) else -1).astype(float)
    df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else 'Unknown')

    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['Group_size'] = df.groupby('Group')['Group'].transform('count')

    spending_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    df['TotalSpending'] = df[spending_cols].sum(axis=1)

    return df


def preprocess_data(df, save=False):

    categorical = ['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
    numerical = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Cabin_num','Group_size','TotalSpending']

    df = df.copy()

    for col in categorical:
        df[col] = df[col].fillna("Unknown")

    for col in numerical:
        df[col] = df[col].fillna(df[col].median())

    encoders = {}

    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[categorical + numerical]

    if save:
        with open("models/preprocessor.pkl","wb") as f:
            pickle.dump(encoders,f)

    return X