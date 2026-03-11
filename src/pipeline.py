from ingest import load_train_data
from preprocess import feature_engineering, preprocess_data
from train import optimize_lr, train_model

def run_pipeline():

    df = load_train_data("data/train.csv")

    df = feature_engineering(df)

    X = preprocess_data(df, save=True)
    y = df["Transported"].astype(int)

    best_params = optimize_lr(X,y)

    model = train_model(X,y,best_params)

    print("Pipeline finished")

if __name__ == "__main__":
    run_pipeline()