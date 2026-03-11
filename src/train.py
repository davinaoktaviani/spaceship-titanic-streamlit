import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle

RANDOM_STATE = 42

def optimize_lr(X,y):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):

        params = {
            "C": trial.suggest_float("C",0.001,100,log=True),
            "penalty": trial.suggest_categorical("penalty",["l1","l2"]),
            "solver": trial.suggest_categorical("solver",["liblinear","saga"]),
            "max_iter": trial.suggest_int("max_iter",100,2000)
        }

        model = LogisticRegression(**params)

        score = cross_val_score(model,X,y,cv=cv,scoring="accuracy").mean()

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=30)

    return study.best_params


def train_model(X,y,best_params):

    model = LogisticRegression(**best_params)

    model.fit(X,y)

    with open("models/model.pkl","wb") as f:
        pickle.dump(model,f)

    return model