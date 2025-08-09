import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from data_prep import DataPrep

feature_names = [
    "BADJ EM","BADJ O","BADJ D","BARTHAG","GAMES","W","L","WIN%","EFG%","EFG%D","FTR","FTRD",
    "TOV%","TOV%D","OREB%","DREB%","OP OREB%","OP DREB%","RAW T","2PT%","2PT%D","3PT%","3PT%D",
    "BLK%","BLKED%","AST%","OP AST%","2PTR","3PTR","2PTRD","3PTRD","BADJ T","AVG HGT","EFF HGT",
    "EXP","TALENT","FT%","OP FT%","PPPO","PPPD","ELITE SOS","WAB","BADJ EM RANK","BADJ O RANK",
    "BADJ D RANK","BARTHAG RANK","EFG% RANK","EFGD% RANK","FTR RANK","FTRD RANK","TOV% RANK",
    "TOV%D RANK","OREB% RANK","DREB% RANK","OP OREB% RANK","OP DREB% RANK","RAW T RANK",
    "2PT% RANK","2PT%D RANK","3PT% RANK","3PT%D RANK","BLK% RANK","BLKED% RANK","AST% RANK",
    "OP AST% RANK","2PTR RANK","3PTR RANK","2PTRD RANK","3PTRD RANK","BADJT RANK","AVG HGT RANK",
    "EFF HGT RANK","EXP RANK","TALENT RANK","FT% RANK","OP FT% RANK","PPPO RANK","PPPD RANK",
    "ELITE SOS RANK"
]

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:,1]

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:,1]

    return {
        "train_acc": accuracy_score(y_train, y_train_pred),
        "train_logloss": log_loss(y_train, y_train_prob),
        "train_roc_auc": roc_auc_score(y_train, y_train_prob),

        "test_acc": accuracy_score(y_test, y_test_pred),
        "test_logloss": log_loss(y_test, y_test_prob),
        "test_roc_auc": roc_auc_score(y_test, y_test_prob)
    }

def run_prediction(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "XGBoost": XGBClassifier(
            n_estimators=1000, max_depth=4, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, gamma=1, reg_lambda=1, random_state=42,
            eval_metric='logloss'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_train, y_train, X_test, y_test)

    return results
 