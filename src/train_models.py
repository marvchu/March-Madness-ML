from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from data_prep import prepare_data

def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter = 10000)
    lr.fit(X_train, y_train)

    l_train_acc = accuracy_score(y_train, lr.predict(X_train))
    l_train_mcc = matthews_corrcoef(y_train, lr.predict(X_train))
    l_train_f1 = f1_score(y_train, lr.predict(X_train))

    l_test_acc = accuracy_score(y_test, lr.predict(X_test))
    l_test_mcc = matthews_corrcoef(y_test, lr.predict(X_test))
    l_test_f1 = f1_score(y_test, lr.predict(X_test))

    print(f"Test Accuracy: {l_test_acc:.4f}")
    print(f"MCC: {l_test_mcc:.4f}")
    print(f"F1 Score: {l_test_f1:.4f}")