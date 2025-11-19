from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bc = load_breast_cancer()
X = bc.data
Y = bc.target

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print(X)
print("--")
print(Y)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42,
)

rf.fit(x_train, y_train)

pred = rf.predict(x_test)
proba = rf.predict_proba(x_test)[:, 1]

print("OOB score : ", rf.oob_score_)
print("accuracy : ", accuracy_score(y_test, pred))
print("ROC AUC : ", roc_auc_score(y_test, proba))
print(
    "Classification Report :\n",
    classification_report(y_test, pred, target_names=bc.target_names),
)


imp_mdi = pd.Series(rf.feature_importances_, index=bc.feature_names)

perm = permutation_importance(
    rf, x_test, y_test, scoring="roc_auc", n_repeats=100, random_state=42, n_jobs=-1
)
imp_perm_mean = pd.Series(perm.importances_mean, index=bc.feature_names)

compare = pd.DataFrame(
    {
        "MDI importance": imp_mdi,
        "MDA importance": imp_perm_mean,
    }
).sort_values("MDA importance", ascending=False)

print(compare)
