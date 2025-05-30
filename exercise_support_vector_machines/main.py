import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()
print(cancer.keys())
print(cancer["DESCR"])
print(cancer["feature_names"])

df_feat = pd.DataFrame(cancer["data"], columns=cancer["feature_names"])
df_feat.info()
print(cancer["target"])
df_target = pd.DataFrame(cancer["target"], columns=["Cancer"])

sns.displot(df_feat["mean radius"].dropna(), kde=False, bins=30)
plt.show()

sns.displot(df_feat["mean area"].dropna(), kde=False, bins=30)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    df_feat, np.ravel(df_target), test_size=0.30, random_state=101
)
model = SVC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

param_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["rbf"],
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
