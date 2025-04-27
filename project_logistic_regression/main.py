import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ad_data = pd.read_csv("advertising.csv")

print(ad_data.head())
print(ad_data.info())
print(ad_data.describe())

ad_data["Age"].hist(bins=30)
plt.xlabel("Age")
plt.show()

sns.jointplot(x="Age", y="Area Income", data=ad_data)
plt.show()

sns.jointplot(
    x="Age", y="Daily Time Spent on Site", data=ad_data, color="red", kind="kde"
)
plt.show()

sns.jointplot(
    x="Daily Time Spent on Site", y="Daily Internet Usage", data=ad_data, color="green"
)
plt.show()

sns.pairplot(ad_data, hue="Clicked on Ad", palette="bwr")
plt.show()

# Logistic regression
X = ad_data[
    ["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Male"]
]
y = ad_data["Clicked on Ad"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
