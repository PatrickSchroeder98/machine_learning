"""
Exercise from ML Python course.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

lm = LinearRegression()
df = pd.read_csv("USA_Housing.csv")

# Data columns (total 7 columns):
#  #   Column                        Non-Null Count  Dtype
# ---  ------                        --------------  -----
#  0   Avg. Area Income              5000 non-null   float64
#  1   Avg. Area House Age           5000 non-null   float64
#  2   Avg. Area Number of Rooms     5000 non-null   float64
#  3   Avg. Area Number of Bedrooms  5000 non-null   float64
#  4   Area Population               5000 non-null   float64
#  5   Price                         5000 non-null   float64
#  6   Address                       5000 non-null   object
# dtypes: float64(6), object(1)

sns.pairplot(df)
plt.show()

sns.displot(df["Price"])
plt.show()

# print(df.columns)
# Index(['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
#        'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address'],

# dataset = df
# dataset.drop(['Address'])
# sns.heatmap(dataset.corr())

# plt.show()

X = df[
    [
        "Avg. Area Income",
        "Avg. Area House Age",
        "Avg. Area Number of Rooms",
        "Avg. Area Number of Bedrooms",
        "Area Population",
    ]
]

y = df[["Price"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101
)
lm.fit(X_train, y_train)
print(lm.intercept_)
print(lm.coef_)

cdf = pd.DataFrame(lm.coef_.T, X.columns, columns=["Coeff"])
print(cdf)

# print(housing.keys())

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
plt.show()

sns.displot((y_test - predictions))
plt.show()

m1 = metrics.mean_absolute_error(y_test, predictions)
m2 = metrics.mean_squared_error(y_test, predictions)
m3 = np.sqrt(metrics.mean_squared_error(y_test, predictions))

print(m1)
print(m2)
print(m3)
