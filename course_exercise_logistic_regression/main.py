import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("titanic_train.csv")

# Data exploration

sns.set_style('whitegrid')

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()  # A lot of Cabin data is missing

sns.countplot(x='Survived', data=df)
plt.show()

sns.countplot(x='Survived', hue='Sex', palette='RdBu_r', data=df)
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=df)
plt.show()

sns.displot(df['Age'].dropna(), kde=False, bins=30)
plt.show()

df['Age'].plot.hist(bins=30)
plt.show()

sns.countplot(x='SibSp', data=df)
plt.show()

df['Fare'].hist(bins=40, figsize=(10, 4))
plt.show()

plt.figure(figsize=(10, 7))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.show()


# Fixing missing data
def impute_age(cols):
    """Function that fills the missing age numbers based on average values from one of 3 classes. """
    age = cols.iloc[0]
    pclass = cols.iloc[1]

    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis=1)

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# The column 'Cabin' and then rows has been dropped
df.drop('Cabin',axis=1,inplace=True)
df.dropna(inplace=True)

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

gender = pd.get_dummies(df['Sex'], drop_first=True)
embark = pd.get_dummies(df['Embarked'], drop_first=True)

train = pd.concat([df, gender, embark], axis=1)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.drop('PassengerId', axis=1, inplace=True)

print(train.head())  # Data has been prepared for ML operations

# Logistic regression
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
