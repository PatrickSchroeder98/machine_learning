import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic_train.csv")

#print(df.head())

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
    age = cols[0]
    pclass = cols[1]

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


# TODO update README