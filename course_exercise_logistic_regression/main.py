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
