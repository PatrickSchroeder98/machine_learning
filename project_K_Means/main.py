import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

df = pd.read_csv('College_Data',index_col=0)
print(df.head())
print(df.info())
print(df.describe())

sns.set_style('whitegrid')
sns.lmplot(data=df, x='Room.Board', y='Grad.Rate', hue='Private', palette='coolwarm', height=6, aspect=1, fit_reg=False)
plt.show()

sns.lmplot(data=df, x='Outstate', y='F.Undergrad', hue='Private', palette='coolwarm', height=6, aspect=1, fit_reg=False)
plt.show()

sns.set_style('darkgrid')
g = sns.FacetGrid(df, hue="Private", palette='coolwarm', height=6, aspect=2)
g = g.map(plt.hist,'Outstate', bins=20, alpha=0.7)
plt.show()

g = sns.FacetGrid(df,hue="Private", palette='coolwarm', height=6, aspect=2)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)
plt.show()

# Fixing a bug in data - graduation rate can't be higher than 100%
df['Grad.Rate']['Cazenovia College'] = 100

g = sns.FacetGrid(df, hue="Private", palette='coolwarm', height=6, aspect=2)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)
plt.show()

# KMeans creation
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private', axis=1))
print(kmeans.cluster_centers_)


# Creation of new column
def converter(cluster):
    if cluster == 'Yes':
        return 0
    else:
        return 1


df['Cluster'] = df['Private'].apply(converter)
print(df.head())

print(confusion_matrix(df['Cluster'], kmeans.labels_))
print(classification_report(df['Cluster'], kmeans.labels_))
