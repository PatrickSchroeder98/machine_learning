import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)

print(df.head())

movie_titles = pd.read_csv("Movie_Id_Titles")
print(movie_titles.head())

df = pd.merge(df,movie_titles,on='item_id')
print(df.head())

sns.set_style('white')
print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())
print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
print(ratings.head())

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
print(ratings.head())

plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)
plt.show()

plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)
plt.show()

sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
plt.show()