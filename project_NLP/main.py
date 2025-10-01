import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

yelp = pd.read_csv('yelp.csv')
print(yelp.head())
print(yelp.describe())
print(yelp.shape)
print(yelp.info())
yelp['text length'] = yelp['text'].apply(len)

sns.set_style('white')
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')
plt.show()

sns.boxplot(x='stars',y='text length',data=yelp)
plt.show()
sns.countplot(x='stars',data=yelp)
plt.show()

