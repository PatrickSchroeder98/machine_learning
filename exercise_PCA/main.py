import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df.head())

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

print(scaled_data.shape)
print(x_pca.shape)

# Scatter Plot from x_pca
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First Principal component')
plt.ylabel('Second Principal Component')
plt.show()

# PCA components from First and econd principal component
print(pca.components_)
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
print(df_comp)

plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)
plt.show()
