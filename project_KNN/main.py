import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix


df = pd.read_csv("KNN_Project_Data", index_col=0)
print(df.head())

sns.pairplot(df,hue='TARGET CLASS', palette='bwr')
plt.show()

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))  # scaled df
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])  # except TARGET CLASS
print(df_feat.head(3))

# KNN
X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
print(pred)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# Looking for optimal n_neighbors
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# Best option: K = 37; alternatively K = 23
knn = KNeighborsClassifier(n_neighbors=37)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
print(pred)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
