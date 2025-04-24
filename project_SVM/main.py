import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
# Model for predicting the species of iris

iris = sns.load_dataset('iris')
sns.pairplot(iris, hue='species', palette='Dark2')
plt.show()

setosa = iris[iris['species']=='setosa']
sns.kdeplot( x=setosa['sepal_width'], y=setosa['sepal_length'],
                 cmap="plasma", fill=True, thresh=0.05)
plt.show()

X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

svc_model = SVC()
svc_model.fit(X_train,y_train)
predictions = svc_model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
