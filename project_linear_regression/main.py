import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('Ecommerce Customers')

#print(df.head())
#print(df.describe())
#print(df.info())

sns.jointplot(data=df, x='Time on Website', y='Yearly Amount Spent')
plt.show()

sns.jointplot(data=df, x='Time on App', y='Yearly Amount Spent')
plt.show()

sns.jointplot(data=df, x='Time on App', y='Length of Membership', kind='hex')
plt.show()

sns.pairplot(df)
plt.show()

# Plots showed that Length of Membership is the most correlated with Yearly Amount Spent

sns.lmplot(data=df, x='Length of Membership', y='Yearly Amount Spent')
plt.show()

#print(df.columns)
y = df['Yearly Amount Spent']
x = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.coef_)

predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.show()

print('MAE: ', metrics.mean_absolute_error(y_test, predictions))
print('MSE: ', metrics.mean_squared_error(y_test, predictions))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print('Explained: ', metrics.explained_variance_score(y_test, predictions))

sns.displot((y_test-predictions), bins=50)
plt.show()

print(pd.DataFrame(lm.coef_, x.columns, columns=['Coeff']))

#                           Coeff
# Avg. Session Length   25.981550
# Time on App           38.590159
# Time on Website        0.190405
# Length of Membership  61.279097

# Conclusion: Website needs development to catch up OR the app because it is more used.
# Length of membership needs to be considered as well.