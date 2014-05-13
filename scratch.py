__author__ = 'gavin'
from sklearn.ensemble import AdaBoostRegressor
from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]

regressor = AdaBoostRegressor(n_estimators=100)
regressor.fit(X_train, y_train)

train_err = mean_squared_error(y_train, regressor.predict(X_train))
print "Training Error = " + str(train_err)
test_err = mean_squared_error(y_test, regressor.predict(X_test))
print "Testing Error = " + str(test_err)

regressor.fit(X, y)
y = regressor.predict(x)
print "Prediction for " + str(x) + " = " + str(y)