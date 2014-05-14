from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

predictions = []

# for i in range(13):
#     n = (i + 1) * 10
#     print n
#     m = i + 1
#     regressor = GradientBoostingRegressor(max_depth=m)
#     regressor.fit(X, y)
#     x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
#     y = regressor.predict(x)
#     predictions.append(y)
#     print "Prediction for " + str(x) + " = " + str(y)
#

x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
regressor = GradientBoostingRegressor(max_depth=1)
regressor.fit(X, y)
y = regressor.predict(x)
print "Prediction for " + str(x) + " = " + str(y)