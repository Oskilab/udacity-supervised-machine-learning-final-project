from __future__ import division
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)

x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]

predictions = []
predictions2 = []
predictions3 = []
predictions4 = []
offset = int(0.7 * len(X))

for i in range(10):
    X, y = shuffle(boston.data, boston.target)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    regressor = GradientBoostingRegressor(max_depth=20, n_estimators=140)
    regressor2 = DecisionTreeRegressor(max_depth=6)
    regressor3 = LinearRegression()
    regressor4 = RandomForestRegressor()
    regressor.fit(X_train, y_train)
    regressor2.fit(X_train, y_train)
    regressor3.fit(X_train, y_train)
    regressor4.fit(X_train, y_train)
    y_pred = regressor.predict(x)
    y_pred2 = regressor2.predict(x)
    y_pred3 = regressor3.predict(x)
    y_pred4 = regressor4.predict(x)
    predictions.append(y_pred)
    predictions2.append(y_pred2)
    predictions3.append(y_pred3)
    predictions4.append(y_pred4)
    print "\nPrediction = " + str(y_pred)
    print "Prediction = " + str(y_pred2)
    print "Prediction = " + str(y_pred3)
    print "Prediction = " + str(y_pred4)

print '\n'
print 'Boosting max', np.max(predictions), 'min', np.min(predictions), 'variance', np.max(predictions) - np.min(predictions)
print 'Decision tree max', np.max(predictions2), 'min', np.min(predictions2), 'variance', np.max(predictions2) - np.min(predictions2)
print 'Random forest max', np.max(predictions4), 'min', np.min(predictions4), 'variance', np.max(predictions4) - np.min(predictions4)
print 'Linear regression max', np.max(predictions3), 'min', np.min(predictions3), 'variance', np.max(predictions3) - np.min(predictions3)

regressor = GradientBoostingRegressor(max_depth=20, n_estimators=140)
regressor.fit(X, y)
y_pred = regressor.predict(x)
print "Prediction for " + str(x) + " = " + str(y_pred)