__author__ = 'gavin'
from numpy import *
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

pipeline = Pipeline([
    ('clf', KNeighborsRegressor())
])
parameters = {
    'clf__n_neighbors': (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14),
}

if __name__ == '__main__':
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='mean_squared_error')
    grid_search.fit(X, y)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])

    regressor = KNeighborsRegressor(n_neighbors=3)
    regressor.fit(X_train, y_train)
    train_err = mean_squared_error(y_train, regressor.predict(X_train))
    print "Training Error = " + str(train_err)
    test_err = mean_squared_error(y_test, regressor.predict(X_test))
    print "Testing Error = " + str(test_err)
    regressor.fit(X, y)
    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    y = regressor.predict(x)
    print "Prediction for " + str(x) + " = " + str(y)
