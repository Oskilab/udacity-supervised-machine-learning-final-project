My best guess for the price of the house corresponding to the feature vector [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13] is $24,822.32. 

I experimented with scikit-learn's implementations of multiple linear regression, decision trees, random forests, k-nearest neighbors, and gradient boosting regressors. I attempted to use sklearn's implementation of AdaBoost, but it threw exceptions. Though I expect that neural networks could be used effectively, the test error of the sample network implemented with PyBrain was high. For this reason I did not train a neural network.

I used grid search with cross validation to optimize the hyperparameters for each model, and then retrained the models on the full data sets before predicting. I chose to predict the value of the response variable for the test instance using the gradient boosting regressor because it yielded the lowest mean squared error on the testing set. Other models produced smaller training errors, but their high test errors indicated that they had overfit the training data. 

K-nearest neighbors had the worst performance of the estimators I tried. Using six neighbors, the training error was 20.3476020088 and the error on the test set was 49.5608406433. The model predicted the value of the response variable to be 20.4, or $20,400.

The multiple linear regression model's training error was 20.0332569078, and its testing error was 27.2642956428. It predicted the price of the house to be $19,662.16.
The decision tree regressor's best training error was 9.6439512462, and its best error on the test set was 12.1813812371. It predicted 21.62974359.
The random forest with 20 estimators and no maximum depth produced a training error of 2.05575402542, a testing error of 13.0226768092, and predicted 24.25.
The gradient-boosting regressor with 140 estimators and a maximum depth of 20 had a training error equal to 0.0325657890507 and a testing error equal to 8.68958674752. It predicted 24.82231953, or $24,822.32.

