"""Hilltop Orchard: three functions for you to finish.

The docstring under each function says what it should do, with examples.
Replace each raise NotImplementedError("Not built yet.") line with your own code.
Return each answer: the tests read what comes back, not what's printed.

X_train and X_test are pandas tables (DataFrames) of features, one row per
tree-season, and y_train and y_test are the matching apples_lbs columns, the way
train_test_split gives them back. The imports below are everything the three
functions need.
"""
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

# The columns the models look at (their features), in this order. Ready to use.
FEATURES = ["rainfall_inches", "tree_age_years"]

# The column the models learn to guess (their label). Ready to use.
LABEL = "apples_lbs"


def curve_features(X_train, X_test, degree):
    """Return the curve features of the training rows and of the test rows, as two arrays.

    Make a PolynomialFeatures(degree=degree, include_bias=False). Fit it on the
    training rows and change them with fit_transform(X_train); change the test
    rows with transform(X_test), using what was fit on the training rows.

    Return both: curve_train, curve_test

    One feature, degree 2: each value, then its square.
    X_train holds 2 and 3, X_test holds 10 -> [[2, 4], [3, 9]] and [[10, 100]]

    Two features, degree 2: a, b, a squared, a times b, b squared.
    A row of 2 and 5 -> [2, 5, 4, 10, 25]

    With the 160 training trees and 40 test trees of orchard.csv and FEATURES,
    degree 2 gives 5 columns: shapes (160, 5) and (40, 5).
    """
    raise NotImplementedError("Not built yet.")


def build_tree(max_depth, seed):
    """Return a new DecisionTreeRegressor that hasn't learned anything yet.

    Give it max_depth=max_depth and random_state=seed. Don't fit it:
    compare_models does that.

    build_tree(3, 0)    -> a tree at most 3 questions deep, random_state 0
    build_tree(None, 7) -> a tree with no depth limit, random_state 7
    """
    raise NotImplementedError("Not built yet.")


def compare_models(models, X_train, y_train, X_test, y_test):
    """Fit each model on the training rows, and return its training MAE and test MAE.

    models is a dictionary of a name to a model that hasn't been fit yet, such as
    {"straight line": LinearRegression()}. For each one, in order:
    fit it on X_train and y_train, then work out
      the training MAE: mean_absolute_error(y_train, its predictions for X_train)
      the test MAE:     mean_absolute_error(y_test, its predictions for X_test)

    Return a dictionary of each name to a dictionary of its two scores:
    {"straight line": {"train_mae": ..., "test_mae": ...}}
    Don't round the scores.

    Worked example: the model is DummyRegressor(strategy="mean"), which always
    guesses the training average.
    Training labels 10, 20, 30 -> it guesses 20 for every row.
      training MAE: (10 + 0 + 10) / 3 = 6.666...
    Test labels 25 and 5 -> misses of 5 and 15.
      test MAE: (5 + 15) / 2 = 10.0
    -> {"average": {"train_mae": 6.666666666666667, "test_mae": 10.0}}
    """
    raise NotImplementedError("Not built yet.")
