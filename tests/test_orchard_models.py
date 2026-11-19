# Tests for orchard_models.py. Run them with: python -m pytest (in the exercise folder, with the virtual environment
# active). Each test calls one of your functions, on the 200 harvests in orchard.csv or on a small table written in
# the test, and checks what comes back. A failing test says what it called, what it expected and what came back.
# None of them checks how accurate a model is: they check the curve features' columns, the tree's settings, and that
# each score is mean_absolute_error on the right rows. You don't need to change this file.
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from orchard_models import FEATURES, LABEL, build_tree, compare_models, curve_features


def check(call, got, expected):
    """Fails with a message that says what the call returned and what it should return."""
    assert got == expected, f"{call} should return {expected!r}, but it returned {got!r}."


def as_lists(array):
    """An array or a table as plain lists of numbers, so it can be compared and printed."""
    return np.asarray(array, dtype=float).tolist()


# curve_features

def test_1_curve_features_degree_2_of_2_and_3_gives_2_4_and_3_9():
    curve_train, curve_test = curve_features(pd.DataFrame({"rainfall_inches": [2, 3]}),
                                             pd.DataFrame({"rainfall_inches": [10]}), 2)
    check("the training rows of curve_features(2 and 3, 10, degree 2)", as_lists(curve_train), [[2, 4], [3, 9]])


def test_2_curve_features_degree_2_changes_the_test_row_10_into_10_and_100():
    curve_train, curve_test = curve_features(pd.DataFrame({"rainfall_inches": [2, 3]}),
                                             pd.DataFrame({"rainfall_inches": [10]}), 2)
    check("the test rows of curve_features(2 and 3, 10, degree 2)", as_lists(curve_test), [[10, 100]])


def test_3_curve_features_degree_2_of_two_features_2_and_5_gives_2_5_4_10_25():
    row = pd.DataFrame({"rainfall_inches": [2], "tree_age_years": [5]})
    curve_train, curve_test = curve_features(row, row, 2)
    check("curve_features of the row 2, 5 with degree 2", as_lists(curve_train), [[2, 5, 4, 10, 25]])


def test_4_curve_features_degree_2_on_the_orchard_gives_160_by_5_and_40_by_5(orchard_split):
    X_train, X_test, y_train, y_test = orchard_split
    curve_train, curve_test = curve_features(X_train, X_test, 2)
    check("the shape of the training curve features, degree 2", np.shape(curve_train), (160, 5))
    check("the shape of the test curve features, degree 2", np.shape(curve_test), (40, 5))


def test_5_curve_features_degree_3_on_the_orchard_gives_9_columns(orchard_split):
    X_train, X_test, y_train, y_test = orchard_split
    curve_train, curve_test = curve_features(X_train, X_test, 3)
    check("the shape of the training curve features, degree 3", np.shape(curve_train), (160, 9))
    check("the shape of the test curve features, degree 3", np.shape(curve_test), (40, 9))


def test_6_curve_features_degree_1_gives_back_the_two_features_with_no_column_of_ones(orchard_split):
    X_train, X_test, y_train, y_test = orchard_split
    curve_train, curve_test = curve_features(X_train, X_test, 1)
    check("the first 2 training rows of curve_features(..., degree 1)", as_lists(curve_train)[:2],
          as_lists(X_train)[:2])


# build_tree

def test_7_build_tree_3_0_returns_a_decision_tree_regressor():
    tree = build_tree(3, 0)
    assert isinstance(tree, DecisionTreeRegressor), \
        f"build_tree(3, 0) should return a DecisionTreeRegressor, but it returned {tree!r}."


def test_8_build_tree_3_0_has_max_depth_3_and_random_state_0():
    tree = build_tree(3, 0)
    check("build_tree(3, 0).max_depth", tree.max_depth, 3)
    check("build_tree(3, 0).random_state", tree.random_state, 0)


def test_9_build_tree_none_7_has_no_depth_limit_and_random_state_7():
    tree = build_tree(None, 7)
    check("build_tree(None, 7).max_depth", tree.max_depth, None)
    check("build_tree(None, 7).random_state", tree.random_state, 7)


def test_10_build_tree_returns_a_tree_that_has_not_been_fit():
    tree = build_tree(3, 0)
    assert not hasattr(tree, "tree_"), \
        "build_tree(3, 0) returned a tree that has already been fit: return it without calling fit."


# compare_models

def test_11_compare_models_average_of_10_20_30_scores_6_667_in_training_and_10_on_25_and_5():
    scores = compare_models({"average": DummyRegressor(strategy="mean")},
                            pd.DataFrame({"x": [1, 2, 3]}), pd.Series([10, 20, 30]),
                            pd.DataFrame({"x": [4, 5]}), pd.Series([25, 5]))
    check("the names compare_models returned", list(scores), ["average"])
    assert scores["average"]["train_mae"] == pytest.approx(20 / 3), \
        f"The training MAE should be 6.666..., but it was {scores['average']['train_mae']!r}."
    assert scores["average"]["test_mae"] == pytest.approx(10.0), \
        f"The test MAE should be 10.0, but it was {scores['average']['test_mae']!r}."


def test_12_compare_models_a_straight_line_on_exact_rows_scores_0_and_0():
    scores = compare_models({"line": LinearRegression()},
                            pd.DataFrame({"x": [1, 2, 3, 4]}), pd.Series([3, 5, 7, 9]),
                            pd.DataFrame({"x": [10, 20]}), pd.Series([21, 41]))
    assert scores["line"]["train_mae"] == pytest.approx(0, abs=1e-9), \
        f"A line through 2x + 1 should have a training MAE of 0, but it was {scores['line']['train_mae']!r}."
    assert scores["line"]["test_mae"] == pytest.approx(0, abs=1e-9), \
        f"A line through 2x + 1 should have a test MAE of 0, but it was {scores['line']['test_mae']!r}."


def test_13_compare_models_returns_every_name_in_order_with_train_mae_and_test_mae(orchard_split):
    X_train, X_test, y_train, y_test = orchard_split
    scores = compare_models({"line": LinearRegression(), "tree": DecisionTreeRegressor(max_depth=3, random_state=0)},
                            X_train, y_train, X_test, y_test)
    check("the names compare_models returned", list(scores), ["line", "tree"])
    check("the keys of each score", sorted(scores["line"]), ["test_mae", "train_mae"])


def test_14_compare_models_scores_equal_mean_absolute_error_on_the_160_and_40_trees(orchard_split):
    X_train, X_test, y_train, y_test = orchard_split
    scores = compare_models({"tree": DecisionTreeRegressor(max_depth=3, random_state=0)},
                            X_train, y_train, X_test, y_test)
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_train, y_train)
    assert scores["tree"]["train_mae"] == pytest.approx(mean_absolute_error(y_train, tree.predict(X_train)), abs=1e-9), \
        f"The training MAE should be {mean_absolute_error(y_train, tree.predict(X_train))!r}, but it was {scores['tree']['train_mae']!r}."
    assert scores["tree"]["test_mae"] == pytest.approx(mean_absolute_error(y_test, tree.predict(X_test)), abs=1e-9), \
        f"The test MAE should be {mean_absolute_error(y_test, tree.predict(X_test))!r}, but it was {scores['tree']['test_mae']!r}."


def test_15_compare_models_fits_on_the_training_trees_only(orchard_split):
    X_train, X_test, y_train, y_test = orchard_split
    scores = compare_models({"average": DummyRegressor(strategy="mean")}, X_train, y_train, X_test, y_test)
    expected = mean_absolute_error(y_test, [y_train.mean()] * len(y_test))
    assert scores["average"]["test_mae"] == pytest.approx(expected, abs=1e-9), \
        (f"The average model should guess the 160 training trees' average, {y_train.mean()!r}, so its test MAE "
         f"should be {expected!r}, but it was {scores['average']['test_mae']!r}.")


def test_16_compare_models_works_with_curve_features_from_curve_features(orchard_split):
    X_train, X_test, y_train, y_test = orchard_split
    curve_train, curve_test = curve_features(X_train, X_test, 2)
    scores = compare_models({"curve": LinearRegression()}, curve_train, y_train, curve_test, y_test)
    line = LinearRegression().fit(curve_train, y_train)
    assert scores["curve"]["test_mae"] == pytest.approx(mean_absolute_error(y_test, line.predict(curve_test)), abs=1e-9), \
        f"The curve's test MAE should be {mean_absolute_error(y_test, line.predict(curve_test))!r}, but it was {scores['curve']['test_mae']!r}."
