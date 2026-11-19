# Splits Hilltop Orchard's 200 harvests, builds the curve features and the trees, and compares five models with your
# three functions from orchard_models.py, printing what each step gave, so you can see them at work. Ready to use.
# Run it with: python report.py
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from orchard_models import FEATURES, LABEL, build_tree, compare_models, curve_features

orchard = pd.read_csv("orchard.csv")
X_train, X_test, y_train, y_test = train_test_split(
    orchard[FEATURES], orchard[LABEL], test_size=0.2, random_state=42
)
print("1) training trees:", len(X_train))
print("2) test trees:", len(X_test))

curve_train, curve_test = curve_features(X_train, X_test, 2)
print("3) curve features, training:", curve_train.shape)
print("4) curve features, test:", curve_test.shape)

tree = build_tree(3, 0)
print("5) tree settings:", "max_depth", tree.max_depth, "random_state", tree.random_state)

# The curve learns from its own columns, so it gets its own call. (A Pipeline, in Module 4, joins the two steps.)
scores = compare_models({
    "average (baseline)": DummyRegressor(strategy="mean"),
    "straight line": LinearRegression(),
    "tree, depth 3": tree,
    "tree, no limit": build_tree(None, 0),
}, X_train, y_train, X_test, y_test)
scores.update(compare_models({"curve, degree 2": LinearRegression()}, curve_train, y_train, curve_test, y_test))

for number, (name, score) in enumerate(scores.items(), start=6):
    print(f"{number}) {name}: training MAE {score['train_mae']:.1f}, test MAE {score['test_mae']:.1f}")
