# Shared set-up for the tests. You don't need to change this file.
# Each test that asks for orchard gets its own fresh copy of the 200 harvests in orchard.csv, read from this
# exercise's folder, and each test that asks for orchard_split gets those harvests split the way report.py splits
# them: 160 training trees and 40 test trees, with random_state 42.
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

ORCHARD = Path(__file__).parent.parent / "orchard.csv"


@pytest.fixture
def orchard():
    """The 200 harvests in orchard.csv, freshly read for one test."""
    return pd.read_csv(ORCHARD)


@pytest.fixture
def orchard_split(orchard):
    """X_train, X_test, y_train, y_test: the harvests' rainfall and age, split 160 / 40 with random_state 42."""
    return train_test_split(orchard[["rainfall_inches", "tree_age_years"]], orchard["apples_lbs"],
                            test_size=0.2, random_state=42)
