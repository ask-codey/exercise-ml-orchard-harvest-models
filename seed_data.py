"""Makes orchard.csv: 200 made-up apple harvests at Hilltop Orchard.

Ready to use; you don't need to run it. The CSV it made is already in this folder,
and the lessons, the tests and report.py all read that file.

Every number in the file is made up, and the pattern is built in on purpose, so you
can check whether a model finds it. A real dataset's pattern is what you're trying
to find; this one is planted. Each row is one tree's harvest in one season, and no
tree appears twice:

    apples_lbs = 300
                 - 0.45 x (rainfall_inches - 30) squared
                 + 4 for each year of the tree's age (tree_age_years, 3 to 15)
                 + a random wobble (a standard deviation of 20 pounds)
    then rounded to the nearest pound.

So the harvest is biggest at 30 inches of rain and falls off on both sides: a curve,
not a straight line. The summer's average temperature (avg_summer_temp_f) has no
effect at all.

The random numbers come from a fixed seed, 203, so this script makes the same file
every time on the same version of NumPy. NumPy doesn't promise the same random
numbers from a seed across its versions (NEP 19), so the CSV in this folder, not
this script, is what keeps every learner's numbers the same.
"""
import numpy as np
import pandas as pd

SEED = 203
TREES = 200


def make_orchard(seed=SEED, count=TREES):
    """Return the made-up harvests as a DataFrame, one row per tree-season."""
    rng = np.random.default_rng(seed)
    rainfall = (rng.uniform(10, 50, size=count) * 100).round() / 100
    summer_temp = (rng.normal(78, 4, size=count) * 10).round() / 10
    age = rng.integers(3, 16, size=count)
    wobble = rng.normal(0, 20, size=count)
    apples = 300 - 0.45 * (rainfall - 30) ** 2 + 4 * age + wobble

    return pd.DataFrame({
        "tree_id": [f"HO-{1001 + i}" for i in range(count)],
        "rainfall_inches": rainfall,
        "avg_summer_temp_f": summer_temp,
        "tree_age_years": age.astype(int),
        "apples_lbs": apples.round().astype(int),
    })


if __name__ == "__main__":
    orchard = make_orchard()
    orchard.to_csv("orchard.csv", index=False)
    print("Wrote orchard.csv:", len(orchard), "tree-seasons.")
