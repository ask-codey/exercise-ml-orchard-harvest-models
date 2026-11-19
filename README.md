# Exercise: Orchard Harvest Models

Exercise code for the lesson **Exercise: Orchard Harvest Models** in the Machine Learning with Python course on [ask-codey.com](https://www.ask-codey.com).

Hilltop Orchard wants to guess each tree's apple harvest from the season's rain and the tree's age, and its harvest rises with rain up to about 30 inches, then falls. A straight line can't follow that. You'll write three functions in `orchard_models.py` that make curve features, build a regression tree, and compare models fairly: each one fit on the same training trees and scored on the same test trees, its training error beside its test error. Tests call each function and check what it returns.

## What's here

- `orchard.csv`: 200 harvests, our own made-up data: each row's `tree_id`, `rainfall_inches`, `avg_summer_temp_f`, `tree_age_years` and `apples_lbs`. The lessons, the tests and `report.py` all read this file. Ready to use.
- `seed_data.py`: the script that made `orchard.csv`, so you can see how the data was built and the pattern planted in it. You don't need to run it: NumPy doesn't promise the same random numbers from a seed on every version, so the CSV, not the script, is what keeps everyone's numbers the same. Ready to use.
- `orchard_models.py`: the three functions, `curve_features`, `build_tree` and `compare_models`, with the `FEATURES` and `LABEL` they use. Each function stops with `NotImplementedError: Not built yet.` for now. The docstring under each says what it should do, with examples. For you to finish.
- `explore.ipynb`: a notebook for looking at the harvests, with a scatter plot of rain against apples, and for trying your functions as you write them. The tests don't read it. Ready to use, and yours to change.
- `report.py`: splits the harvests, makes the curve features and the trees, and compares five models with your three functions, printing what each step gave. Ready to use.
- `tests/test_orchard_models.py`: tests that call your functions and check what comes back. You don't need to change this file.
- `tests/conftest.py`: gives each test a fresh copy of the 200 harvests from `orchard.csv`, and the same harvests split into 160 training trees and 40 test trees. Ready to use.
- `requirements.txt`: the packages this exercise needs, each at an exact version: scikit-learn, pandas, NumPy, matplotlib (for the notebook's scatter plot), ipykernel (so VS Code can run the notebook) and pytest. Ready to use.
- `pytest.ini`: tells pytest where the tests are and where to find `orchard_models.py`. Ready to use.
- `.gitignore`: tells Git to leave out the virtual environment and the files Python, pytest and the notebook make as they run. Ready to use.

## Before you start

The packages go in a virtual environment: a private set of packages for this folder only. In VS Code's terminal, in this folder, run these three commands.

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On a Mac (zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- The first command makes the virtual environment, in a folder named `.venv`.
- The second turns it on for this terminal: the prompt starts with `(.venv)` from then on.
- The third installs the packages `requirements.txt` lists into it.

*Virtual Environments and pip* explains all three. The `.venv` folder is in `.gitignore`, so Git leaves it out: these commands make it again on any computer.

If PowerShell answers that running scripts is disabled on this system, run this once, then run the second command again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Each new terminal starts with the virtual environment off. If the prompt doesn't start with `(.venv)`, run the second command again. (VS Code often turns it on for you.)

To look at the harvests, open `explore.ipynb`, choose the `.venv` kernel with **Select Kernel** (top right), and run the cells.

## Check your work

```bash
python -m pytest
```

At first every test fails: that is expected, because each function stops with a `NotImplementedError` for now. Each function you finish turns its tests green. A failing test's message says what it called, what it expected and what came back, for example *build_tree(3, 0).max_depth should return 3, but it returned None.*

No test checks how accurate a model is: that depends on the data's random wobble. The tests check the curve features' columns, the tree's settings, and that each score is the mean absolute error on the right rows.

To see your functions at work on the whole file, run:

```bash
python report.py
```

## The rules

- Keep each function's name and parameters as they are, and keep `FEATURES` and `LABEL` as they are.
- Fit `PolynomialFeatures` on the training rows only, and use `transform` for the test rows.
- `build_tree` passes `seed` on as `random_state`, and returns a tree that hasn't been fit yet.
- `compare_models` fits every model on the training rows only, never on the test rows.
- Return each answer; don't only print it.
- Don't change the tests, `conftest.py`, `orchard.csv`, `seed_data.py` or `report.py`.

## Tips

- Try each step in `explore.ipynb` first, on the real harvests, then move the lines that work into `orchard_models.py`.
- `PolynomialFeatures(degree=2, include_bias=False)` leaves out the column of 1s that a linear regression's intercept already covers.
- `compare_models` loops over the dictionary with `.items()`, and builds a new dictionary of scores as it goes.
- After you change `orchard_models.py`, restart the notebook's kernel before you try it there, so the notebook imports your newest code.

## Hand it in

When it works, push your solution to a public repo of your own on GitHub, then paste that repo's address into **Submit your exercise** on the lesson page at [ask-codey.com](https://www.ask-codey.com). That's what completes the lesson. The lesson page shows the Git commands, step by step.
