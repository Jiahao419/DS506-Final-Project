# tests/test_environment.py

def test_imports():
    """Basic sanity check: required libraries can be imported."""
    import pandas as pd  # noqa: F401
    import numpy as np   # noqa: F401
    import pyarrow       # noqa: F401
    import matplotlib    # noqa: F401
    import seaborn       # noqa: F401
    import sklearn       # noqa: F401
    import joblib        # noqa: F401
    import meteostat     # noqa: F401
