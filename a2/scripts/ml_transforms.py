# scripts/ml_transforms.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NormalizeCategoricals(BaseEstimator, TransformerMixin):
    """
    For selected columns: strip whitespace, map {'', ' ', 'None','none','NULL','null','NaN','nan'} -> np.nan.
    Leaves other columns unchanged.
    """
    def __init__(self, columns):
        self.columns = columns
        self._bad = {"", " ", "NONE", "None", "none", "NULL", "null", "NaN", "nan", "N/A", "NA"}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import numpy as np, pandas as pd
        X = X.copy()
        for c in self.columns:
            if c in X.columns:
                s = X[c].astype("object")
                s = s.map(lambda v: None if (v is None) else str(v).strip())
                s = s.map(lambda v: np.nan if (v is None or v in self._bad) else v)
                X[c] = s
        return X
    
class LoanTypeBinarizer(BaseEstimator, TransformerMixin):
    """
    Splits a single string column with '|' separators into multi-hot features.
    Handles None/NaN/empty, strips whitespace, and fixes the feature space at fit().
    """
    def __init__(self, sep="|"):
        self.sep = sep
        self.classes_ = None

    def fit(self, X, y=None):
        # X is shape (n_samples, 1) when coming from ColumnTransformer
        col = pd.Series(X.ravel(), dtype="object")
        tokens = set()
        for v in col.dropna():
            for t in str(v).split(self.sep):
                t = t.strip()
                if t:
                    tokens.add(t)
        self.classes_ = sorted(tokens)
        return self

    def transform(self, X):
        import numpy as np
        n = X.shape[0]
        if not self.classes_:
            return np.zeros((n, 0), dtype=float)
        idx = {t:i for i,t in enumerate(self.classes_)}
        out = np.zeros((n, len(self.classes_)), dtype=float)
        col = pd.Series(X.ravel(), dtype="object")
        for r, v in enumerate(col):
            if pd.isna(v):
                continue
            for t in str(v).split(self.sep):
                key = t.strip()
                if key:
                    j = idx.get(key)
                    if j is not None:
                        out[r, j] = 1.0
        return out

    def get_feature_names_out(self, input_features=None):
        # Compatible with sklearn's ColumnTransformer
        base = (input_features or ["Type_of_Loan"])[0]
        return np.array([f"{base}__{t}" for t in self.classes_], dtype=object)
