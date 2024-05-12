from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer
import numpy as np
import pandas as pd

class CyclicEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Placeholder for storing n_splines and periods for each column
        self.columns_info_ = {}
        self.feature_names = []

    def fit(self, X, y=None):
        # Determine n_splines and periods for each column in X
        for col in X:
            period = X[col].nunique()
            #n_splines = min(max(period // 2, 2), 6)
            n_splines = round(np.log(2*period) + 1)
            self.columns_info_[col] = (period, n_splines)
            print("column info", self.columns_info_[col])
            self.feature_names.extend([f"{col}_spline_{i}" for i in range(n_splines)])
        return self

    def transform(self, X):
        # Ensure fit has been called
        if not self.columns_info_:
            raise RuntimeError("CyclicEncoder has not been fitted, call fit before transform.")
        transformed_cols = []
        for col, (period, n_splines) in self.columns_info_.items():
            spline_transformer = SplineTransformer(
                degree=2,
                n_knots=5,
                knots=np.linspace(0, period, n_splines + 1).reshape(-1, 1),
                extrapolation="periodic",
                include_bias=True,
            )
            transformed_col = spline_transformer.fit_transform(X[[col]])
            # Assign new column names reflecting the transformation
            for i in range(transformed_col.shape[1]):
                transformed_cols.append(pd.Series(transformed_col[:, i], index=X.index, name=f"{col}_spline_{i}"))

        # Combine transformed columns into a DataFrame
        return pd.concat(transformed_cols, axis=1)
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names

