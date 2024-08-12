from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, PowerTransformer, SplineTransformer
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
    
class GlobalRobustScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        flattened_X = X.flatten()
        self.global_median_ = np.median(flattened_X)
        self.global_iqr_ = np.percentile(flattened_X, 75) - np.percentile(flattened_X, 25)
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return (X - self.global_median_) / self.global_iqr_

    def inverse_transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return (X * self.global_iqr_) + self.global_median_

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else np.arange(X.shape[1])


class GlobalLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=1e-1):
        self.epsilon = epsilon
        self.global_min_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Store feature names before converting to NumPy array
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()

        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Flatten the input to compute global min
        X_flattened = X.flatten()

        # Compute the global min to ensure positive values for log transformation
        self.global_min_ = X_flattened.min()

        # Set the number of features for get_feature_names_out
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Flatten the input
        X_flattened = X.flatten()

        # Add epsilon to ensure all values are positive and apply global log transformation
        X_shifted = X_flattened + abs(self.global_min_) + self.epsilon
        X_log = np.log1p(X_shifted)

        # Reshape back to original shape
        X_log_reshaped = X_log.reshape(X.shape)

        return X_log_reshaped

    def inverse_transform(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Flatten the input
        X_flattened = X.flatten()

        # Inverse global log transformation and remove epsilon
        X_inv_log = np.expm1(X_flattened) - abs(self.global_min_) - self.epsilon

        # Reshape back to original shape
        X_inv_log_reshaped = X_inv_log.reshape(X.shape)

        return X_inv_log_reshaped

    def fit_transform(self, X, y=None):
        # Combine fit and transform
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        # Return the input features as the output feature names
        if input_features is None and self.feature_names_in_ is not None:
            return self.feature_names_in_
        elif input_features is not None:
            return np.asarray(input_features)
        else:
            return np.array([f"feature_{i}" for i in range(self.n_features_in_)])

    def set_output(self, transform="default"):
        # Ensure compatibility with scikit-learn 1.2+
        return self
    

class GlobalLogStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.global_min_ = None
        self.global_mean_ = None
        self.global_std_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Store feature names before converting to NumPy array
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()

        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Flatten the input to compute global statistics
        X_flattened = X.flatten()

        # Compute the global min to ensure positive values for log transformation
        self.global_min_ = X_flattened.min()

        # Apply log transformation
        X_shifted = X_flattened + abs(self.global_min_) + self.epsilon
        X_log = np.log1p(X_shifted)

        # Compute global mean and standard deviation after log transformation
        self.global_mean_ = np.mean(X_log)
        self.global_std_ = np.std(X_log)

        # Set the number of features for get_feature_names_out
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Flatten the input
        X_flattened = X.flatten()

        # Add epsilon and apply global log transformation
        X_shifted = X_flattened + abs(self.global_min_) + self.epsilon
        X_log = np.log1p(X_shifted)

        # Apply global standardization
        X_scaled = (X_log - self.global_mean_) / self.global_std_

        # Reshape back to original shape
        X_scaled_reshaped = X_scaled.reshape(X.shape)

        return X_scaled_reshaped

    def inverse_transform(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Flatten the input
        X_flattened = X.flatten()

        # Inverse global standardization
        X_inv_standard = (X_flattened * self.global_std_) + self.global_mean_

        # Inverse global log transformation and remove epsilon
        X_inv_log = np.expm1(X_inv_standard) - abs(self.global_min_) - self.epsilon

        # Reshape back to original shape
        X_inv_log_reshaped = X_inv_log.reshape(X.shape)

        return X_inv_log_reshaped

    def fit_transform(self, X, y=None):
        # Combine fit and transform
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        # Return the input features as the output feature names
        if input_features is None and self.feature_names_in_ is not None:
            return self.feature_names_in_
        elif input_features is not None:
            return np.asarray(input_features)
        else:
            return np.array([f"feature_{i}" for i in range(self.n_features_in_)])

    def set_output(self, transform="default"):
        # Ensure compatibility with scikit-learn 1.2+
        return self


class GlobalMinMaxScalerWithClipping(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1), clip_percentile=(0, 99)):
        self.feature_range = feature_range
        self.clip_percentile = clip_percentile
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.scale_ = None
        self.min_ = None
        self.shift_ = None

    def fit(self, X, y=None):
        # Compute the min and max of the entire dataset after clipping
        lower_percentile = np.percentile(X, self.clip_percentile[0])
        upper_percentile = np.percentile(X, self.clip_percentile[1])
        
        X_clipped = np.clip(X, lower_percentile, upper_percentile)
        
        self.data_min_ = X_clipped.min()
        self.data_max_ = X_clipped.max()
        self.data_range_ = self.data_max_ - self.data_min_
        
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_
        
        # Set the number of features for get_feature_names_out
        self.n_features_in_ = X.shape[1]
        X_scaled = X * self.scale_ + self.min_
        self.shift = X_scaled.mean()
        print(self.shift, "shift")
        
        return self

    def transform(self, X):
        # Apply the scaling to the entire dataset
        X_scaled = X * self.scale_ + self.min_
        X_shifted = X_scaled - self.shift
        return X_shifted

    def fit_transform(self, X, y=None):
        # Combine fit and transform
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        # Return the input features as the output feature names
        if input_features is None:
            return np.array([f"feature_{i}" for i in range(self.n_features_in_)])
        else:
            return np.asarray(input_features)

    def set_output(self, transform="default"):
        # Ensure compatibility with scikit-learn 1.2+
        return self



class GlobalMinMaxScalerWithGlobalShift(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1), clip_percentile=(0, 99)):
        self.feature_range = feature_range
        self.clip_percentile = clip_percentile
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.scale_ = None
        self.min_ = None
        self.shift_ = None
        self.lower_percentile_ = None
        self.upper_percentile_ = None

    def fit(self, X, y=None):
        
        X = np.asarray(X)
        # Flatten the data to find the global min and max after clipping
        X_flat = X.flatten()
        
        self.lower_percentile_ = np.percentile(X_flat, self.clip_percentile[0])
        self.upper_percentile_ = np.percentile(X_flat, self.clip_percentile[1])
        X_clipped = np.clip(X_flat, self.lower_percentile_, self.upper_percentile_)
        
        self.data_min_ = X_clipped.min()
        self.data_max_ = X_clipped.max()
        self.data_range_ = self.data_max_ - self.data_min_

        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_

        # Calculate the mean of the scaled data globally
        X_scaled_flat = X_flat * self.scale_ + self.min_
        self.shift_ = X_scaled_flat.mean()
        
        # Print the global shift value
        print("Global shift (mean of scaled data):", self.shift_)

        # Set the number of features for get_feature_names_out
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        # Apply the global Min-Max scaling
        X_scaled = X * self.scale_ + self.min_
        # Shift the data to have a global mean of 0
        X_shifted = X_scaled - self.shift_
        return X_shifted

    def fit_transform(self, X, y=None):
        # Combine fit and transform
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        # Return the input features as the output feature names
        if input_features is None:
            return np.array([f"feature_{i}" for i in range(self.n_features_in_)])
        else:
            return np.asarray(input_features)

    def set_output(self, transform="default"):
        # Ensure compatibility with scikit-learn 1.2+
        return self


class LogQuantileTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=1000, output_distribution='normal', epsilon=1e-6):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.epsilon = epsilon
        self.quantile_transformer = QuantileTransformer(n_quantiles=self.n_quantiles, output_distribution=self.output_distribution)
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Store feature names before converting to NumPy array
        self.feature_names_in_ = X.columns.to_numpy()

        # Ensure X is a NumPy array
        X = X.to_numpy()

        # Add epsilon to ensure all values are positive
        X_shifted = X + self.epsilon

        # Apply logarithmic transformation to all values
        X_log = np.log1p(X_shifted)

        # Fit the quantile transformer
        self.quantile_transformer.fit(X_log)

        # Set the number of features for get_feature_names_out
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        # Ensure X is a NumPy array
        X = X.to_numpy()

        # Add epsilon to ensure all values are positive
        X_shifted = X + self.epsilon

        # Apply logarithmic transformation to all values
        X_log = np.log1p(X_shifted)

        # Apply the quantile transformation
        X_quantile = self.quantile_transformer.transform(X_log)

        return X_quantile

    def inverse_transform(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Inverse quantile transformation
        X_inv_quantile = self.quantile_transformer.inverse_transform(X)

        # Inverse logarithmic transformation and remove epsilon
        X_inv_log = np.expm1(X_inv_quantile) - self.epsilon

        return X_inv_log

    def fit_transform(self, X, y=None):
        # Combine fit and transform
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        # Return the input features as the output feature names
        if input_features is None:
            return self.feature_names_in_
        else:
            return np.asarray(input_features)

    def set_output(self, transform="default"):
        # Ensure compatibility with scikit-learn 1.2+
        return self


class LogQuantileStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=1000, output_distribution='normal'):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.quantile_transformer = QuantileTransformer(n_quantiles=self.n_quantiles, output_distribution=self.output_distribution)
        self.standard_scaler = StandardScaler()
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Store feature names before converting to NumPy array
        self.feature_names_in_ = X.columns.to_numpy()

        # Ensure X is a NumPy array
        X = X.to_numpy()

        # Apply logarithmic transformation to all values
        X_log = np.log1p(X)

        # Fit the quantile transformer
        X_quantile = self.quantile_transformer.fit_transform(X_log)

        # Fit the standard scaler
        self.standard_scaler.fit(X_quantile)

        # Set the number of features for get_feature_names_out
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        # Ensure X is a NumPy array
        X = X.to_numpy()

        # Apply logarithmic transformation to all values
        X_log = np.log1p(X)

        # Apply the quantile transformation
        X_quantile = self.quantile_transformer.transform(X_log)
        
        # Apply the standard scaling
        X_scaled = self.standard_scaler.transform(X_quantile)

        return X_scaled

    def inverse_transform(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Inverse standard scaling
        X_inv_standard = self.standard_scaler.inverse_transform(X)

        # Inverse quantile transformation
        X_inv_quantile = self.quantile_transformer.inverse_transform(X_inv_standard)
        
        # Inverse logarithmic transformation
        X_inv_log = np.expm1(X_inv_quantile)

        return X_inv_log

    def fit_transform(self, X, y=None):
        # Combine fit and transform
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        # Return the input features as the output feature names
        if input_features is None:
            return self.feature_names_in_
        else:
            return np.asarray(input_features)

    def set_output(self, transform="default"):
        # Ensure compatibility with scikit-learn 1.2+
        return self



class GlobalLogQuantileStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=1000, output_distribution='normal', epsilon=1e-9):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.epsilon = epsilon
        self.quantile_transformer = QuantileTransformer(n_quantiles=self.n_quantiles, output_distribution=self.output_distribution)
        self.standard_scaler = StandardScaler()
        self.n_features_in_ = None

    def fit(self, X, y=None):
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Flatten the data to treat it globally
        X_flat = X.flatten()

        # Apply logarithmic transformation to all values
        X_log = np.log1p(X_flat) # + self.epsilon)

        # Fit the quantile transformer
        X_quantile = self.quantile_transformer.fit_transform(X_log.reshape(-1, 1)).flatten()

        # Fit the standard scaler
        self.standard_scaler.fit(X_quantile.reshape(-1, 1))

        # Set the number of features for get_feature_names_out
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Flatten the data to treat it globally
        X_flat = X.flatten()

        # Apply logarithmic transformation to all values
        X_log = np.log1p(X_flat)# + self.epsilon)

        # Apply the quantile transformation
        X_quantile = self.quantile_transformer.transform(X_log.reshape(-1, 1)).flatten()

        # Apply the standard scaling
        X_scaled = self.standard_scaler.transform(X_quantile.reshape(-1, 1)).flatten()

        # Reshape back to the original shape
        X_scaled = X_scaled.reshape(X.shape)

        return X_scaled

    def fit_transform(self, X, y=None):
        # Combine fit and transform
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        # Return the input features as the output feature names
        if input_features is None:
            return np.array([f"feature_{i}" for i in range(self.n_features_in_)])
        else:
            return np.asarray(input_features)

    def set_output(self, transform="default"):
        # Ensure compatibility with scikit-learn 1.2+
        return self


class EpsilonPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='box-cox', epsilon=1e-6):
        self.method = method
        self.epsilon = epsilon
        self.power_transformer = PowerTransformer(method=self.method)
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Store feature names before converting to NumPy array
        self.feature_names_in_ = X.columns.to_numpy()

        # Ensure X is a NumPy array
        X = X.to_numpy()

        # Add epsilon to ensure all values are positive (for box-cox)
        X_shifted = X + self.epsilon

        # Fit the power transformer
        self.power_transformer.fit(X_shifted)

        # Set the number of features for get_feature_names_out
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        # Ensure X is a NumPy array
        X = X.to_numpy()

        # Add epsilon to ensure all values are positive (for box-cox)
        X_shifted = X + self.epsilon

        # Apply the power transformation
        X_power = self.power_transformer.transform(X_shifted)

        return X_power

    def inverse_transform(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Inverse power transformation
        X_inv_power = self.power_transformer.inverse_transform(X)

        # Remove epsilon
        X_inv_log = X_inv_power - self.epsilon

        return X_inv_log

    def fit_transform(self, X, y=None):
        # Combine fit and transform
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        # Return the input features as the output feature names
        if input_features is None:
            return self.feature_names_in_
        else:
            return np.asarray(input_features)

    def set_output(self, transform="default"):
        # Ensure compatibility with scikit-learn 1.2+
        return self


class PassthroughScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.n_features_in_ = None

    def fit(self, X, y=None):
        # Set the number of features based on the input
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        # Passthrough transformation
        return X

    def fit_transform(self, X, y=None):
        # Combine fit and transform
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        # Return the input features as the output feature names
        if input_features is None:
            return np.array([f"feature_{i}" for i in range(self.n_features_in_)])
        else:
            return np.asarray(input_features)

    def set_output(self, transform="default"):
        # Ensure compatibility with scikit-learn 1.2+
        return self