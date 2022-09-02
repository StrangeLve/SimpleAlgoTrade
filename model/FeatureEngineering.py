from sklearn.base import BaseEstimator, TransformerMixin
from talib import RSI, BBANDS, MACD, MA, PPO
import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one, _name_estimators
import pandas as pd
from scipy import sparse
import warnings

warnings.filterwarnings('ignore')

"""
All Classes are designed to work on univariate timeseries
"""


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names]


class CalcReturn(BaseEstimator, TransformerMixin):

    def fit(self, x=None, y=None):
        return self

    def transform(self, x):
        return x.pct_change()


class CalcShift(BaseEstimator, TransformerMixin):
    def __init__(self, shift_val):
        self.shift_val = shift_val
        shift_direction = ""
        if shift_val < 0:
            shift_direction = "forward"
        elif shift_val > 0:
            shift_direction = "backward"
        self.name = f"returns_{shift_direction}_{abs(shift_val)}"

    def fit(self, x=None, y=None):
        return self

    def transform(self, x):
        return x.shift(self.shift_val).to_frame(self.name)


class CalcMa(BaseEstimator, TransformerMixin):

    def __init__(self, period):
        self.period = period
        self.name = f"MA_{self.period}"

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return MA(x, self.period).to_frame(self.name)


class CalcRsi(BaseEstimator, TransformerMixin):

    def __init__(self, period):
        self.period = period
        self.name = f"RSI_{self.period}"

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return RSI(x, self.period).to_frame(self.name)


class CalcBB(BaseEstimator, TransformerMixin):

    def __init__(self, period, type_="low"):
        self.period = period
        self.type_ = type_
        self.name = f"BB_{self.type_}_{self.period}"

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        high, mid, low = BBANDS(x, self.period)
        if self.type_ == "low":
            return low.to_frame(self.name)
        elif self.type_ == "high":
            return high.to_frame(self.name)
        elif self.type_ == "mid":
            return mid.to_frame(self.name)
        else:
            raise NameError


class CalcPpo(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.name = "PPO"

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return PPO(x).to_frame(self.name)


class CalcMacd(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.name = "MACD"

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return MACD(x)[0].to_frame(self.name)


class CalcQuantile(BaseEstimator, TransformerMixin):

    def __init__(self, period, quantile_val):
        self.period = period
        self.quantile_val = quantile_val
        self.name = f"Quantile_{self.period}_{self.quantile_val}"

    def fit(self, x, y=None):
        return self

    @staticmethod
    def compute_quantile(x, quantile_val):
        return x.quantile(quantile_val)

    def transform(self, x):
        return x.rolling(self.period).apply(lambda x: CalcQuantile.compute_quantile(x, self.quantile_val)).to_frame(self.name)


class CalcStd(BaseEstimator, TransformerMixin):

    def __init__(self, period):
        self.period = period
        self.name = f"Std_{self.period}"

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.rolling(self.period).apply(lambda x: np.std(x)).to_frame(self.name)


class PandasFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs


def make_union(*transformers, **kwargs):
    n_jobs = kwargs.pop('n_jobs', None)
    verbose = kwargs.pop('verbose', False)
    if kwargs:
        # We do not currently support `transformer_weights` as we may want to
        # change its type spec in make_union
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return PandasFeatureUnion(
        _name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)

