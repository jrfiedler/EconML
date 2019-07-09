# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import numpy as np
from .bootstrap import BootstrapEstimator

"""Options for performing inference in estimators."""


class Inference:
    def fit(self, estimator, *args, **kwargs):
        pass


class BootstrapInference(Inference):
    """
    Inference instance to perform bootstrapping.

    This class can be used for inference with any CATE estimator.

    Parameters
    ----------
    n_bootstrap_samples : int, optional (default 100)
        How many draws to perform.

    n_jobs: int, optional (default -1)
        The maximum number of concurrently running jobs, as in joblib.Parallel.

    """

    def __init__(self, n_bootstrap_samples=100, n_jobs=-1):
        self._n_bootstrap_samples = n_bootstrap_samples
        self._n_jobs = n_jobs

    def fit(self, estimator, *args, **kwargs):
        est = BootstrapEstimator(estimator, self._n_bootstrap_samples, self._n_jobs, compute_means=False)
        est.fit(*args, **kwargs)
        self._est = est

    def __getattr__(self, name):
        m = getattr(self._est, name)

        def wrapped(*args, alpha=0.1, **kwargs):
            return m(*args, lower=100 * alpha / 2, upper=100 * (1 - alpha / 2), **kwargs)
        return wrapped


class StatsModelsInference(Inference):
    """
    Stores statsmodels covariance options.

    This class can be used for inference by the LinearDMLCateEstimator.

    Parameters
    ----------
    cov_type : string, optional (default 'HC1')
        The type of covariance estimation method to use.  Supported values are 'nonrobust',
        'fixed scale', 'HC0', 'HC1', 'HC2', and 'HC3'.  See the statsmodels documentation for
        further information.

    cov_kwds : optional additional keywords supported by the chosen method
        Of the supported types, only 'fixed scale' has any keywords:
        the optional keyword 'scale' which defaults to 1 if not specified.  See the statsmodels
        documentation for further information.
    """

    def __init__(self, cov_type='HC1', **cov_kwds):
        if cov_kwds and cov_type != 'fixed scale':
            raise ValueError("Keyword arguments are only supported by the 'fixed scale' cov_type")
        if cov_type not in ['nonrobust', 'fixed scale', 'HC0', 'HC1', 'HC2', 'HC3']:
            raise ValueError("Unsupported cov_type; "
                             "must be one of 'nonrobust', "
                             "'fixed scale', 'HC0', 'HC1', 'HC2', or 'HC3'")
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds

    def fit(self, estimator, *args, **kwargs):
        estimator._model_final.fit_args = {'cov_type': self.cov_type, 'cov_kwds': self.cov_kwds}
        self._est = estimator

    def effect_interval(self, X, *, T0=0, T1=1, alpha=0.1):
        dT = T1 - T0
        preds = self._est._model_final._model.predict_interval(self._est._model_final._combine(X, dT), alpha=alpha)
        return preds

    def const_marginal_effect_interval(self, X, *, alpha=0.1):
        # TODO: is there a cleaner way to achieve this?
        XT = self._est._model_final._transform(X)
        preds = self._est._model_final._model.predict_interval(XT, alpha=alpha)
        return np.stack([self._est._model_final._untransform(pred) for pred in preds])
