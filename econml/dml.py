# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Double ML.

"Double Machine Learning" is an algorithm that applies arbitrary machine learning methods
to fit the treatment and response, then uses a linear model to predict the response residuals
from the treatment residuals.

"""

import numpy as np
import copy
from .utilities import shape, reshape, ndim, hstack, cross_product, transpose
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn.base import clone, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from statsmodels.regression.linear_model import OLS
from .cate_estimator import LinearCateEstimator
from .inference import StatsModelsInference


class _RLearner(LinearCateEstimator):
    """
    Base class for orthogonal learners.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features and controls. Must implement
        `fit` and `predict` methods.  Unlike sklearn estimators both methods must
        take an extra second argument (the controls).

    model_t: estimator
        The estimator for fitting the treatment to the features and controls. Must implement
        `fit` and `predict` methods.  Unlike sklearn estimators both methods must
        take an extra second argument (the controls).

    model_final: estimator for fitting the response residuals to the features and treatment residuals
        Must implement `fit` and `predict` methods. Unlike sklearn estimators the fit methods must
        take an extra second argument (the treatment residuals).  Predict, on the other hand,
        should just take the features and return the constant marginal effect.

    n_splits: int
        The number of splits to use when fitting the first-stage models.

    random_state: int, RandomState instance or None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, model_y, model_t, model_final,
                 discrete_treatment, n_splits, random_state):
        self._models_y = [clone(model_y, safe=False) for _ in range(n_splits)]
        self._models_t = [clone(model_t, safe=False) for _ in range(n_splits)]
        self._model_final = clone(model_final, safe=False)
        self._n_splits = n_splits
        self._discrete_treatment = discrete_treatment
        self._random_state = check_random_state(random_state)
        super().__init__()

    def fit(self, Y, T, X=None, W=None, sample_weight=None, inference=None):
        if X is None:
            X = np.ones((shape(Y)[0], 1))
        if W is None:
            W = np.empty((shape(Y)[0], 0))
        assert shape(Y)[0] == shape(T)[0] == shape(X)[0] == shape(W)[0]

        Y_res, T_res = self.fit_nuisances(Y, T, X, W, sample_weight=sample_weight)

        self.fit_final(X, Y_res, T_res, sample_weight=sample_weight)

        return super().fit(Y, T, X=X, W=None, sample_weight=sample_weight, inference=inference)

    def fit_nuisances(self, Y, T, X, W, sample_weight=None):
        if self._discrete_treatment:
            folds = StratifiedKFold(self._n_splits, shuffle=True,
                                    random_state=self._random_state).split(np.empty_like(X), T)
            self._label_encoder = LabelEncoder()
            self._one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
            T = self._label_encoder.fit_transform(T)
            T_out = self._one_hot_encoder.fit_transform(reshape(T, (-1, 1)))
            T_out = T_out[:, 1:]  # drop first column since all columns sum to one
        else:
            folds = KFold(self._n_splits, shuffle=True, random_state=self._random_state).split(X)
            T_out = T

        Y_res = np.zeros(shape(Y))
        T_res = np.zeros(shape(T_out))
        for idx, (train_idxs, test_idxs) in enumerate(folds):
            Y_train, Y_test = Y[train_idxs], Y[test_idxs]
            T_train, T_test = T[train_idxs], T_out[test_idxs]
            X_train, X_test = X[train_idxs], X[test_idxs]
            W_train, W_test = W[train_idxs], W[test_idxs]
            # TODO: If T is a vector rather than a 2-D array, then the model's fit must accept a vector...
            #       Do we want to reshape to an nx1, or just trust the user's choice of input?
            #       (Likewise for Y below)
            if sample_weight is not None:
                self._models_t[idx].fit(X_train, W_train, T_train, sample_weight=sample_weight[train_idxs])
            else:
                self._models_t[idx].fit(X_train, W_train, T_train)
            if self._discrete_treatment:
                T_res[test_idxs] = T_test - self._models_t[idx].predict(X_test, W_test)[:, 1:]
            else:
                T_res[test_idxs] = T_test - self._models_t[idx].predict(X_test, W_test)
            if sample_weight is not None:
                self._models_y[idx].fit(X_train, W_train, Y_train, sample_weight=sample_weight[train_idxs])
            else:
                self._models_y[idx].fit(X_train, W_train, Y_train)
            Y_res[test_idxs] = Y_test - self._models_y[idx].predict(X_test, W_test)
        return Y_res, T_res

    def fit_final(self, X, Y_res, T_res, sample_weight=None):
        if sample_weight is not None:
            self._model_final.fit(X, T_res, Y_res, sample_weight=sample_weight)
        else:
            self._model_final.fit(X, T_res, Y_res)

    def const_marginal_effect(self, X=None):
        """
        Calculate the constant marginal CATE θ(·).

        The marginal effect is conditional on a vector of
        features on a set of m test samples {Xᵢ}.

        Parameters
        ----------
        X: optional (m × dₓ) matrix
            Features for each sample.
            If X is None, it will be treated as a column of ones with a single row

        Returns
        -------
        theta: (m × d_y × dₜ) matrix
            Constant marginal CATE of each treatment on each outcome for each sample.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        if X is None:
            X = np.ones((1, 1))
        return self._model_final.predict(X)

    def const_marginal_effect_interval(self, X=None, *, alpha=0.1):
        if X is None:
            X = np.ones((1, 1))
        return super().const_marginal_effect_interval(X, alpha=alpha)

    # if T is scalar, expand to match number of rows of X
    # if T is a discrete treatment, transform it to one-hot representation
    def _expand_treatment(self, T, X):
        if ndim(T) == 0:
            T = np.repeat(T, 1 if X is None else shape(X)[0])
        if self._discrete_treatment:
            T = self._one_hot_encoder.transform(reshape(self._label_encoder.transform(T), (-1, 1)))[:, 1:]
        return T

    # need to override super's effect to handle discrete treatments
    # TODO: should this logic be moved up to the LinearCateEstimator class and
    #       removed from here and from the OrthoForest implementation?
    def effect(self, X, T0=0, T1=1):
        return super().effect(X, self._expand_treatment(T0, X), self._expand_treatment(T1, X))

    def effect_interval(self, X, *, T0=0, T1=1, alpha=0.1):
        # for effect_interval, perform the same discrete treatment transformation as done in effect
        return super().effect_interval(X,
                                       T0=self._expand_treatment(T0, X), T1=self._expand_treatment(T1, X),
                                       alpha=alpha)

    def score(self, Y, T, X=None, W=None):
        if self._discrete_treatment:
            T = self._one_hot_encoder.transform(reshape(self._label_encoder.transform(T), (-1, 1)))[:, 1:]
        if T.ndim == 1:
            T = reshape(T, (-1, 1))
        if Y.ndim == 1:
            Y = reshape(Y, (-1, 1))
        if X is None:
            X = np.ones((shape(Y)[0], 1))
        if W is None:
            W = np.empty((shape(Y)[0], 0))
        Y_test_pred = np.zeros(shape(Y) + (self._n_splits,))
        T_test_pred = np.zeros(shape(T) + (self._n_splits,))
        for ind in range(self._n_splits):
            if self._discrete_treatment:
                T_test_pred[:, :, ind] = reshape(self._models_t[ind].predict(X, W)[:, 1:], shape(T))
            else:
                T_test_pred[:, :, ind] = reshape(self._models_t[ind].predict(X, W), shape(T))
            Y_test_pred[:, :, ind] = reshape(self._models_y[ind].predict(X, W), shape(Y))
        Y_test_pred = Y_test_pred.mean(axis=2)
        T_test_pred = T_test_pred.mean(axis=2)
        Y_test_res = Y - Y_test_pred
        T_test_res = T - T_test_pred
        effects = reshape(self._model_final.predict(X), (-1, shape(Y)[1], shape(T)[1]))
        Y_test_res_pred = reshape(np.einsum('ijk,ik->ij', effects, T_test_res), shape(Y))
        mse = ((Y_test_res - Y_test_res_pred)**2).mean()
        return mse


class DMLCateEstimator(_RLearner):
    """
    The base class for parametric Double ML estimators.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when sparseLinear is `True`.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when sparseLinear is `True`.

    model_final: estimator
        The estimator for fitting the response residuals to the treatment residuals. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    featurizer: transformer
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    sparseLinear: bool
        Whether to use sparse linear model assumptions

    discrete_treatment: bool
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int
        The number of splits to use when fitting the first-stage models.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self,
                 model_y, model_t, model_final,
                 featurizer,
                 sparseLinear=False,
                 discrete_treatment=False,
                 n_splits=2,
                 random_state=None):

        class FirstStageWrapper:
            def __init__(self, model, is_Y):
                self._model = clone(model, safe=False)
                self._featurizer = clone(featurizer, safe=False)
                self._is_Y = is_Y

            def _combine(self, X, W):
                if self._is_Y and sparseLinear:
                    F = self._featurizer.fit_transform(X)
                    XW = hstack([X, W])
                    return cross_product(XW, hstack([np.ones((shape(XW)[0], 1)), F, W]))
                else:
                    return hstack([X, W])

            def fit(self, X, W, Target, sample_weight=None):
                if sample_weight is not None:
                    self._model.fit(self._combine(X, W), Target, sample_weight=sample_weight)
                else:
                    self._model.fit(self._combine(X, W), Target)

            def predict(self, X, W):
                if (not self._is_Y) and discrete_treatment:
                    return self._model.predict_proba(self._combine(X, W))
                else:
                    return self._model.predict(self._combine(X, W))

        class FinalWrapper:
            def __init__(self):
                self._model = clone(model_final, safe=False)
                self._featurizer = clone(featurizer, safe=False)

            def fit(self, X, T_res, Y_res, sample_weight=None):
                # Track training dimensions to see if Y or T is a vector instead of a 2-dimensional array
                self._d_t = shape(T_res)[1:]
                self._d_y = shape(Y_res)[1:]
                if sample_weight is not None:
                    self._model.fit(self._combine(X, T_res),
                                    Y_res, sample_weight=sample_weight)
                else:
                    self._model.fit(self._combine(X, T_res), Y_res)

            def _combine(self, X, T):
                return cross_product(self._featurizer.fit_transform(X), T)

            # combine X with each marginal treatment
            def _transform(self, X):
                # create an identity matrix of size d_t (or just a 1-element array if T was a vector)
                # the nth row will allow us to compute the marginal effect of the nth component of treatment
                d_x = shape(X)[0]
                d_t = self._d_t[0] if self._d_t else 1
                eye = np.eye(d_t)
                # tile T and repeat X along axis 0 (so that the duplicated rows of X remain consecutive)
                T = np.tile(eye, (d_x, 1))
                Xs = np.repeat(X, d_t, axis=0)
                return self._combine(Xs, T)

            def _untransform(self, A):
                A = reshape(A, (-1,) + self._d_t + self._d_y)
                if self._d_t and self._d_y:
                    return transpose(A, (0, 2, 1))  # need to return as m by d_y by d_t matrix
                else:
                    return A

            def predict(self, X):
                XT = self._transform(X)
                return self._untransform(self._model.predict(XT))

            @property
            def coef_(self):
                # TODO: handle case where final model doesn't directly expose coef_?
                return reshape(self._model.coef_, self._d_y + self._d_t + (-1,))

        super().__init__(model_y=FirstStageWrapper(model_y, is_Y=True),
                         model_t=FirstStageWrapper(model_t, is_Y=False),
                         model_final=FinalWrapper(),
                         discrete_treatment=discrete_treatment,
                         n_splits=n_splits,
                         random_state=random_state)

    @property
    def coef_(self):
        """
        Get the final model's coefficients.

        Note that this relies on the final model having a `coef_` property of its own.
        Most sklearn linear models support this, but there are cases that don't
        (e.g. a `Pipeline` or `GridSearchCV` which wraps a linear model)
        """
        return self._model_final.coef_


class LinearDMLCateEstimator(DMLCateEstimator):
    """
    The Double ML Estimator with a low-dimensional linear final stage implemented as a statsmodel regression.

    Note that only a single outcome is supported (because of the dependency on statsmodel's weighted least squares)

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods.

    featurizer: transformer, optional (default is `PolynomialFeatures(degree=1, include_bias=True)`)
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    discrete_treatment: bool, optional (default is False)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, optional (default is 2)
        The number of splits to use when fitting the first-stage models.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    inference: string, `Inference` instance, or None
        Method for performing inference.  This estimator supports 'bootstrap'
        (or an instance of `BootstrapInference`) and 'statsmodels' (or an instance of 'StatsModelsInference`)
    """

    def _get_inference_options(self):
        # add statsmodels to parent's options
        options = super()._get_inference_options()
        options.update(statsmodels=StatsModelsInference)
        return options

    class StatsModelsWrapper:
        def __init__(self):
            self.fit_args = {}

        def fit(self, X, y, sample_weight=None):
            assert ndim(y) == 1 or (ndim(y) == 2 and shape(y)[1] == 1)
            y = reshape(y, (-1,))
            if sample_weight is not None:
                ols = OLS(y, X, weights=sample_weight)
            else:
                ols = OLS(y, X)
            self.results = ols.fit(**self.fit_args)
            return self

        def predict(self, X):
            return self.results.predict(X)

        def predict_interval(self, X, alpha):
            # NOTE: we use `obs = False` to get a confidence, rather than prediction, interval
            preds = self.results.get_prediction(X).conf_int(alpha=alpha, obs=False)
            # statsmodels uses the last dimension instead of the first to store the confidence intervals,
            # so we need to transpose the result
            return transpose(preds)

    def __init__(self,
                 model_y=LassoCV(), model_t=LassoCV(),
                 featurizer=PolynomialFeatures(degree=1, include_bias=True),
                 discrete_treatment=False,
                 n_splits=2,
                 random_state=None):
        super().__init__(model_y=model_y,
                         model_t=model_t,
                         model_final=self.StatsModelsWrapper(),
                         featurizer=featurizer,
                         sparseLinear=True,
                         discrete_treatment=discrete_treatment,
                         n_splits=n_splits,
                         random_state=random_state)

    @property
    def coef_(self):
        return self._model_final._model.results.params


class SparseLinearDMLCateEstimator(DMLCateEstimator):
    """
    A specialized version of the Double ML estimator for the sparse linear case.

    Specifically, this estimator can be used when the controls are high-dimensional,
    the treatment and response are linear functions of the features and controls,
    and the coefficients of the nuisance functions are sparse.

    Parameters
    ----------
    linear_model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    linear_model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    model_final: estimator, optional (default is `LinearRegression(fit_intercept=False)`)
        The estimator for fitting the response residuals to the treatment residuals. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    featurizer: transformer, optional (default is `PolynomialFeatures(degree=1, include_bias=True)`)
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    discrete_treatment: bool, optional (default is False)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, optional (default is 2)
        The number of splits to use when fitting the first-stage models.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self,
                 linear_model_y=LassoCV(), linear_model_t=LassoCV(), model_final=LinearRegression(fit_intercept=False),
                 featurizer=PolynomialFeatures(degree=1, include_bias=True),
                 discrete_treatment=False,
                 n_splits=2,
                 random_state=None):
        super().__init__(model_y=linear_model_y,
                         model_t=linear_model_t,
                         model_final=model_final,
                         featurizer=featurizer,
                         sparseLinear=True,
                         discrete_treatment=discrete_treatment,
                         n_splits=n_splits,
                         random_state=random_state)


class KernelDMLCateEstimator(LinearDMLCateEstimator):
    """
    A specialized version of the linear Double ML Estimator that uses random fourier features.

    Parameters
    ----------
    model_y: estimator, optional (default is `LassoCV()`)
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.

    model_t: estimator, optional (default is `LassoCV()`)
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods.

    dim: int, optional (default is 20)
        The number of random Fourier features to generate

    bw: float, optional (default is 1.0)
        The bandwidth of the Gaussian used to generate features

    n_splits: int, optional (default is 2)
        The number of splits to use when fitting the first-stage models.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, model_y=LassoCV(), model_t=LassoCV(),
                 dim=20, bw=1.0, n_splits=2, random_state=None):
        class RandomFeatures(TransformerMixin):
            def fit(innerself, X):
                innerself.omegas = self._random_state.normal(0, 1 / bw, size=(shape(X)[1], dim))
                innerself.biases = self._random_state.uniform(0, 2 * np.pi, size=(1, dim))
                return innerself

            def transform(innerself, X):
                return np.sqrt(2 / dim) * np.cos(np.matmul(X, innerself.omegas) + innerself.biases)

        super().__init__(model_y=model_y, model_t=model_t,
                         featurizer=RandomFeatures(), n_splits=n_splits, random_state=random_state)
