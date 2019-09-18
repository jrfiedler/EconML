# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Utility methods."""

import numpy as np
import scipy.sparse
import sparse as sp
import itertools
from operator import getitem
from collections import defaultdict, Counter
from sklearn.base import TransformerMixin
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from functools import reduce
from sklearn.utils import check_array, check_X_y
from statsmodels.regression.linear_model import WLS, OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant
import warnings

MAX_RAND_SEED = np.iinfo(np.int32).max


class IdentityFeatures(TransformerMixin):
    """Featurizer that just returns the input data."""

    def fit(self, X):
        """Fit method (does nothing, just returns self)."""
        return self

    def transform(self, X):
        """Perform the identity transform, which returns the input unmodified."""
        return X


def issparse(X):
    """Determine whether an input is sparse.

    For the purposes of this function, both `scipy.sparse` matrices and `sparse.SparseArray`
    types are considered sparse.

    Parameters
    ----------
    X : array-like
        The input to check

    Returns
    -------
    bool
        Whether the input is sparse

    """
    return scipy.sparse.issparse(X) or isinstance(X, sp.SparseArray)


def iscoo(X):
    """Determine whether an input is a `sparse.COO` array.

    Parameters
    ----------
    X : array-like
        The input to check

    Returns
    -------
    bool
        Whether the input is a `COO` array

    """
    return isinstance(X, sp.COO)


def tocoo(X):
    """
    Convert an array to a sparse COO array.

    If the input is already an `sparse.COO` object, this returns the object directly; otherwise it is converted.
    """
    if isinstance(X, sp.COO):
        return X
    elif isinstance(X, sp.DOK):
        return sp.COO(X)
    elif scipy.sparse.issparse(X):
        return sp.COO.from_scipy_sparse(X)
    else:
        return sp.COO.from_numpy(X)


def todense(X):
    """
    Convert an array to a dense numpy array.

    If the input is already a numpy array, this may create a new copy.
    """
    if scipy.sparse.issparse(X):
        return X.toarray()
    elif isinstance(X, sp.SparseArray):
        return X.todense()
    else:
        # TODO: any way to avoid creating a copy if the array was already dense?
        #       the call is necessary if the input was something like a list, though
        return np.array(X)


def size(X):
    """Return the number of elements in the array.

    Parameters
    ----------
    a : array_like
        Input data

    Returns
    -------
    int
        The number of elements of the array
    """
    return X.size if issparse(X) else np.size(X)


def shape(X):
    """Return a tuple of array dimensions."""
    return X.shape if issparse(X) else np.shape(X)


def ndim(X):
    """Return the number of array dimensions."""
    return X.ndim if issparse(X) else np.ndim(X)


def reshape(X, shape):
    """Return a new array that is a reshaped version of an input array.

    The output will be sparse iff the input is.

    Parameters
    ----------
    X : array_like
        The array to reshape

    shape : tuple of ints
        The desired shape of the output array

    Returns
    -------
    ndarray or SparseArray
        The reshaped output array
    """
    if scipy.sparse.issparse(X):
        # scipy sparse arrays don't support reshaping (even for 2D they throw not implemented errors),
        # so convert to pydata sparse first
        X = sp.COO.from_scipy_sparse(X)
        if len(shape) == 2:
            # in the 2D case, we can convert back to scipy sparse; in other cases we can't
            return X.reshape(shape).to_scipy_sparse()
    return X.reshape(shape)


def _apply(op, *XS):
    """
    Apply a function to a sequence of sparse or dense array arguments.

    If any array is sparse then all arrays are converted to COO before the function is applied;
    if all of the arrays are scipy sparse arrays, and if the result is 2D,
    the returned value will be a scipy sparse array as well
    """
    all_scipy_sparse = all(scipy.sparse.issparse(X) for X in XS)
    if any(issparse(X) for X in XS):
        XS = tuple(tocoo(X) for X in XS)
    result = op(*XS)
    if all_scipy_sparse and len(shape(result)) == 2:
        # both inputs were scipy and we can safely convert back to scipy because it's 2D
        return result.to_scipy_sparse()
    return result


def tensordot(X1, X2, axes):
    """
    Compute tensor dot product along specified axes for arrays >= 1-D.

    Parameters
    ----------
    X1, X2 : array_like, len(shape) >= 1
        Tensors to "dot"

    axes : int or (2,) array_like
        integer_like
            If an int N, sum over the last N axes of `X1` and the first N axes
            of `X2` in order. The sizes of the corresponding axes must match
        (2,) array_like
            Or, a list of axes to be summed over, first sequence applying to `X1`,
            second to `X2`. Both elements array_like must be of the same length.
    """
    def td(X1, X2):
        return sp.tensordot(X1, X2, axes) if iscoo(X1) else np.tensordot(X1, X2, axes)
    return _apply(td, X1, X2)


def cross_product(*XS):
    """
    Compute the cross product of features.

    Parameters
    ----------
    X1 : n x d1 matrix
        First matrix of n samples of d1 features
        (or an n-element vector, which will be treated as an n x 1 matrix)
    X2 : n x d2 matrix
        Second matrix of n samples of d2 features
        (or an n-element vector, which will be treated as an n x 1 matrix)
    …

    Returns
    -------
    n x (d1*d2*...) matrix
        Matrix of n samples of d1*d2*... cross product features,
        arranged in form such that each row t of X12 contains:
        [X1[t,0]*X2[t,0]*..., ..., X1[t,d1-1]*X2[t,0]*..., X1[t,0]*X2[t,1]*..., ..., X1[t,d1-1]*X2[t,1]*..., ...]

    """
    for X in XS:
        assert 2 >= ndim(X) >= 1
    n = shape(XS[0])[0]
    for X in XS:
        assert n == shape(X)[0]

    # TODO: wouldn't making X1 vary more slowly than X2 be more intuitive?
    #       (but note that changing this would necessitate changes to callers
    #       to switch the order to preserve behavior where order is important)
    def cross(XS):
        k = len(XS)
        XS = [reshape(XS[i], (n,) + (1,) * (k - i - 1) + (-1,) + (1,) * i) for i in range(k)]
        return reshape(reduce(np.multiply, XS), (n, -1))
    return _apply(cross, XS)


def stack(XS, axis=0):
    """
    Join a sequence of arrays along a new axis.

    The axis parameter specifies the index of the new axis in the dimensions of the result.
    For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last dimension.

    Parameters
    ----------
    arrays : sequence of array_like
        Each array must have the same shape

    axis : int, optional
        The axis in the result array along which the input arrays are stacked

    Returns
    -------
    ndarray or SparseArray
        The stacked array, which has one more dimension than the input arrays.
        It will be sparse if the inputs are.
    """
    def st(*XS):
        return sp.stack(XS, axis=axis) if iscoo(XS[0]) else np.stack(XS, axis=axis)
    return _apply(st, *XS)


def concatenate(XS, axis=0):
    """
    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    X1, X2, ... : sequence of array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).

    axis : int, optional
        The axis along which the arrays will be joined.  Default is 0.

    Returns
    -------
    ndarray or SparseArray
        The concatenated array. It will be sparse if the inputs are.
    """
    def conc(*XS):
        return sp.concatenate(XS, axis=axis) if iscoo(XS[0]) else np.concatenate(XS, axis=axis)
    return _apply(conc, *XS)


# note: in contrast to np.hstack this only works with arrays of dimension at least 2
def hstack(XS):
    """
    Stack arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis


    Parameters
    ----------
    XS : sequence of ndarrays
        The arrays must have the same shape along all but the second axis.

    Returns
    -------
    ndarray or SparseArray
        The array formed by stacking the given arrays. It will be sparse if the inputs are.
    """
    # Confusingly, this needs to concatenate, not stack (stack returns an array with an extra dimension)
    return concatenate(XS, 1)


def vstack(XS):
    """
    Stack arrays in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after
    1-D arrays of shape (N,) have been reshaped to (1,N).

    Parameters
    ----------
    XS : sequence of ndarrays
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Returns
    -------
    ndarray or SparseArray
        The array formed by stacking the given arrays, will be at least 2-D.   It will be sparse if the inputs are.
    """
    # Confusingly, this needs to concatenate, not stack (stack returns an array with an extra dimension)
    return concatenate(XS, 0)


def transpose(X, axes=None):
    """
    Permute the dimensions of an array.

    Parameters
    ----------
    X : array_like
        Input array.
    axes :  list of ints, optional
            By default, reverse the dimensions, otherwise permute the axes according to the values given

    Returns
    -------
    p : ndarray or SparseArray
        `X` with its axes permuted. This will be sparse if `X` is.

    """
    def t(X):
        if iscoo(X):
            return X.transpose(axes)
        else:
            return np.transpose(X, axes)
    return _apply(t, X)


def reshape_Y_T(Y, T):
    """
    Reshapes Y and T when Y.ndim = 2 and/or T.ndim = 1.

    Parameters
    ----------
    Y : array_like, shape (n, ) or (n, 1)
        Outcome for the treatment policy. Must be a vector or single-column matrix.

    T : array_like, shape (n, ) or (n, d_t)
        Treatment policy.

    Returns
    -------
    Y : array_like, shape (n, )
        Flattened outcome for the treatment policy.

    T : array_like, shape (n, 1) or (n, d_t)
        Reshaped treatment policy.

    """
    assert(len(Y) == len(T))
    assert(Y.ndim <= 2)
    if Y.ndim == 2:
        assert(Y.shape[1] == 1)
        Y = Y.flatten()
    if T.ndim == 1:
        T = T.reshape(-1, 1)
    return Y, T


def check_inputs(Y, T, X, W=None, multi_output_T=True, multi_output_Y=True):
    """
    Input validation for CATE estimators.

    Checks Y, T, X, W for consistent length, enforces X, W 2d.
    Standard input checks are only applied to all inputs,
    such as checking that an input does not have np.nan or np.inf targets.
    Converts regular Python lists to numpy arrays.

    Parameters
    ----------
    Y : array_like, shape (n, ) or (n, d_y)
        Outcome for the treatment policy.

    T : array_like, shape (n, ) or (n, d_t)
        Treatment policy.

    X : array-like, shape (n, d_x)
        Feature vector that captures heterogeneity.

    W : array-like, shape (n, d_w) or None (default=None)
        High-dimensional controls.

    multi_output_T : bool
        Whether to allow more than one treatment.

    multi_output_Y: bool
        Whether to allow more than one outcome.

    Returns
    -------
    Y : array_like, shape (n, ) or (n, d_y)
        Converted and validated Y.

    T : array_like, shape (n, ) or (n, d_t)
        Converted and validated T.

    X : array-like, shape (n, d_x)
        Converted and validated X.

    W : array-like, shape (n, d_w) or None (default=None)
        Converted and validated W.

    """
    X, T = check_X_y(X, T, multi_output=multi_output_T, y_numeric=True)
    _, Y = check_X_y(X, Y, multi_output=multi_output_Y, y_numeric=True)
    if W is not None:
        W, _ = check_X_y(W, Y)
    return Y, T, X, W


def broadcast_unit_treatments(X, d_t):
    """
    Generate `d_t` unit treatments for each row of `X`.

    Parameters
    ----------
    d_t: int
        Number of treatments
    X : array
        Features

    Returns
    -------
    X, T : (array, array)
        The updated `X` array (with each row repeated `d_t` times),
        and the generated `T` array
    """
    d_x = shape(X)[0]
    eye = np.eye(d_t)
    # tile T and repeat X along axis 0 (so that the duplicated rows of X remain consecutive)
    T = np.tile(eye, (d_x, 1))
    Xs = np.repeat(X, d_t, axis=0)
    return Xs, T


def reshape_treatmentwise_effects(A, d_t, d_y):
    """
    Given an effects matrix ordered first by treatment, transform it to be ordered by outcome.

    Parameters
    ----------
    A : array
        The array of effects, of size n*d_y*d_t
    d_t : tuple of int
        Either () if T was a vector, or a 1-tuple of the number of columns of T if it was an array
    d_y : tuple of int
        Either () if Y was a vector, or a 1-tuple of the number of columns of Y if it was an array

    Returns
    -------
    A : array (shape (m, d_y, d_t))
        The transformed array.  Note that singleton dimensions will be dropped for any inputs which
        were vectors, as in the specification of `BaseCateEstimator.marginal_effect`.
    """
    A = reshape(A, (-1,) + d_t + d_y)
    if d_t and d_y:
        return transpose(A, (0, 2, 1))  # need to return as m by d_y by d_t matrix
    else:
        return A


def einsum_sparse(subscripts, *arrs):
    """
    Evaluate the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional array operations can be represented
    in a simple fashion. This function provides a way to compute such summations.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
        Unlike `np.eisnum` elipses are not supported and the output must be explicitly included

    arrs : list of COO arrays
        These are the sparse arrays for the operation.

    Returns
    -------
    SparseArray
        The sparse array calculated based on the Einstein summation convention.
    """
    inputs, outputs = subscripts.split('->')
    inputs = inputs.split(',')
    outputInds = set(outputs)
    allInds = set.union(*[set(i) for i in inputs])

    # same number of input definitions as arrays
    assert len(inputs) == len(arrs)

    # input definitions have same number of dimensions as each array
    assert all(arr.ndim == len(input) for (arr, input) in zip(arrs, inputs))

    # all result indices are unique
    assert len(outputInds) == len(outputs)

    # all result indices must match at least one input index
    assert outputInds <= allInds

    # map indices to all array, axis pairs for that index
    indMap = {c: [(n, i) for n in range(len(inputs)) for (i, x) in enumerate(inputs[n]) if x == c] for c in allInds}

    for c in indMap:
        # each index has the same cardinality wherever it appears
        assert len({arrs[n].shape[i] for (n, i) in indMap[c]}) == 1

    # State: list of (set of letters, list of (corresponding indices, value))
    # Algo: while list contains more than one entry
    #         take two entries
    #         sort both lists by intersection of their indices
    #         merge compatible entries (where intersection of indices is equal - in the resulting list,
    #         take the union of indices and the product of values), stepping through each list linearly

    # TODO: might be faster to break into connected components first
    #       e.g. for "ab,d,bc->ad", the two components "ab,bc" and "d" are independent,
    #       so compute their content separately, then take cartesian product
    #       this would save a few pointless sorts by empty tuples

    # TODO: Consider investigating other performance ideas for these cases
    #       where the dense method beat the sparse method (usually sparse is faster)
    #       e,facd,c->cfed
    #        sparse: 0.0335489
    #        dense:  0.011465999999999997
    #       gbd,da,egb->da
    #        sparse: 0.0791625
    #        dense:  0.007319099999999995
    #       dcc,d,faedb,c->abe
    #        sparse: 1.2868097
    #        dense:  0.44605229999999985

    def merge(x1, x2):
        (s1, l1), (s2, l2) = x1, x2
        keys = {c for c in s1 if c in s2}  # intersection of strings
        outS = ''.join(set(s1 + s2))  # union of strings
        outMap = [(True, s1.index(c)) if c in s1 else (False, s2.index(c)) for c in outS]

        def keyGetter(s):
            inds = [s.index(c) for c in keys]
            return lambda p: tuple(p[0][ind] for ind in inds)
        kg1 = keyGetter(s1)
        kg2 = keyGetter(s2)
        l1.sort(key=kg1)
        l2.sort(key=kg2)
        i1 = i2 = 0
        outL = []
        while i1 < len(l1) and i2 < len(l2):
            k1, k2 = kg1(l1[i1]), kg2(l2[i2])
            if k1 < k2:
                i1 += 1
            elif k2 < k1:
                i2 += 1
            else:
                j1, j2 = i1, i2
                while j1 < len(l1) and kg1(l1[j1]) == k1:
                    j1 += 1
                while j2 < len(l2) and kg2(l2[j2]) == k2:
                    j2 += 1
                for c1, d1 in l1[i1:j1]:
                    for c2, d2 in l2[i2:j2]:
                        outL.append((tuple(c1[charIdx] if inFirst else c2[charIdx] for inFirst, charIdx in outMap),
                                     d1 * d2))
                i1 = j1
                i2 = j2
        return outS, outL

    # when indices are repeated within an array, pre-filter the coordinates and data
    def filter_inds(coords, data, n):
        counts = Counter(inputs[n])
        repeated = [(c, counts[c]) for c in counts if counts[c] > 1]
        if len(repeated) > 0:
            mask = np.full(len(data), True)
            for (k, v) in repeated:
                inds = [i for i in range(len(inputs[n])) if inputs[n][i] == k]
                for i in range(1, v):
                    mask &= (coords[:, inds[0]] == coords[:, inds[i]])
            if not all(mask):
                return coords[mask, :], data[mask]
        return coords, data

    xs = [(s, list(zip(c, d)))
          for n, (s, arr) in enumerate(zip(inputs, arrs))
          for c, d in [filter_inds(arr.coords.T, arr.data, n)]]

    # TODO: would using einsum's paths to optimize the order of merging help?
    while len(xs) > 1:
        xs.append(merge(xs.pop(), xs.pop()))

    results = defaultdict(int)

    for (s, l) in xs:
        coordMap = [s.index(c) for c in outputs]
        for (c, d) in l:
            results[tuple(c[i] for i in coordMap)] += d

    return sp.COO(np.array([k for k in results.keys()]).T,
                  np.array([v for v in results.values()]),
                  [arrs[indMap[c][0][0]].shape[indMap[c][0][1]] for c in outputs])


class WeightedModelWrapper(object):
    """Helper class for assiging weights to models without this option.

    Parameters
    ----------
    model_instance : estimator
        Model that requires weights.

    sample_type : string, optional (default=`weighted`)
        Method for adding weights to the model. `weighted` for linear regression models
        where the weights can be incorporated in the matrix multiplication,
        `sampled` for other models. `sampled` samples the training set according
        to the normalized weights and creates a dataset larger than the original.

    """

    def __init__(self, model_instance, sample_type="weighted"):
        self.model_instance = model_instance
        if sample_type == "weighted":
            self.data_transform = self._weighted_inputs
        else:
            warnings.warn("The model provided does not support sample weights. " +
                          "Manual weighted sampling may icrease the variance in the results.", UserWarning)
            self.data_transform = self._sampled_inputs

    def fit(self, X, y, sample_weight=None):
        """Fit underlying model instance with weighted inputs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples, n_outcomes)
            Target values.

        Returns
        -------
        self: an instance of the underlying estimator.
        """
        if sample_weight is not None:
            X, y = self.data_transform(X, y, sample_weight)
        return self.model_instance.fit(X, y)

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples, n_outcomes)
            Returns predicted values.
        """
        return self.model_instance.predict(X)

    def _weighted_inputs(self, X, y, sample_weight):
        normalized_weights = X.shape[0] * sample_weight / np.sum(sample_weight)
        sqrt_weights = np.sqrt(normalized_weights)
        weight_mat = np.diag(sqrt_weights)
        return np.matmul(weight_mat, X), np.matmul(weight_mat, y)

    def _sampled_inputs(self, X, y, sample_weight):
        # Normalize weights
        normalized_weights = sample_weight / np.sum(sample_weight)
        data_length = int(min(1 / np.min(normalized_weights[normalized_weights > 0]), 10) * X.shape[0])
        data_indices = np.random.choice(X.shape[0], size=data_length, p=normalized_weights)
        return X[data_indices], y[data_indices]


class MultiModelWrapper(object):
    """Helper class for assiging weights to models without this option.

    Parameters
    ----------
    model_list : array-like, shape (n_T, )
        List of models to be trained separately for each treatment group.
    """

    def __init__(self, model_list=[]):
        self.model_list = model_list
        self.n_T = len(model_list)

    def fit(self, Xt, y, sample_weight=None):
        """Fit underlying list of models with weighted inputs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features + n_treatments)
            Training data. The last n_T columns should be a one-hot encoding of the treatment assignment.

        y : array-like, shape (n_samples, )
            Target values.

        Returns
        -------
        self: an instance of the class
        """
        X = Xt[:, :-self.n_T]
        t = Xt[:, -self.n_T:]
        if sample_weight is None:
            for i in range(self.n_T):
                mask = (t[:, i] == 1)
                self.model_list[i].fit(X[mask], y[mask])
        else:
            for i in range(self.n_T):
                mask = (t[:, i] == 1)
                self.model_list[i].fit(X[mask], y[mask], sample_weight[mask])
        return self

    def predict(self, Xt):
        """Predict using the linear model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features + n_treatments)
            Samples. The last n_T columns should be a one-hot encoding of the treatment assignment.

        Returns
        -------
        C : array, shape (n_samples, )
            Returns predicted values.
        """
        X = Xt[:, :-self.n_T]
        t = Xt[:, -self.n_T:]
        predictions = [self.model_list[np.nonzero(t[i])[0][0]].predict(X[[i]]) for i in range(len(X))]
        return np.concatenate(predictions)

def _safe_norm_ppf(q, loc=0, scale=1):
    prelim = loc.copy()
    if hasattr(prelim, "__len__"):
        if np.any(scale > 0):
            prelim[scale > 0] = scipy.stats.norm.ppf(q, loc=loc[scale > 0], scale=scale[scale > 0])
    elif scale > 0:
        prelim = scipy.stats.norm.ppf(q, loc=loc, scale=scale)
    return prelim

class StatsModelsWrapperOLD:
    """
    Helper class to wrap a StatsModels OLS model to conform to the sklearn API.

    Parameters
    ----------
    fit_intercept: bool (default False)
        Whether to fit an intercept

    Attributes
    ----------
    fit_args: dict of str: object
        The arguments to pass to the `OLS` regression's `fit` method.  See the
        statsmodels documentation for more information.
    results: RegressionResults
        After `fit` has been called, this attribute will store the regression results.
    """

    def __init__(self, fit_intercept=True, fit_args={}):
        self.fit_args = fit_args
        self.fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        """
        Fit the ordinary least squares model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array_like, shape (n_samples, 1) or (n_samples,)
            Target values
        sample_weight : array_like, shape (n_samples,)
            Individual weights for each sample

        Returns
        -------
        self
        """
        assert ndim(y) == 1 or (ndim(y) == 2 and shape(y)[1] == 1)
        y = reshape(y, (-1,))
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        if sample_weight is not None:
            ols = WLS(y, X, weights=sample_weight, hasconst=self.fit_intercept)
        else:
            ols = WLS(y, X, hasconst=self.fit_intercept)
        self.results = ols.fit(**self.fit_args)
        return self

    def predict(self, X):
        """
        Predict using the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape (n_samples,)
            Predicted values
        """
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        return self.results.predict(X)

    def predict_interval(self, X, alpha=.05):
        """
        Get a confidence interval for the prediction at `X`.

        Parameters
        ----------
        X : array-like
            The features at which to predict
        alpha : float
            The significance level to use for the interval

        Returns
        -------
        array, shape (2, n_samples)
            Lower and upper bounds for the confidence interval at each sample point
        """
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        # NOTE: we use `obs = False` to get a confidence, rather than prediction, interval
        preds = self.results.get_prediction(X).conf_int(alpha=alpha, obs=False)
        # statsmodels uses the last dimension instead of the first to store the confidence intervals,
        # so we need to transpose the result
        return transpose(preds)

    @property
    def coef_(self):
        if self.fit_intercept:
            return self.results.params[1:]
        else:
            return self.results.params

    def coef__interval(self, alpha):
        if self.fit_intercept:
            return transpose(self.results.conf_int(alpha=alpha)[1:])
        else:
            return transpose(self.results.conf_int(alpha=alpha))

    @property
    def intercept_(self):
        if self.fit_intercept:
            return self.results.params[0]
        else:
            return 0
    
    def intercept__interval(self, alpha):
        if self.fit_intercept:
            return self.results.conf_int(alpha=alpha)[0]
        else:
            return np.array([0, 0])

class StatsModelsWrapper:
    def __init__(self, fit_intercept=True, fit_args={}):
        self.fit_args = fit_args
        self._fit_intercept = fit_intercept
        return

    def _check_input(self, X, y, sample_weight, var_weight):
        """ Check dimensions and other assertions """
        
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        elif np.any(np.not_equal(np.mod(sample_weight, 1), 0)):
            raise AttributeError("Sample weights must all be integers for inference to be valid!")
        
        if var_weight is None:
            if np.any(np.not_equal(sample_weight, 1)):
                warnings.warn("""No variance information was given for samples with sample_weight not equal to 1, that represent summaries of multiple original samples. Inference will be invalid!""")
            var_weight = np.zeros(y.shape)

        if var_weight.ndim < 2:
            if np.any(np.equal(sample_weight, 1) & np.not_equal(var_weight, 0)):
                warnings.warn("Variance was set to non-zero for an observation with sample_weight=1! var_weight represents the variance of the original observations that are summarized in this sample. Hence, cannot have a non-zero variance if only one observations was summarized. Inference will be invalid!")
        else:
            if np.any(np.equal(sample_weight, 1) & np.not_equal(np.sum(var_weight, axis=1), 0)):
                warnings.warn("Variance was set to non-zero for an observation with sample_weight=1! var_weight represents the variance of the original observations that are summarized in this sample. Hence, cannot have a non-zero variance if only one observations was summarized. Inference will be invalid!")
            

        assert X.shape[0] == y.shape[0] == sample_weight.shape[0] == var_weight.shape[0], "Input shapes not compatible!"
        if y.ndim >= 2:
            assert y.ndim == var_weight.ndim, "Input shapes not compatible: {}, {}!".format(y.ndim, var_weight.ndim)
            assert y.shape[1] == var_weight.shape[1], "Input shapes not compatible: {}, {}!".format(y.shape[1], var_weight.shape[1])

        return X, y, sample_weight, var_weight


    def fit(self, X, y, sample_weight=None, var_weight=None):
        """
        TODO. Add other types of covariance estimation (e.g. Newey-West (HAC), HC2, HC3)
        Parameters
        ----------
        X : (N, d) nd array like
            co-variates
        y : {(N,), (N, p)} nd array like
            output variable(s)
        sample_weight : (N,) nd array like of integers
            Weight for the observation. Observation i is treated as the mean
            outcome of sample_weight[i] independent observations
        var_weight : {(N,), (N, p)} nd array like
            Variance of the outcome(s) of the original sample_weight[i] observations
            that were used to compute the mean outcome represented by observation i.
        """
        
        X, y, sample_weight, var_weight = self._check_input(X, y, sample_weight, var_weight)

        if self._fit_intercept:
            X = add_constant(X, has_constant='add')
        WX = X * np.sqrt(sample_weight).reshape(-1, 1)

        if y.ndim < 2:
            self._n_out = 0
            wy = y * np.sqrt(sample_weight)
        else:
            self._n_out = y.shape[1]
            wy = y * np.sqrt(sample_weight).reshape(-1, 1)
        
        param, _, rank, _ = np.linalg.lstsq(WX, wy, rcond=None)

        if rank < param.shape[0]:
            warnings.warn("Co-variance matrix is undertermined. Inference will be invalid!")

        sigma_inv = np.linalg.pinv(np.matmul(WX.T, WX))

        self._param = param
        var_i = var_weight + (y - np.matmul(X, param))**2
        n_obs = np.sum(sample_weight)
        df = len(param) if self._n_out == 0 else param.shape[0]

        if n_obs <= df:
            warnings.warn("Number of observations <= than number of parameters. Using biased variance calculation!")
            correction = 1
        else:
            correction = (n_obs / (n_obs - df))

        if (not 'cov_type' in self.fit_args) or (self.fit_args['cov_type'] == 'nonrobust'):
            if y.ndim < 2:
                self._var = correction * np.average(var_i, weights=sample_weight) * sigma_inv
            else:
                vars = correction * np.average(var_i, weights=sample_weight, axis=0)
                self._var = [v * sigma_inv for v in vars]
        elif (self.fit_args['cov_type'] == 'HC0'):
            if y.ndim < 2:
                weighted_sigma = np.matmul(WX.T, WX * var_i.reshape(-1, 1))
                self._var = np.matmul(sigma_inv, np.matmul(weighted_sigma, sigma_inv))
            else:
                self._var = []
                for j in range(self._n_out):
                    weighted_sigma = np.matmul(WX.T, WX * var_i[:, [j]])
                    self._var.append(np.matmul(sigma_inv, np.matmul(weighted_sigma, sigma_inv)))
        elif (self.fit_args['cov_type'] == 'HC1'):
            if y.ndim < 2:
                weighted_sigma = np.matmul(WX.T, WX * var_i.reshape(-1, 1))
                self._var = correction * np.matmul(sigma_inv, np.matmul(weighted_sigma, sigma_inv))
            else:
                self._var = []
                for j in range(self._n_out):
                    weighted_sigma = np.matmul(WX.T, WX * var_i[:, [j]])
                    self._var.append(correction * np.matmul(sigma_inv, np.matmul(weighted_sigma, sigma_inv)))
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : (n, d) array like
            The covariates on which to predict
        Returns
        -------
        predictions : {(n,) array, (n,p) array}
            The predicted mean outcomes
        """
        if self._fit_intercept:
            X = add_constant(X, has_constant='add')
        return np.matmul(X, self._param)

    @property
    def coef_(self):
        """
        Returns
        -------
        coef_ : {(d,), (p, d)} nd array like
            The coefficients of the variables in the linear regression. If label y
            was p-dimensional, then the result is a matrix of coefficents, whose p-th
            row containts the coefficients corresponding to the p-th coordinate of the label.
        """
        if self._fit_intercept:
            if self._n_out == 0:
                return self._param[1:]
            else:
                return self._param[1:].T
        else:
            if self._n_out == 0:
                return self._param
            else:
                return self._param.T

    @property
    def intercept_(self):
        """
        Returns
        -------
        intercept_ : float or (p,) nd array like
            The intercept of the linear regresion. If label y was p-dimensional, then the result is a vector, whose p-th
            entry containts the intercept corresponding to the p-th coordinate of the label.
        """
        return self._param[0] if self._fit_intercept else (0 if self._n_out==0 else np.zeros(self._n_out))

    @property
    def _param_var(self):
        """
        Returns
        -------
        var : {(d (+1), d (+1)), (p, d (+1), d (+1))} nd array like
            The covariance matrix of all the parameters in the regression (including the intercept as the first parameter).
            If intercept was set to False then this is the covariance matrix of the coefficients, otherwise, the intercept
            is treated as the first parameter of the regression and the coefficients as the remaining. If
            outcome y is p-dimensional, then this is a tensor whose p-th entry contains the co-variance matrix
            for the parameters corresponding to the regression of the p-th coordinate of the outcome.
        """
        return np.array(self._var)

    @property
    def _param_stderr(self):
        """
        Returns
        -------
        _param_stderr : {(d (+1),) (d (+1), p)} nd array like
            The standard error of each parameter that was estimated.
        """
        if self._n_out == 0:
            return np.sqrt(np.diag(self._param_var))
        else:
            return np.array([np.sqrt(np.diag(v)) for v in self._param_var]).T

    @property
    def coef_stderr_(self):
        """
        Returns
        -------
        coef_stderr_ : {(d,), (p, d)} nd array like
            The standard error of the coefficients
        """
        return self._param_stderr[1:].T if self._fit_intercept else self._param_stderr.T

    @property
    def intercept_stderr_(self):
        """
        Returns
        -------
        intercept_stderr_ : float or (p,) nd array like
            The stadnard error of the intercept(s)
        """
        return self._param_stderr[0] if self._fit_intercept else (0 if self._n_out==0 else np.zeros(self._n_out))

    def prediction_stderr(self, X):
        """
        Parameters
        ----------
        X : (n, d) array like
            The covariates at which to predict
        Returns
        -------
        prediction_stderr : (n, p) array like
            The standard error of each coordinate of the output at each point we predict
        """
        if self._fit_intercept:
            X = add_constant(X, has_constant='add')
        if self._n_out == 0:
            return np.sqrt(np.sum(np.matmul(X, self._param_var) * X, axis=1))
        else:
            return np.array([np.sqrt(np.sum(np.matmul(X, v) * X, axis=1)) for v in self._var]).T

    def coef__interval(self, alpha=.05):
        """
        Parameters
        ----------
        alpha : float
            The confidence level. Will calculate the alpha/2-quantile and the (1-alpha/2)-quantile
            of the parameter distribution as confidence interval
        Returns
        -------
        coef__interval : {tuple ((p, d) array, (p,d) array), tuple ((d,) array, (d,) array)}
            The lower and upper bounds of the confidence interval of the coefficients
        """
        return np.array([_safe_norm_ppf(alpha/2, loc=p, scale=err)
                 for p, err in zip(self.coef_, self.coef_stderr_)]),\
               np.array([_safe_norm_ppf(1-alpha/2, loc=p, scale=err)
                 for p, err in zip(self.coef_, self.coef_stderr_)])

    def intercept__interval(self, alpha=.05):
        """
        Parameters
        ----------
        alpha : float
            The confidence level. Will calculate the alpha/2-quantile and the (1-alpha/2)-quantile
            of the parameter distribution as confidence interval
        Returns
        -------
        intercept__interval : {tuple ((p,) array, (p,) array), tuple (float, float)}
            The lower and upper bounds of the confidence interval of the intercept(s)
        """
        if not self._fit_intercept:
            return (0 if self._n_out==0 else np.zeros(self._n_out)), (0 if self._n_out==0 else np.zeros(self._n_out))

        if self._n_out == 0:
            return _safe_norm_ppf(alpha/2, loc=self.intercept_, scale=self.intercept_stderr_),\
                    _safe_norm_ppf(1-alpha/2, loc=self.intercept_, scale=self.intercept_stderr_)
        else:
            return np.array([_safe_norm_ppf(alpha/2, loc=p, scale=err)
                 for p, err in zip(self.intercept_, self.intercept_stderr_)]),\
               np.array([_safe_norm_ppf(1-alpha/2, loc=p, scale=err)
                 for p, err in zip(self.intercept_, self.intercept_stderr_)])

    def predict_interval(self, X, alpha=.05):
        """
        Parameters
        ----------
        X : (n, d) array like
            The covariates on which to predict
        alpha : float
            The confidence level. Will calculate the alpha/2-quantile and the (1-alpha/2)-quantile
            of the parameter distribution as confidence interval
        Returns
        -------
        prediction_intervals : {tuple ((n,) array, (n,) array), tuple ((n,p) array, (n,p) array)}
            The lower and upper bounds of the confidence intervals of the predicted mean outcomes
        """
        return np.array([_safe_norm_ppf(alpha/2, loc=p, scale=err)
                 for p, err in zip(self.predict(X), self.prediction_stderr(X))]),\
               np.array([_safe_norm_ppf(1-alpha/2, loc=p, scale=err)
                 for p, err in zip(self.predict(X), self.prediction_stderr(X))])


class LassoCVWrapper:
    """Helper class to wrap either LassoCV or MultiTaskLassoCV depending on the shape of the target."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, Y):
        assert shape(X)[0] == shape(Y)[0]
        assert ndim(Y) <= 2
        if ndim(Y) == 2 and shape(Y)[1] > 1:
            self.model = MultiTaskLassoCV(*self.args, **self.kwargs)
        else:
            self.model = LassoCV(*self.args, **self.kwargs)
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)
