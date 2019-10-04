# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for lasso extensions."""

import numpy as np
import pytest
import unittest
import warnings
from utilities import WeightedLasso, DebiasedLasso
from sklearn.linear_model import Lasso, LinearRegression


class TestWeightedLasso(unittest.TestCase):
    """Test WeightedLasso."""

    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        np.random.seed(123)
        # DGP constants
        cls.n_samples = 1000
        cls.n_dim = 50
        cls.X = np.random.normal(size=(cls.n_samples, cls.n_dim))
        # DGP coefficients
        cls.coefs1 = np.zeros(cls.n_dim)
        nonzero_idx1 = np.random.choice(cls.n_dim, replace=False, size=5)
        cls.coefs2 = np.zeros(cls.n_dim)
        nonzero_idx2 = np.random.choice(cls.n_dim, replace=False, size=5)
        cls.coefs1[nonzero_idx1] = 1
        cls.coefs2[nonzero_idx2] = 1
        cls.intercept = 3
        cls.intercept1 = 2
        cls.intercept2 = 0
        cls.error_sd = 0.2
        # Generated outcomes
        cls.y1 = cls.intercept1 + np.dot(cls.X[:cls.n_samples // 2], cls.coefs1) + \
            np.random.normal(scale=cls.error_sd, size=cls.n_samples // 2)
        cls.y2 = cls.intercept2 + np.dot(cls.X[cls.n_samples // 2:], cls.coefs2) + \
            np.random.normal(scale=cls.error_sd, size=cls.n_samples // 2)
        cls.y = np.concatenate((cls.y1, cls.y2))
        cls.y_simple = np.dot(cls.X, cls.coefs1) + np.random.normal(scale=cls.error_sd, size=cls.n_samples)

    def test_one_DGP(self):
        """Test WeightedLasso with one set of coefficients."""
        # Define weights
        sample_weight = np.concatenate((np.ones(TestWeightedLasso.n_samples // 2),
                                        np.ones(TestWeightedLasso.n_samples // 2) * 2))
        # Define extended datasets
        X_expanded = np.concatenate((TestWeightedLasso.X, TestWeightedLasso.X[TestWeightedLasso.n_samples // 2:]))
        y_expanded = np.concatenate(
            (TestWeightedLasso.y_simple, TestWeightedLasso.y_simple[TestWeightedLasso.n_samples // 2:]))
        # Range of alphas
        alpha_range = [0.001, 0.01, 0.1]
        # Compare with Lasso
        # --> No intercept
        params = {'fit_intercept': False}
        self._compare_with_lasso(X_expanded, y_expanded, TestWeightedLasso.X,
                                 TestWeightedLasso.y_simple, sample_weight, alpha_range, params)
        # --> With intercept
        params = {'fit_intercept': True}
        # When DGP has no intercept
        self._compare_with_lasso(X_expanded, y_expanded, TestWeightedLasso.X,
                                 TestWeightedLasso.y_simple, sample_weight, alpha_range, params)
        # When DGP has intercept
        self._compare_with_lasso(X_expanded, y_expanded + TestWeightedLasso.intercept, TestWeightedLasso.X,
                                 TestWeightedLasso.y_simple + TestWeightedLasso.intercept,
                                 sample_weight, alpha_range, params)
        # --> Coerce coefficients to be positive
        params = {'positive': True}
        self._compare_with_lasso(X_expanded, y_expanded, TestWeightedLasso.X,
                                 TestWeightedLasso.y_simple, sample_weight, alpha_range, params)
        # --> Toggle max_iter & tol
        params = {'max_iter': 100, 'tol': 1e-3}
        self._compare_with_lasso(X_expanded, y_expanded, TestWeightedLasso.X,
                                 TestWeightedLasso.y_simple, sample_weight, alpha_range, params)

    def test_mixed_DGP(self):
        """Test WeightedLasso with two sets of coefficients."""
        # Define weights
        sample_weight = np.concatenate((np.ones(TestWeightedLasso.n_samples // 2),
                                        np.zeros(TestWeightedLasso.n_samples // 2)))
        # Data from one DGP has weight 0. Check that we recover correct coefficients
        self._compare_with_lasso(TestWeightedLasso.X[:TestWeightedLasso.n_samples // 2], TestWeightedLasso.y1,
                                 TestWeightedLasso.X, TestWeightedLasso.y, sample_weight)
        # Mixed DGP scenario.
        sample_weight = np.concatenate((np.ones(TestWeightedLasso.n_samples // 2),
                                        np.ones(TestWeightedLasso.n_samples // 2) * 2))
        # Define extended datasets
        X_expanded = np.concatenate((TestWeightedLasso.X, TestWeightedLasso.X[TestWeightedLasso.n_samples // 2:]))
        y_expanded = np.concatenate(
            (TestWeightedLasso.y1, TestWeightedLasso.y2, TestWeightedLasso.y2))
        self._compare_with_lasso(X_expanded, y_expanded,
                                 TestWeightedLasso.X, TestWeightedLasso.y, sample_weight)

    def test_multiple_outputs(self):
        """Test multiple outputs."""
        # Define weights
        sample_weight = np.concatenate((np.ones(TestWeightedLasso.n_samples // 2),
                                        np.zeros(TestWeightedLasso.n_samples // 2)))
        # Define multioutput
        y_2D = np.concatenate((TestWeightedLasso.y_simple.reshape(-1, 1), TestWeightedLasso.y.reshape(-1, 1)), axis=1)
        self._compare_with_lasso(TestWeightedLasso.X[:TestWeightedLasso.n_samples // 2],
                                 y_2D[:TestWeightedLasso.n_samples // 2],
                                 TestWeightedLasso.X, y_2D, sample_weight)

    def _compare_with_lasso(self, lasso_X, lasso_y, wlasso_X, wlasso_y, sample_weight, alpha_range=[0.01], params={}):
        for alpha in alpha_range:
            lasso = Lasso(alpha=alpha)
            lasso.set_params(**params)
            lasso.fit(lasso_X, lasso_y)
            wlasso = WeightedLasso(alpha=alpha)
            wlasso.set_params(**params)
            wlasso.fit(wlasso_X, wlasso_y, sample_weight=sample_weight)
            # Check results are similar with tolerance 1e-6
            if np.ndim(lasso_y) > 1:
                for i in range(lasso_y.shape[1]):
                    self.assertTrue(np.allclose(lasso.coef_[i], wlasso.coef_[i]))
                    self.assertAlmostEqual(lasso.intercept_[i], wlasso.intercept_[i])
            else:
                self.assertTrue(np.allclose(lasso.coef_, wlasso.coef_))
                self.assertAlmostEqual(lasso.intercept_, wlasso.intercept_)
