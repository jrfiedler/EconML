import numpy as np
from econml.dml import DMLCateEstimator, LinearDMLCateEstimator
from econml.utilities import hstack, StatsModelsWrapperOLD
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV, Lasso, MultiTaskLassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import scipy.special
import time
from econml.utilities import StatsModelsWrapper as OLS
from econml.utilities import StatsModelsWrapperOLD as StatsModelsOLS
import unittest
import joblib

def _compare_classes(est, lr, X_test, alpha=.05):
    assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
    assert np.all(np.abs(np.array(est.coef__interval(alpha=alpha)) - np.array(lr.coef__interval(alpha=alpha))) < 1e-12),\
            "{}, {}".format(est.coef__interval(alpha=alpha), np.array(lr.coef__interval(alpha=alpha)))
    assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.intercept_, lr.intercept_)
    assert np.all(np.abs(np.array(est.intercept__interval(alpha=alpha)) - np.array(lr.intercept__interval(alpha=alpha))) < 1e-12),\
            "{}, {}".format(est.intercept__interval(alpha=alpha), lr.intercept__interval(alpha=alpha))
    assert np.all(np.abs(est.predict(X_test) - lr.predict(X_test)) < 1e-12), "{}, {}".format(est.predict(X_test), lr.predict(X_test))
    assert np.all(np.abs(np.array(est.predict_interval(X_test, alpha=alpha)) - np.array(lr.predict_interval(X_test, alpha=alpha))) < 1e-12),\
            "{}, {}".format(est.predict_interval(X_test, alpha=alpha), lr.predict_interval(X_test, alpha=alpha))

def _summarize(X, y):
    X_unique = np.unique(X, axis=0)
    y_sum_first = []
    n_sum_first = []
    var_first = []
    X_final_first = []
    y_sum_sec = []
    n_sum_sec = []
    var_sec = []
    X_final_sec= []
    X1 = []
    X2 = []
    y1 = []
    y2 = []
    for it, xt in enumerate(X_unique):
        mask = (X==xt).all(axis=1)
        if mask.any():
            y_mask = y[mask]
            X_mask = X[mask]
            if np.sum(mask) >=2:
                X_mask_first = X_mask[:y_mask.shape[0]//2]
                X_mask_sec = X_mask[y_mask.shape[0]//2:]
                y_mask_first = y_mask[:y_mask.shape[0]//2]
                y_mask_sec = y_mask[y_mask.shape[0]//2:]
                
                X1 = np.vstack([X1, X_mask_first]) if len(X1) > 0 else X_mask_first
                y1 = np.concatenate((y1, y_mask_first)) if len(y1) > 0 else y_mask_first
                X2 = np.vstack([X2, X_mask_sec]) if len(X2) > 0 else X_mask_sec
                y2 = np.concatenate((y2, y_mask_sec)) if len(y2) > 0 else y_mask_sec

                y_sum_first.append(np.mean(y_mask_first, axis=0))
                n_sum_first.append(len(y_mask_first))
                var_first.append(np.var(y_mask_first, axis=0))
                X_final_first.append(xt)
                y_sum_sec.append(np.mean(y_mask_sec, axis=0))
                n_sum_sec.append(len(y_mask_sec))
                var_sec.append(np.var(y_mask_sec, axis=0))
                X_final_sec.append(xt)
            else:
                if np.random.binomial(1, .5, size=1)==1:
                    X1 = np.vstack([X1, X_mask]) if len(X1) > 0 else X_mask
                    y1 = np.concatenate((y1, y_mask)) if len(y1) > 0 else y_mask
                    y_sum_first.append(np.mean(y_mask, axis=0))
                    n_sum_first.append(len(y_mask))
                    var_first.append(np.var(y_mask, axis=0))
                    X_final_first.append(xt)
                else:
                    X2 = np.vstack([X2, X_mask]) if len(X2) > 0 else X_mask
                    y2 = np.concatenate((y2, y_mask)) if len(y2) > 0 else y_mask
                    y_sum_sec.append(np.mean(y_mask, axis=0))
                    n_sum_sec.append(len(y_mask))
                    var_sec.append(np.var(y_mask, axis=0))
                    X_final_sec.append(xt)

    return X1, X2, y1, y2, X_final_first, X_final_sec, y_sum_first, y_sum_sec, n_sum_first, n_sum_sec, var_first, var_sec

def _compare_dml_classes(est, lr, X_test, alpha=.05):
    assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
    assert np.all(np.abs(np.array(est.coef__interval(alpha=alpha)) - np.array(lr.coef__interval(alpha=alpha))) < 1e-10),\
            "{}, {}".format(np.array(est.coef__interval(alpha=alpha)), np.array(lr.coef__interval(alpha=alpha)))
    assert np.all(np.abs(est.effect(X_test) - lr.effect(X_test)) < 1e-12), "{}, {}".format(est.effect(X_test), lr.effect(X_test))
    assert np.all(np.abs(np.array(est.effect_interval(X_test, alpha=alpha)) - np.array(lr.effect_interval(X_test, alpha=alpha))) < 1e-10),\
            "{}, {}".format(est.effect_interval(X_test, alpha=alpha), lr.effect_interval(X_test, alpha=alpha))


class TestStatsModels(unittest.TestCase):

    def test_comp_with_lr(self):
        """ Testing that we recover the same as sklearn's linear regression in terms of point estimates """
        np.random.seed(123)
        n = 1000
        d = 3
        X = np.random.binomial(1, .8, size=(n, d))
        T = np.random.binomial(1, .5*X[:, 0]+.25, size=(n,))
        true_effect = lambda x: x[:, 0] + .5
        y = true_effect(X)*T + X[:, 0] + X[:, 2]
        weights = np.random.randint(5, 100, size=(X.shape[0]))

        est = OLS().fit(X, y)
        lr = LinearRegression().fit(X, y)
        assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
        assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.coef_, lr.intercept_)

        est = OLS(fit_intercept=False).fit(X, y)
        lr = LinearRegression(fit_intercept=False).fit(X, y)
        assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
        assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.coef_, lr.intercept_)

        est = OLS(fit_intercept=False).fit(X, y, sample_weight=weights)
        lr = LinearRegression(fit_intercept=False).fit(X, y, sample_weight=weights)
        assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
        assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.coef_, lr.intercept_)

        est = OLS(fit_intercept=False).fit(X, y, sample_weight=weights, var_weight=np.ones(y.shape))
        lr = LinearRegression(fit_intercept=False).fit(X, y, sample_weight=weights)
        assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
        assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.coef_, lr.intercept_)

        n = 1000
        d = 3
        for p in np.arange(1, 4):
            X = np.random.binomial(1, .8, size=(n, d))
            T = np.random.binomial(1, .5*X[:, 0]+.25, size=(n,))
            true_effect = lambda x: np.hstack([x[:, [0]] + .5 + t for t in range(p)])
            y = np.zeros((n, p))
            y = true_effect(X)*T.reshape(-1,1) + X[:, [0]*p] + (0*X[:,[0]*p] + 1)*np.random.normal(0, 1, size=(n,p))
            weights = np.random.randint(5, 100, size=(X.shape[0]))

            est = OLS().fit(X, y)
            lr = LinearRegression().fit(X, y)
            assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
            assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.intercept_, lr.intercept_)

            est = OLS(fit_intercept=False).fit(X, y)
            lr = LinearRegression(fit_intercept=False).fit(X, y)
            assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
            assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.intercept_, lr.intercept_)

            est = OLS(fit_intercept=False).fit(X, y, sample_weight=weights)
            lr = LinearRegression(fit_intercept=False).fit(X, y, sample_weight=weights)
            assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
            assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.intercept_, lr.intercept_)

            est = OLS(fit_intercept=False).fit(X, y, sample_weight=weights, var_weight=np.ones(y.shape))
            lr = LinearRegression(fit_intercept=False).fit(X, y, sample_weight=weights)
            assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
            assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.intercept_, lr.intercept_)

    def test_inference(self):
        """ Testing that we recover the expected standard errors and confidence intervals in a known example """

        # 1-d output
        d = 3
        X = np.vstack([np.eye(d)])
        y = X[:, 0]
        est = OLS(fit_intercept=False).fit(X, y)
        assert np.all(np.abs(est.coef_ - [1, 0, 0]) <= 1e-12), "{}, {}".format(est.coef_, [1, 0, 0])
        assert np.all(np.abs(est.coef__interval() - np.array([[1, 0, 0], [1, 0, 0]])) <= 1e-12),\
                "{}, {}".format(est.coef__interval(), np.array([[1, 0, 0], [1, 0, 0]]))
        assert np.all(est.coef_stderr_ <= 1e-12)
        assert np.all(est._param_var <= 1e-12)

        d = 3
        X = np.vstack([np.eye(d), np.ones((1, d)), np.zeros((1, d))])
        y = X[:, 0]
        est = OLS(fit_intercept=True).fit(X, y)
        assert np.all(np.abs(est.coef_ - np.array([1]+[0]*(d-1))) <= 1e-12), "{}, {}".format(est.coef_, [1]+[0]*(d-1))
        assert np.all(np.abs(est.coef__interval() - np.array([[1]+[0]*(d-1), [1]+[0]*(d-1)])) <= 1e-12),\
                "{}, {}".format(est.coef__interval(), np.array([[1]+[0]*(d-1), [1]+[0]*(d-1)]))
        assert np.all(est.coef_stderr_ <= 1e-12)
        assert np.all(est._param_var <= 1e-12)
        assert np.abs(est.intercept_) <= 1e-12
        assert np.all(np.abs(est.intercept__interval()) <= 1e-12)

        d = 3
        X = np.vstack([np.eye(d)])
        y = np.concatenate((X[:, 0] - 1, X[:, 0] + 1))
        X = np.vstack([X, X])
        est = OLS(fit_intercept=False).fit(X, y)
        assert np.all(np.abs(est.coef_ - ([1]+[0]*(d-1))) <= 1e-12), "{}, {}".format(est.coef_, [1]+[0]*(d-1))
        assert np.all(np.abs(est.coef_stderr_ - np.array([1]*d)) <= 1e-12)
        assert np.all(np.abs(est.coef__interval()[0] - np.array([scipy.stats.norm.ppf(.025, loc=1, scale=1)]\
                                                                +[scipy.stats.norm.ppf(.025, loc=0, scale=1)]*(d-1))) <= 1e-12),\
                "{}, {}".format(est.coef__interval()[0], np.array([scipy.stats.norm.ppf(.025, loc=1, scale=1)]\
                                                                +[scipy.stats.norm.ppf(.025, loc=0, scale=1)]*(d-1)))
        assert np.all(np.abs(est.coef__interval()[1] - np.array([scipy.stats.norm.ppf(.975, loc=1, scale=1)]\
                                                    +[scipy.stats.norm.ppf(.975, loc=0, scale=1)]*(d-1))) <= 1e-12),\
                "{}, {}".format(est.coef__interval()[1], np.array([scipy.stats.norm.ppf(.975, loc=1, scale=1)]\
                                                    +[scipy.stats.norm.ppf(.975, loc=0, scale=1)]*(d-1)))

        # 2-d output
        d = 3
        p = 4
        X = np.vstack([np.eye(d)])
        y = np.vstack((X[:, [0]*p] - 1, X[:, [0]*p] + 1))
        X = np.vstack([X, X])
        est = OLS(fit_intercept=False).fit(X, y)
        for t in range(p):
            assert np.all(np.abs(est.coef_[t] - ([1]+[0]*(d-1))) <= 1e-12), "{}, {}".format(est.coef_[t], [1]+[0]*(d-1))
            assert np.all(np.abs(est.coef_stderr_[t] - np.array([1]*d)) <= 1e-12), "{}".format(est.coef_stderr_[t])
            assert np.all(np.abs(est.coef__interval()[0][t] - np.array([scipy.stats.norm.ppf(.025, loc=1, scale=1)]\
                                                                    +[scipy.stats.norm.ppf(.025, loc=0, scale=1)]*(d-1))) <= 1e-12),\
                    "{}, {}".format(est.coef__interval()[0][t], np.array([scipy.stats.norm.ppf(.025, loc=1, scale=1)]\
                                                                    +[scipy.stats.norm.ppf(.025, loc=0, scale=1)]*(d-1)))
            assert np.all(np.abs(est.coef__interval()[1][t] - np.array([scipy.stats.norm.ppf(.975, loc=1, scale=1)]\
                                                        +[scipy.stats.norm.ppf(.975, loc=0, scale=1)]*(d-1))) <= 1e-12),\
                    "{}, {}".format(est.coef__interval()[1][t], np.array([scipy.stats.norm.ppf(.975, loc=1, scale=1)]\
                                                        +[scipy.stats.norm.ppf(.975, loc=0, scale=1)]*(d-1)))
            assert np.all(np.abs(est.intercept_[t]) <= 1e-12), "{}, {}".format(est.intercept_[t])
            assert np.all(np.abs(est.intercept_stderr_[t]) <= 1e-12), "{}".format(est.intercept_stderr_[t])
            assert np.all(np.abs(est.intercept__interval()[0][t]) <= 1e-12), "{}".format(est.intercept__interval()[0][t])

        d = 3
        p = 4
        X = np.vstack([np.eye(d), np.zeros((1,d))])
        y = np.vstack((X[:, [0]*p] - 1, X[:, [0]*p] + 1))
        X = np.vstack([X, X])
        est = OLS(fit_intercept=True).fit(X, y)
        for t in range(p):
            assert np.all(np.abs(est.coef_[t] - ([1]+[0]*(d-1))) <= 1e-12), "{}, {}".format(est.coef_[t], [1]+[0]*(d-1))
            assert np.all(np.abs(est.coef_stderr_[t] - np.array([np.sqrt(2)]*d)) <= 1e-12), "{}".format(est.coef_stderr_[t])
            assert np.all(np.abs(est.coef__interval()[0][t] - np.array([scipy.stats.norm.ppf(.025, loc=1, scale=np.sqrt(2))]\
                                                                    +[scipy.stats.norm.ppf(.025, loc=0, scale=np.sqrt(2))]*(d-1))) <= 1e-12),\
                    "{}, {}".format(est.coef__interval()[0][t], np.array([scipy.stats.norm.ppf(.025, loc=1, scale=np.sqrt(2))]\
                                                                    +[scipy.stats.norm.ppf(.025, loc=0, scale=np.sqrt(2))]*(d-1)))
            assert np.all(np.abs(est.coef__interval()[1][t] - np.array([scipy.stats.norm.ppf(.975, loc=1, scale=np.sqrt(2))]\
                                                        +[scipy.stats.norm.ppf(.975, loc=0, scale=np.sqrt(2))]*(d-1))) <= 1e-12),\
                    "{}, {}".format(est.coef__interval()[1][t], np.array([scipy.stats.norm.ppf(.975, loc=1, scale=np.sqrt(2))]\
                                                        +[scipy.stats.norm.ppf(.975, loc=0, scale=np.sqrt(2))]*(d-1)))
            assert np.all(np.abs(est.intercept_[t]) <= 1e-12), "{}, {}".format(est.intercept_[t])
            assert np.all(np.abs(est.intercept_stderr_[t] - 1) <= 1e-12), "{}".format(est.intercept_stderr_[t])
            assert np.all(np.abs(est.intercept__interval()[0][t] - scipy.stats.norm.ppf(.025, loc=0, scale=1)) <= 1e-12),\
                    "{}, {}".format(est.intercept__interval()[0][t], scipy.stats.norm.ppf(.025, loc=0, scale=1))

    def test_comp_with_statsmodels(self):
        """ Comparing with confidence intervals and standard errors of statsmodels in the un-weighted case """
        np.random.seed(123)
        
        # Single dimensional output y
        n = 1000
        d = 3
        X = np.random.binomial(1, .8, size=(n, d))
        T = np.random.binomial(1, .5*X[:, 0]+.25, size=(n,))
        true_effect = lambda x: x[:, 0] + .5
        y = true_effect(X)*T + X[:, 0] + X[:, 2] + np.random.normal(0, 1, size=(n,))
        X_test = np.unique(np.random.binomial(1, .5, size=(n, d)), axis=0)
        for fit_intercept in [True, False]:
            for cov_type in ['nonrobust', 'HC0', 'HC1']:
                est = OLS(fit_intercept=fit_intercept, fit_args={'cov_type':cov_type}).fit(X, y)
                lr = StatsModelsOLS(fit_intercept=fit_intercept, fit_args={'cov_type':cov_type, 'use_t': False}).fit(X, y)
                _compare_classes(est, lr, X_test)
        
        n = 1000
        d = 3
        X = np.random.normal(0, 1, size=(n, d))
        y = X[:, 0] + X[:, 2] + np.random.normal(0, 1, size=(n,))
        X_test = np.unique(np.random.binomial(1, .5, size=(n, d)), axis=0)
        for fit_intercept in [True, False]:
            for cov_type in ['nonrobust', 'HC0', 'HC1']:
                est = OLS(fit_intercept=fit_intercept, fit_args={'cov_type':cov_type}).fit(X, y)
                lr = StatsModelsOLS(fit_intercept=fit_intercept, fit_args={'cov_type':cov_type, 'use_t': False}).fit(X, y)
                _compare_classes(est, lr, X_test)

        d = 3
        X = np.vstack([np.eye(d)])
        y = np.concatenate((X[:, 0] - 1, X[:, 0] + 1))
        X = np.vstack([X, X])
        X_test = np.unique(np.random.binomial(1, .5, size=(n, d)), axis=0)
        
        for cov_type in ['nonrobust', 'HC0', 'HC1']:
            for alpha in [.01, .05, .1]:
                _compare_classes(OLS(fit_intercept=False, fit_args={'cov_type':cov_type}).fit(X, y),
                                StatsModelsOLS(fit_intercept=False, fit_args={'cov_type':cov_type, 'use_t': False}).fit(X, y),
                                X_test, alpha=alpha)

        d = 3
        X = np.vstack([np.eye(d), np.ones((1, d)), np.zeros((1, d))])
        y = np.concatenate((X[:, 0] - 1, X[:, 0] + 1))
        X = np.vstack([X, X])
        X_test = np.unique(np.random.binomial(1, .5, size=(n, d)), axis=0)
        for cov_type in ['nonrobust', 'HC0', 'HC1']:
            _compare_classes(OLS(fit_intercept=True, fit_args={'cov_type':cov_type}).fit(X, y),
                            StatsModelsOLS(fit_intercept=True, fit_args={'cov_type':cov_type, 'use_t': False}).fit(X, y), X_test)

        # Multi-dimensional output y
        n = 1000
        d = 3
        for p in np.arange(1, 4):
            X = np.random.binomial(1, .8, size=(n, d))
            T = np.random.binomial(1, .5*X[:, 0]+.25, size=(n,))
            true_effect = lambda x: np.hstack([x[:, [0]] + .5 + t for t in range(p)])
            y = np.zeros((n, p))
            y = true_effect(X)*T.reshape(-1,1) + X[:, [0]*p] + (0*X[:,[0]*p] + 1)*np.random.normal(0, 1, size=(n,p))

            for cov_type in ['nonrobust', 'HC0', 'HC1']:    
                for fit_intercept in [True, False]:
                    for alpha in [.01, .05, .2]:
                        est = OLS(fit_intercept=fit_intercept, fit_args={'cov_type':cov_type}).fit(X, y)
                        lr = [StatsModelsOLS(fit_intercept=fit_intercept, fit_args={'cov_type':cov_type, 'use_t': False}).fit(X, y[:, t]) for t in range(p)]
                        for t in range(p):
                            assert np.all(np.abs(est.coef_[t] - lr[t].coef_) < 1e-12), "{}, {}, {}: {}, {}".format(cov_type, fit_intercept, t, est.coef_[t], lr[t].coef_)
                            assert np.all(np.abs(np.array(est.coef__interval(alpha=alpha))[:, t] - lr[t].coef__interval(alpha=alpha)) < 1e-12),\
                                    "{}, {}, {}: {} vs {}".format(cov_type, fit_intercept, t, np.array(est.coef__interval(alpha=alpha))[:, t], lr[t].coef__interval(alpha=alpha))
                            assert np.all(np.abs(est.intercept_[t] - lr[t].intercept_) < 1e-12),\
                                    "{}, {}, {}: {} vs {}".format(cov_type, fit_intercept, t, est.intercept_[t], lr[t].intercept_)
                            assert np.all(np.abs(np.array(est.intercept__interval(alpha=alpha))[:, t] - lr[t].intercept__interval(alpha=alpha)) < 1e-12),\
                                    "{}, {}, {}: {} vs {}".format(cov_type, fit_intercept, t, np.array(est.intercept__interval(alpha=alpha))[:, t], lr[t].intercept__interval(alpha=alpha))
                            assert np.all(np.abs(est.predict(X_test)[:, t] - lr[t].predict(X_test)) < 1e-12),\
                                    "{}, {}, {}: {} vs {}".format(cov_type, fit_intercept, t, est.predict(X_test)[:, t], lr[t].predict(X_test))
                            assert np.all(np.abs(np.array(est.predict_interval(X_test, alpha=alpha))[:, :, t] - lr[t].predict_interval(X_test, alpha=alpha)) < 1e-12),\
                                    "{}, {}, {}: {} vs {}".format(cov_type, fit_intercept, t, np.array(est.predict_interval(X_test, alpha=alpha))[:, :, t], lr[t].predict_interval(X_test, alpha=alpha))

    def test_sum_vs_original(self):
        """ Testing that the summarized version gives the same results as the non-summarized."""
        np.random.seed(123)

        # 1-d y
        n = 100
        p = 1
        d = 5
        X_test = np.random.binomial(1, .5, size=(100, d))

        X = np.random.binomial(1, .8, size=(n, d))
        y = X[:, [0]*p] + (1*X[:, [0]] + 1)*np.random.normal(0, 1, size=(n,p))
        y = y.flatten()

        X1, X2, y1, y2, X_final_first, X_final_sec, y_sum_first, y_sum_sec, n_sum_first, n_sum_sec, var_first, var_sec = _summarize(X, y)
        X = np.vstack([X1, X2])
        y = np.concatenate((y1, y2))
        X_final = np.vstack([X_final_first, X_final_sec])
        y_sum = np.concatenate((y_sum_first, y_sum_sec))
        n_sum = np.concatenate((n_sum_first, n_sum_sec))
        var_sum = np.concatenate((var_first, var_sec))

        for cov_type in ['nonrobust', 'HC0', 'HC1']:
            for fit_intercept in [True, False]:
                for alpha in [.01, .05, .2]:
                    _compare_classes(OLS(fit_intercept=fit_intercept, fit_args={'cov_type':cov_type}).fit(X, y),
                        OLS(fit_intercept=fit_intercept, fit_args={'cov_type':cov_type}).fit(X_final, y_sum,
                        sample_weight=n_sum, var_weight=var_sum), X_test, alpha=alpha)
                    _compare_classes(StatsModelsOLS(fit_intercept=fit_intercept, fit_args={'cov_type':cov_type, 'use_t': False}).fit(X, y),
                        OLS(fit_intercept=fit_intercept, fit_args={'cov_type':cov_type}).fit(X_final, y_sum,
                        sample_weight=n_sum, var_weight=var_sum), X_test, alpha=alpha)

        # multi-d y
        n = 100
        for d in [1, 5]:
            for p in [1, 5]:
                X_test = np.random.binomial(1, .5, size=(100, d))

                X = np.random.binomial(1, .8, size=(n, d))
                y = X[:, [0]*p] + (1*X[:, [0]] + 1)*np.random.normal(0, 1, size=(n,p))

                X1, X2, y1, y2, X_final_first, X_final_sec, y_sum_first, y_sum_sec, n_sum_first, n_sum_sec, var_first, var_sec = _summarize(X, y)
                X = np.vstack([X1, X2])
                y = np.concatenate((y1, y2))
                X_final = np.vstack([X_final_first, X_final_sec])
                y_sum = np.concatenate((y_sum_first, y_sum_sec))
                n_sum = np.concatenate((n_sum_first, n_sum_sec))
                var_sum = np.concatenate((var_first, var_sec))

                for cov_type in ['nonrobust', 'HC0', 'HC1']:
                        for fit_intercept in [True, False]:
                            for alpha in [.01, .05, .2]:
                                _compare_classes(OLS(fit_intercept=fit_intercept, fit_args={'cov_type':cov_type}).fit(X, y),
                                    OLS(fit_intercept=fit_intercept, fit_args={'cov_type':cov_type}).fit(X_final, y_sum,
                                    sample_weight=n_sum, var_weight=var_sum), X_test, alpha=alpha)


    def test_dml_sum_vs_original(self):
        """ Testing that the summarized version of DML gives the same results as the non-summarized. """
        from econml.dml import LinearDMLCateEstimator
        from econml.inference import StatsModelsInference
        n = 100
        for d in [1, 5]:
            for p in [1, 5]:
                for cov_type in ['nonrobust', 'HC0', 'HC1']:
                    for alpha in [.01, .05, .2]:
                        X = np.random.binomial(1, .8, size=(n, d))
                        T = np.random.binomial(1, .5*X[:, 0]+.25, size=(n,))
                        true_effect = lambda x: np.hstack([x[:, [0]] + t for t in range(p)])
                        y = true_effect(X)*T.reshape(-1, 1) + X[:, [0]*p] + (1*X[:, [0]] + 1)*np.random.normal(0, 1, size=(n,p))
                        if p==1:
                            y = y.flatten()
                        X_test = np.random.binomial(1, .5, size=(100, d))

                        XT = np.hstack([X, T.reshape(-1, 1)])
                        X1, X2, y1, y2, X_final_first, X_final_sec, y_sum_first, y_sum_sec, n_sum_first, n_sum_sec, var_first, var_sec = _summarize(XT, y)
                        X = np.vstack([X1, X2])
                        y = np.concatenate((y1, y2))
                        X_final = np.vstack([X_final_first, X_final_sec])
                        y_sum = np.concatenate((y_sum_first, y_sum_sec))
                        n_sum = np.concatenate((n_sum_first, n_sum_sec))
                        var_sum = np.concatenate((var_first, var_sec))
                        first_half_sum = len(y_sum_first)
                        first_half = len(y1)

                        class SplitterSum:
                            def __init__(self):
                                return
                            def split(self, X, T):
                                return [(np.arange(0, first_half_sum), np.arange(first_half_sum, X.shape[0])), 
                                        (np.arange(first_half_sum, X.shape[0]), np.arange(0, first_half_sum))]

                        est = LinearDMLCateEstimator(model_y = LinearRegression(),
                                            model_t = LinearRegression(),
                                            n_splits=SplitterSum(),
                                            linear_first_stages=False,
                                            discrete_treatment=False).fit(y_sum, X_final[:, -1], X_final[:, :-1], None, sample_weight=n_sum,
                                            var_weight=var_sum, inference=StatsModelsInference(cov_type=cov_type))

                        class Splitter:
                            def __init__(self):
                                return
                            def split(self, X, T):
                                return [(np.arange(0, first_half), np.arange(first_half, X.shape[0])), 
                                        (np.arange(first_half, X.shape[0]), np.arange(0, first_half))]
                        
                        lr = LinearDMLCateEstimator(model_y = LinearRegression(),
                                            model_t = LinearRegression(),
                                            n_splits=Splitter(),
                                            linear_first_stages=False,
                                            discrete_treatment=False).fit(y, X[:, -1], X[:, :-1], None,
                                            inference=StatsModelsInference(cov_type=cov_type))

                        _compare_dml_classes(est, lr, X_test, alpha=alpha)

                        if p==1:
                            lr = LinearDMLCateEstimator(model_y = LinearRegression(),
                                                model_t = LinearRegression(),
                                                model_final = StatsModelsOLS(fit_intercept=False),
                                                n_splits=Splitter(),
                                                linear_first_stages=False,
                                                discrete_treatment=False).fit(y, X[:, -1], X[:, :-1], None,
                                                inference=StatsModelsInference(cov_type=cov_type, use_t=False))
                            
                            _compare_dml_classes(est, lr, X_test, alpha=alpha)
    
    def test_dml_sum_vs_original_lasso(self):
        """ Testing that the summarized version of DML gives the same results as the non-summarized when
        Lasso is used for first stage models. """
        from econml.dml import LinearDMLCateEstimator
        from econml.inference import StatsModelsInference
        from econml.utilities import WeightedModelWrapper
        n = 100
        for d in [1, 5]:
            for p in [1, 5]:
                for cov_type in ['nonrobust', 'HC0', 'HC1']:
                    for alpha in [.01, .05, .2]:
                        X = np.random.binomial(1, .8, size=(n, d))
                        T = np.random.binomial(1, .5*X[:, 0]+.25, size=(n,))
                        true_effect = lambda x: np.hstack([x[:, [0]] + t for t in range(p)])
                        y = true_effect(X)*T.reshape(-1, 1) + X[:, [0]*p] + (1*X[:, [0]] + 1)*np.random.normal(0, 1, size=(n,p))
                        if p==1:
                            y = y.flatten()
                        X_test = np.random.binomial(1, .5, size=(100, d))

                        XT = np.hstack([X, T.reshape(-1, 1)])
                        X1, X2, y1, y2, X_final_first, X_final_sec, y_sum_first, y_sum_sec, n_sum_first, n_sum_sec, var_first, var_sec = _summarize(XT, y)
                        X = np.vstack([X1, X2])
                        y = np.concatenate((y1, y2))
                        X_final = np.vstack([X_final_first, X_final_sec])
                        y_sum = np.concatenate((y_sum_first, y_sum_sec))
                        n_sum = np.concatenate((n_sum_first, n_sum_sec))
                        var_sum = np.concatenate((var_first, var_sec))
                        first_half_sum = len(y_sum_first)
                        first_half = len(y1)

                        class SplitterSum:
                            def __init__(self):
                                return
                            def split(self, X, T):
                                return [(np.arange(0, first_half_sum), np.arange(first_half_sum, X.shape[0])), 
                                        (np.arange(first_half_sum, X.shape[0]), np.arange(0, first_half_sum))]
                        
                        est = LinearDMLCateEstimator(model_y = WeightedModelWrapper(Lasso(alpha=0.01, fit_intercept=False)),
                                            model_t = WeightedModelWrapper(Lasso(alpha=0.01, fit_intercept=False)),
                                            n_splits=SplitterSum(),
                                            linear_first_stages=False,
                                            discrete_treatment=False).fit(y_sum, X_final[:, -1], X_final[:, :-1], None, sample_weight=n_sum,
                                            var_weight=var_sum, inference=StatsModelsInference(cov_type=cov_type))

                        class Splitter:
                            def __init__(self):
                                return
                            def split(self, X, T):
                                return [(np.arange(0, first_half), np.arange(first_half, X.shape[0])), 
                                        (np.arange(first_half, X.shape[0]), np.arange(0, first_half))]
                        
                        lr = LinearDMLCateEstimator(model_y = Lasso(alpha=0.01, fit_intercept=False),
                                            model_t = Lasso(alpha=0.01, fit_intercept=False),
                                            n_splits=Splitter(),
                                            linear_first_stages=False,
                                            discrete_treatment=False).fit(y, X[:, -1], X[:, :-1], None,
                                            inference=StatsModelsInference(cov_type=cov_type))

                        _compare_dml_classes(est, lr, X_test, alpha=alpha)

                        if p==1:
                            lr = LinearDMLCateEstimator(model_y = LinearRegression(),
                                                model_t = LinearRegression(),
                                                model_final = StatsModelsOLS(fit_intercept=False),
                                                n_splits=Splitter(),
                                                linear_first_stages=False,
                                                discrete_treatment=False).fit(y, X[:, -1], X[:, :-1], None,
                                                inference=StatsModelsInference(cov_type=cov_type, use_t=False))
                            
                            _compare_dml_classes(est, lr, X_test, alpha=alpha)