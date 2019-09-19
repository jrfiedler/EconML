import numpy as np
from econml.dml import LinearDMLCateEstimator
from sklearn.linear_model import LinearRegression
from econml.inference import StatsModelsInference
from econml.tests.test_statsmodels import _summarize
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
import os

def _coverage_profile(est, X_test, alpha, true_coef, true_effect):
    cov = {}
    coef_interval = est.coef__interval(alpha=alpha)
    cov['coef_cov'] = (true_coef >= coef_interval[0]) & (true_coef <= coef_interval[1])
    cov['coef_length'] = coef_interval[1] - coef_interval[0]
    effect_interval = est.effect_interval(X_test, alpha=alpha)
    true_eff = true_effect(X_test)
    cov['effect_cov'] = (true_eff >= effect_interval[0]) & (true_eff <= effect_interval[1])
    cov['effect_length'] = effect_interval[1] - effect_interval[0]
    return cov

def _append_coverage(key, coverage, est, X_test, alpha, true_coef, true_effect):
    cov = _coverage_profile(est, X_test, alpha, true_coef, true_effect)
    if key not in coverage:
        coverage[key] = {}
        for cov_key, value in cov.items():
            coverage[key][cov_key] = [value]
    else:
        for cov_key, value in cov.items():
            coverage[key][cov_key].append(value)

def _agg_coverage(coverage):
    mean_coverage_est = {}
    for key, cov_dict in coverage.items():
        mean_coverage_est[key] = {}
        for cov_key, cov_list in cov_dict.items():
            mean_coverage_est[key][cov_key] = np.mean(cov_list, axis=0)
    return mean_coverage_est

def plot_coverage(coverage, cov_key, d_list, d_x_list, p_list, cov_type_list, alpha_list, print_matrix=False):
    for d in d_list:
        for d_x in d_x_list:
            if d_x > d:
                continue
            for p in p_list:
                for cov_type in cov_type_list:
                    for alpha in alpha_list:
                        key = "d_{}_d_x_{}_p_{}_cov_type_{}_alpha_{}".format(d, d_x, p, cov_type, alpha)
                        if print_matrix:
                            print(coverage[key][cov_key])
                        plt.figure()
                        plt.title("{}_{}".format(key, cov_key))
                        plt.hist(coverage[key][cov_key].flatten())
                        if not os.path.exists('figures'):
                            os.makedirs('figures')
                        plt.savefig("figures/{}_{}.png".format(key, cov_key))
                        plt.close()

def monte_carlo():
    
    coverage_est = {}
    coverage_lr = {}
    n = 500
    d_list = [1, 10]
    d_x_list = [1, 5]
    p_list = [1, 5]
    n_exp = 10000
    cov_type_list = ['nonrobust', 'HC0', 'HC1'] 
    alpha_list = [.01, .05, .2]
    for d in d_list:
        for d_x in d_x_list:
            if d_x > d:
                continue
            for p in p_list:
                X_test = np.unique(np.random.binomial(1, .5, size=(100, d_x)), axis=0)
                for it in range(n_exp):
                    X = np.random.binomial(1, .8, size=(n, d))
                    T = np.random.binomial(1, .5*X[:, 0]+.25, size=(n,))
                    true_coef = np.hstack([np.arange(p).reshape(-1, 1), np.ones((p, 1)), np.zeros((p, d_x-1))])
                    true_effect = lambda x: np.hstack([np.ones((x.shape[0], 1)), x[:, :d_x]]) @ true_coef.T
                    y = true_effect(X)*T.reshape(-1, 1) + X[:, [0]*p] + (1*X[:, [0]] + 1)*np.random.normal(0, 1, size=(n,p))
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
                    for cov_type in cov_type_list:
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
                                            discrete_treatment=False).fit(y_sum,
                                                                        X_final[:, -1],
                                                                        X_final[:, :d_x],
                                                                        X_final[:, d_x:-1],
                                                                        sample_weight=n_sum,
                                                                        var_weight=var_sum,
                                                                        inference=StatsModelsInference(cov_type=cov_type))
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
                                            discrete_treatment=False).fit(y, X[:, -1], X[:, :d_x], X[:, d_x:-1],
                                                                        inference=StatsModelsInference(cov_type=cov_type))
                        for alpha in alpha_list:
                            key = "d_{}_d_x_{}_p_{}_cov_type_{}_alpha_{}".format(d, d_x, p, cov_type, alpha)
                            _append_coverage(key, coverage_est, est, X_test, alpha, true_coef, true_effect)
                            _append_coverage(key, coverage_lr, lr, X_test, alpha, true_coef, true_effect)

    agg_coverage_est = _agg_coverage(coverage_est)
    agg_coverage_lr = _agg_coverage(coverage_lr)
 
    plot_coverage(agg_coverage_est, 'coef_cov', d_list, d_x_list, p_list, cov_type_list, alpha_list)
    plot_coverage(agg_coverage_lr, 'coef_cov', d_list, d_x_list, p_list, cov_type_list, alpha_list)
    plot_coverage(agg_coverage_est, 'effect_cov', d_list, d_x_list, p_list, cov_type_list, alpha_list)
    plot_coverage(agg_coverage_lr, 'effect_cov', d_list, d_x_list, p_list, cov_type_list, alpha_list)

if __name__ == "__main__":
    monte_carlo()