#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Chapters 27.2-3 from
http://www.kevinsheppard.com/images/0/09/Python_introduction.pdf

CAY risk factor:
http://faculty.haas.berkeley.edu/lettau/data/cay_q_13Q3.txt

Fama-French risk factors:
http://www.kevinsheppard.com/images/0/0b/FamaFrench.zip

GMM estimator is located here:
https://github.com/khrapovs/MyGMM

Fama-McBeth estimation library is here:
https://github.com/khrapovs/famamcbeth

"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pylab as plt
import seaborn as sns

from famamcbeth import FamaMcBeth


def import_data():
    parse = lambda x: dt.datetime.strptime(x, '%Y%m')
    date_name = 'date'
    factor_names = ['VWMe', 'SMB', 'HML']
    rf_name = 'RF'
    data = pd.read_csv('../data/FamaFrench.csv', index_col=date_name,
                     parse_dates=[date_name], date_parser=parse)

    riskfree = data[[rf_name]].values
    factors = data[factor_names].values
    # Augment factors with the constant
    factors = np.hstack((np.ones_like(riskfree), factors))
    portfolios = data[data.columns - factor_names - [rf_name]].values
    excess_ret = portfolios - riskfree

    return factors, excess_ret


def import_cay():
    import calendar
    calendar.monthrange(2002,1)
    parse = lambda x: dt.datetime.strptime(x, '%Y\:%q')
    date_name = 'date'

    def parse(value):
        year = int(value[:4])
        month = 3*int(value[5:])
        day = calendar.monthrange(year, month)[1]
        return dt.datetime(year, month, day)

    cay = pd.read_csv('../data/cay_q_13Q3.csv', index_col=date_name,
                       parse_dates=[date_name], date_parser=parse)[['cay']]

    parse = lambda x: dt.datetime.strptime(x, '%Y%m')
    date_name = 'date'
    rf_name = 'RF'
    data = pd.read_csv('../data/FamaFrench.csv', index_col=date_name,
                     parse_dates=[date_name], date_parser=parse)
    ff_factors = data.resample('Q').mean()
    data = pd.merge(cay, ff_factors, left_index=True, right_index=True)

    factor_names = ['cay', 'VWMe', 'SMB', 'HML']
    riskfree = data[[rf_name]].values
    factors = data[factor_names].values
    # Augment factors with the constant
    factors = np.hstack((np.ones_like(riskfree), factors))
    portfolios = data[data.columns - factor_names - [rf_name]].values
    excess_ret = portfolios - riskfree

    return factors, excess_ret


def estimate(factors, excess_ret):

    model = FamaMcBeth(factors, excess_ret)
    (param, gamma_rsq, gamma_rmse, theta_rsq, theta_rmse) \
        = model.two_step_ols()
    alpha, beta, gamma = model.convert_theta_to2d(param)

    kernel = 'Bartlett'
    band = 3
    jstat, jpval = model.jtest(param, kernel=kernel, band=band)
    tstat = model.alpha_beta_gamma_tstat(param, kernel=kernel, band=band)
    alpha_tstat, beta_tstat, gamma_tstat = tstat

    print('OLS results:')
    print(gamma)
    print(gamma_tstat)
    print('J-stat = %.2f, p-value = %.2f\n' % (jstat, jpval))

    method = 'L-BFGS-B'
#    method = 'Nelder-Mead'
#    method = 'basin'
    res = model.gmmest(param, kernel=kernel, band=band, method=method)
    param_final = model.convert_theta_to2d(res.theta)
    alpha_final, beta_final, gamma_final = param_final
    tstat_final = model.convert_theta_to2d(res.tstat)
    alpha_tstat, beta_tstat, gamma_tstat = tstat_final

    print('GMM results:')
    print(gamma_final)
    print(gamma_tstat)
    jstat, jpval = model.jtest(res.theta, kernel=kernel, band=band)
    print('J-stat = %.2f, p-value = %.2f' % (jstat, jpval))

    ret_realized = model.get_realized_ret()
    ret_predicted = model.get_predicted_ret(res.theta)

    plt.scatter(ret_realized, ret_predicted)
    x = np.linspace(*plt.gca().get_xlim())
    plt.gca().plot(x, x)
    plt.xlabel('Realized')
    plt.ylabel('Predicted')
    plt.show()


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)
    sns.set_context('notebook')

    factors, excess_ret = import_data()
    estimate(factors, excess_ret)

    factors, excess_ret = import_cay()
    estimate(factors, excess_ret)
