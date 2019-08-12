import pandas as pd
import numpy as np
import math
from math import *
import matplotlib as plt
from scipy import stats
from random import sample

stock = pd.read_csv("DWDP.csv")
stock = stock[['open', 'high', 'low', 'close', 'vwap', 'changeOverTime']]
print(stock.head(5))

corr = stock.corr(method='pearson')

# The correlation
print(corr)

print('Change Over Time variable')
print('The population size is ', stock['changeOverTime'].size)
print('The population mean is ', stock['changeOverTime'].mean())
print('The population variance is', stock['changeOverTime'].var())

#### FIRST AUX VARIABLE: VWAP
# 1. Aux variable is VWAP (volume weighted price average)

aux_var = stock['vwap']

# 2. double sample: Perform double sampling
n = 67
n_i = 2*n # this is n'
phase_1_sample = stock.sample(n=n_i)
phase_2_sample = phase_1_sample.sample(n=n)
print('First Phase Sample:',phase_1_sample.head())
print('Second Phase Sample:',phase_2_sample.head())

# 3. Regression analysis x~y -> vwap ~ changeOverTime: Perform a diagnostic analysis to determine if x and y have a linear
# relationship and fitted line goes through the origin based on the sample data. Do regression analysis y ∼ x.

from statsmodels.api import OLS
x = phase_2_sample[['vwap']]
y = phase_2_sample[['changeOverTime']]
reg = OLS(y, x).fit()
reg.summary()

yi_sum = phase_2_sample['changeOverTime'].sum()
yi_sum
xi_sum = phase_2_sample['vwap'].sum()
xi_sum
r = yi_sum / xi_sum
r
print('ratio estimator (r) =', r)

# 5. estimate your parameter of interest by ratio estimator: Estimate your parameter of interest by
# ratio estimator. Estimate its variance and standard deviation.
N = 368
t_hat_x = N/n_i*xi_sum
t_hat_x
t_hat_r = r*t_hat_x
t_hat_r
mu_hat_x = t_hat_r/N
print('The ratio estimator of the mean of x (= vwap)', mu_hat_x)

# variance
xi = phase_2_sample['vwap']
yi = phase_2_sample['changeOverTime']
both_xy = pd.DataFrame(columns = ['vwap','changeOverTime'])
both_xy['vwap'] = phase_2_sample['vwap']
both_xy['changeOverTime'] = phase_2_sample['changeOverTime']
both_xy.head()
s_2 = yi.var()
var_hat_t_hat = 0;
for index, row in both_xy.iterrows():
    var_hat_t_hat = (N*(N-n_i)* s_2/n_i + math.pow(N, 2)*(n_i-n)/(n_i*n*(n-1))*
                             math.pow((row['changeOverTime'] - r*row['vwap']),2))

#std_dev_t_hat = math.sqrt(var_hat_t_hat)

var_hat_mu_hat = var_hat_t_hat/math.pow(N,2)
std_dev_mu_hat = math.sqrt(var_hat_mu_hat)

print('estimated variance, var_hat_mu_hat =', var_hat_mu_hat)
print('standard deviation ~ mu_hat:', std_dev_mu_hat)



# Repeating with second auxiliary variable.
# SECOND AUX VARIABLE: CLOSE

aux_var = stock['close']

#2 double sample
n = 67
n_i = 2*n # this is n'
phase_1_sample = stock.sample(n=n_i)
phase_2_sample = phase_1_sample.sample(n=n)
print('First Phase Sample:',phase_1_sample.head())
print('Second Phase Sample:',phase_2_sample.head())

#3 Reg analysis x~y -> vwap ~ changeOverTime
from statsmodels.api import OLS
x = phase_2_sample[['close']]
y = phase_2_sample[['changeOverTime']]
reg = OLS(y, x).fit()
reg.summary()

#Reg Estimator
yi_sum = phase_2_sample['changeOverTime'].sum()
yi_sum
xi_sum = phase_2_sample['close'].sum()
xi_sum
r = yi_sum / xi_sum
r
print('ratio estimator (r) =', r)

#5 estimate your parameter of interest by ratio estimator.
N = 368
t_hat_x = N/n_i*xi_sum
t_hat_x
t_hat_r = r*t_hat_x
t_hat_r
mu_hat_x = t_hat_r/N
print('The ratio estimator of the mean of x (= close)', mu_hat_x)

#variance
xi = phase_2_sample['close']
yi = phase_2_sample['changeOverTime']
both_xy = pd.DataFrame(columns = ['close','changeOverTime'])
both_xy['close'] = phase_2_sample['close']
both_xy['changeOverTime'] = phase_2_sample['changeOverTime']
both_xy.head()
s_2 = yi.var()
var_hat_t_hat = 0;
for index, row in both_xy.iterrows():
    var_hat_t_hat = (N*(N-n_i)* s_2/n_i + math.pow(N, 2)*(n_i-n)/(n_i*n*(n-1))*
                             math.pow((row['changeOverTime'] - r*row['close']),2))
#std_dev_t_hat = math.sqrt(var_hat_t_hat)

var_hat_mu_hat = var_hat_t_hat/math.pow(N,2)
std_dev_mu_hat = math.sqrt(var_hat_mu_hat)

print('estimated variance, var_hat_mu_hat =', var_hat_mu_hat)
print('standard deviation ~ mu_hat:', std_dev_mu_hat)



