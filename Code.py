# -*- coding: utf-8 -*-

#################################################
# Jegadeesh & Titman (1993) Momentum Strategies #
#                    Sep 2018                   #  
#            Cheng Xie Fudan University         #
#################################################

import pandas as pd
import numpy as np
from numpy import *
from pandas.tseries.offsets import *
from scipy import stats

#########################################################################
# Read data of stock codes, dates, returns and market value of equity   #
# in Chinese A share market during the period from Jan 2000 to Dec 2015 # 
#########################################################################

#	Designate the path of data.
data_path = 'E:/data.csv'

astock_m = pd.DataFrame.from_csv(data_path)

# Line up date to be end of month.
astock_m['date'] = pd.to_datetime(astock_m['date'])
astock_m['date'] = astock_m['date'] + MonthEnd(0)

# Label the size.
astock_m['me'] = astock_m['me'].fillna(0)
astock_m['size'] = astock_m.groupby('date')['me'].transform(lambda x: pd.qcut(x, 3, labels = False))

##############################################################################
# Create momentum portfolio based upon the past (J) month compounded returns #
##############################################################################

#	Designate the span of the formation period J and the span of the holding period K.
J = 24
K = 36

tmp_astock = astock_m[['stock_code', 'date', 'ret']].sort_values(['stock_code', 'date']).set_index('date')
tmp_astock['logret'] = np.log(1 + tmp_astock['ret'])

#	Calculate the cumulative return of the formation period.
umd = tmp_astock.groupby(['stock_code'])['logret'].rolling(J, min_periods = J).sum().reset_index()
umd['cumret'] = np.exp(umd['logret']) - 1
umd = umd.dropna(axis = 0, subset = ['cumret'])

#	Grant each stock a rank at different time based upon the cumulative return of the formation period. And 1 = the lowest/losers, 10 = the highest/winners.
umd['rank_1'] = umd.groupby('date')['cumret'].transform(lambda x: pd.qcut(x, 10, labels = False))
umd.rank_1 = umd.rank_1.astype(int)
umd['rank_1'] = umd['rank_1'] + 1
umd['rank'] = umd['rank_1']
del umd['rank_1']

#	Specify the start and the end of the holding period.
umd['hold_start'] = umd['date'] + MonthBegin(1)
umd['hold_end'] = umd['date'] + MonthEnd(K)
umd = umd.rename(columns = {'date': 'form_date'})
umd = umd[['stock_code', 'form_date', 'rank', 'hold_start', 'hold_end']]

#	Match the data.
port = pd.merge(astock_m[['stock_code', 'date', 'ret', 'size']], umd, on = ['stock_code'], how = 'inner')
port = port[(port['hold_start'] <= port['date']) & (port['date'] <= port['hold_end'])]
umd2 = port.sort_values(by = ['date', 'rank', 'form_date', 'stock_code','size'])

#	The average of returns for each form_date, rank, date, size.
umd3 = umd2.groupby(['date', 'rank', 'form_date', 'size'])['ret'].mean().reset_index()

#	The average of returns for each date, rank, size.
umd4 = umd3.groupby(['date', 'rank', 'size'])['ret'].mean().reset_index()
umd4 = umd4.sort_values(by = ['size', 'rank'])

##############################################################################
#                          Statistical summary                               #
##############################################################################

umd4['rank_0'] = umd4['size'].map({0.0: 'small', 1.0: 'medium', 2.0: 'large'}) + list(np.array(['-']*len(umd4))) + umd4['rank'].map(str)
del umd4['rank']
del umd4['size']
umd4 = umd4.rename(columns = {'rank_0':'rank'})

#	Summarize the average return for different combinations of size and rank.
out_1 = umd4.groupby(['rank'])['ret'].describe()[['count', 'mean', 'std']]

umd5 = umd4.pivot(index = 'date', columns = 'rank', values = 'ret')
umd5 = umd5.add_prefix('port')
umd5 = umd5.rename(columns = {'portsmall-1':'small_losers', 'portsmall-10':'small_winners', 'portmedium-1':'medium_losers', 'portmedium-10':'medium_winners', 'portlarge-1':'large_losers', 'portlarge-10':'large_winners'})
for k in ['small', 'medium', 'large']:
    umd5[k + '_long_short'] = umd5[k + '_winners'] - umd5[k + '_losers']
    umd5[k + '_1+losers'] = 1 + umd5[k + '_losers']
    umd5[k + '_1+winners'] = 1 + umd5[k + '_winners']
    umd5[k + '_1+long_short'] = 1 + umd5[k + '_long_short']
    umd5[k + '_cumret_winners'] = umd5[k + '_1+winners'].cumprod() - 1
    umd5[k + '_cumret_losers'] = umd5[k + '_1+losers'].cumprod() - 1
    umd5[k + '_cumret_long_short'] = umd5[k + '_1+long_short'].cumprod() - 1
    out_2 = umd5[[k + '_winners', k + '_losers', k + '_long_short']].mean().to_frame()
    out_2 = out_2.rename(columns = {0:'mean'}).reset_index()
    t_losers = pd.Series(stats.ttest_1samp(umd5[k + '_losers'], 0.0)).to_frame().T
    t_winners = pd.Series(stats.ttest_1samp(umd5[k + '_winners'], 0.0)).to_frame().T
    t_long_short = pd.Series(stats.ttest_1samp(umd5[k + '_long_short'], 0.0)).to_frame().T
    t_losers['rank'] = k + '_losers'
    t_winners['rank'] = k + '_winners'
    t_long_short['rank'] = k + '_long_short'
    out_3 = pd.concat([t_winners, t_losers, t_long_short]).rename(columns = {0:'t-stat', 1:'p-value'})
    out_4 = pd.merge(out_2, out_3, on = ['rank'], how = 'inner')
    out_1.to_csv(k + '_out_1.csv')
    out_4.to_csv(k + '_out_2.csv')
