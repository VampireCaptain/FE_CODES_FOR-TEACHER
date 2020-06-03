import numpy as np
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt
symbols = ['399300']
indexes = pd.date_range('2020-01-01', '2020-06-07')
indexes = indexes.map(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
data = pd.DataFrame(index=indexes)
for sym in symbols:
    k_d = ts.get_k_data(sym, '2020-01-01', ktype='D')
    k_d.set_index('date', inplace=True)
    data[sym] = k_d['close']
data = data.dropna()
(data / data.iloc[0] * 100).plot(figsize=(8, 5))
plt.xticks(rotation=45)
plt.style.use('ggplot')
rets = np.log(data / data.shift(1))
rets.mean() * 252
rets.corr()
import seaborn as sns
sns.heatmap(rets.corr(),annot=True,cmap='rainbow',linewidths=1.0,annot_kws={'size':8})
plt.xticks(rotation=45)
plt.yticks(rotation=45)
rets = np.log(data / data.shift(1))
rets.mean() * 252
rets.cov()*252
sns.heatmap(rets.cov(),annot=True,cmap='rainbow',linewidths=1.0,annot_kws={'size':8})
plt.xticks(rotation=45)
plt.yticks(rotation=45)
noa = 8
weights = np.random.random(noa)
weights /= np.sum(weights)
weights
np.dot(weights.T, np.dot(rets.cov() * 252, weights))
np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
prets = []
pvols = []
for p in range(50000):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(np.sum(rets.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))))

prets = np.array(prets)
pvols = np.array(pvols)

plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets, c=prets / pvols, marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
def statistics(weights):
    """
    Return portfolio statistics
    :param weights: weights for different securities in portfolio
    :return:
    pret:float
    expected portfolio return
    pvol:float
    expected portfolio volatility
    pret/pvol:float
    Sharpe ratio for rf=2.0970%
    """
    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])
import scipy.optimize as sco
def min_func_sharpe(weights):
    return -statistics(weights)[2]
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
noa * [1. / noa, ]
%%time
opts = sco.minimize(min_func_sharpe, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)
opts
opts['x'].round(9)
statistics(opts['x'].round(9))
def min_func_variance(weights):
    return statistics(weights)[1]**2
optv = sco.minimize(min_func_variance, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)
optv
optv['x'].round(9)
statistics(optv['x']).round(3)
def min_func_port(weights):
    return statistics(weights)[1]
trets = np.linspace(0.0, 0.5, 200)
tvols = []
bnds = tuple((0, 1) for x in weights)
for tret in trets:
    cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = sco.minimize(min_func_port, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

plt.figure(figsize=(8,4))
# random portfolio composition
plt.scatter(pvols,prets,c=prets/pvols,marker='o')
# efficient frontier
plt.scatter(tvols,trets,c=trets/tvols,marker='x')
# portfolio with highest Sharpe ratio
plt.plot(statistics(opts['x'])[1],statistics(opts['x'])[0],'r*',markersize=15.0)
# minimum variance portfolio
plt.plot(statistics(optv['x'])[1],statistics(optv['x'])[0],'y*',markersize=15.0)
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
max_index = RandomPortfolios.Sharpe.idxmax()
RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
x = RandomPortfolios.loc[max_index,'Volatility']
y = RandomPortfolios.loc[max_index,'Returns']
plt.scatter(x, y, color='red')
plt.text(np.round(x,4),np.round(y,4),(np.round(x,4),np.round(y,4)),ha='left',va='bottom',fontsize=10)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
p=0.1490;K=3.900;ETF0=3.978;index0=3983.65
indext=np.linspace(3000,5000,500)
ETFt=indext*ETF0/index0
return_ETF=1e4*(ETFt-ETF0)
return_put=1*1e4*(np.maximum(K-ETFt,0)-p)
rp=return_ETF+return_put
plt.plot(indext,return_ETF,'--',label='沪深300组合')
plt.plot(indext,return_put,'--',label='沪深300ETF认沽期权多头')
plt.plot(indext,rp,label='买入保护看跌期权组合收益')
plt.xlabel('沪深300点数')
plt.ylabel('组合收益金额')
plt.title('买入保护看跌期权组合收益图')
plt.legend()
plt.grid()
plt.show()