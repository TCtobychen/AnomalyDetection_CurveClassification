# import needed packages
#-----------------------
from scipy.fftpack import fft
import math
import numpy  as np
import pandas as pd
import statsmodels.tsa.holtwinters as st 
import matplotlib.pyplot as plt 

from sklearn        import linear_model
from scipy.optimize import fmin_l_bfgs_b

# bring in the passenger data from HW4 to test the function against R output
#---------------------------------------------------------------------------

'''

low_bound=-1000
high_bound=1000
samples=10000
x=[]
data = pd.read_csv("AWSresults.csv")

sec_control = 1
StdOutlierx = []
StdOutliery = []
OutlierCnt = 0
for i in range(len(data['anomaly_score'])):
    if data['anomaly_score'][i] == 1:
        StdOutlierx.append(i)
        StdOutliery.append(data['value'][i])
        OutlierCnt += 1
ExpectRatio = OutlierCnt / len(data['value'])


diff=(high_bound-low_bound)/samples
temp=low_bound
for i in range(samples):
    x.append(temp)
    temp+=diff
y=[]
for i in x:
    y.append(math.sin(i)*10)
for i in range(samples-20,samples):
    y[i]+=10*np.random.random()

value=[]
for i in data['value']:
    value.append(i)
Len = len(value)
SavedSamples = Len//2
InitialTake = Len//100
tsA=value[:-SavedSamples]
anomaly_train = data['anomaly_score'][:-SavedSamples]
#plt.plot(value)

'''


# define main function [holtWinters] to generate retrospective smoothing/predictions
#-----------------------------------------------------------------------------------
 

def naive_sigma(ts, mul, mean = None, sigma = None):
    if sigma == None:
        sigma = np.std(ts)
    if mean == None:
        mean = np.mean(ts)
    feature_outlierc = []
    for i in range(len(ts)):
        if abs(ts[i] - mean) > mul * sigma:
            feature_outlierc.append(1)
        else:
            feature_outlierc.append(0)

    return np.array(feature_outlierc)

def sigma_diff(ts, mean):
    feature_diff = []
    if mean == None:
        mean = np.mean(ts)
    for i in range(len(ts)):
        feature_diff.append(ts[i]-mean)
    return np.array(feature_diff)

#naive_sigma(tsA, 3)


def holtWinters(ts, anomaly_cnt, p = 288, sp = 100, ahead = 100, mtype = 'additive', alpha = None, beta = None, gamma = None):
    '''HoltWinters retrospective smoothing & future period prediction algorithm 
       both the additive and multiplicative methods are implemented and the (alpha, beta, gamma)
       parameters have to be either all user chosen or all optimized via one-step-ahead prediction MSD
       initial (a, b, s) parameter values are calculated with a fixed-period seasonal decomposition and a
       simple linear regression to estimate the initial level (B0) and initial trend (B1) values
    @params:
        - ts[list]:      time series of data to model
        - p[int]:        period of the time series (for the calculation of seasonal effects)
        - sp[int]:       number of starting periods to use when calculating initial parameter values
        - ahead[int]:    number of future periods for which predictions will be generated
        - mtype[string]: which method to use for smoothing/forecasts ['additive'/'multiplicative']
        - alpha[float]:  user-specified level  forgetting factor (one-step MSD optimized if None)
        - beta[float]:   user-specified slope  forgetting factor (one-step MSD optimized if None)
        - gamma[float]:  user-specified season forgetting factor (one-step MSD optimized if None)
    @return: 
        - alpha[float]:    chosen/optimal level  forgetting factor used in calculations
        - beta[float]:     chosen/optimal trend  forgetting factor used in calculations
        - gamma[float]:    chosen/optimal season forgetting factor used in calculations
        - MSD[float]:      chosen/optimal Mean Square Deviation with respect to one-step-ahead predictions
        - params[tuple]:   final (a, b, s) parameter values used for the prediction of future observations
        - smoothed[list]:  smoothed values (level + trend + seasonal) for the original time series
        - predicted[list]: predicted values for the next @ahead periods of the time series
    sample calls:
        results = holtWinters(ts, 12, 4, 24, 'additive')
        results = holtWinters(ts, 12, 4, 24, 'multiplicative', alpha = 0.1, beta = 0.2, gamma = 0.3)'''

    a, b, s = _initValues(mtype, ts, p, sp)

    if alpha == None or beta == None or gamma == None:
        ituning   = [0.1, 0.1, 0.1]
        ibounds   = [(0,1), (0,1), (0,1)]
        optimized = fmin_l_bfgs_b(_MSD, ituning, args = (mtype, ts, p, a, b, s[:]), bounds = ibounds, approx_grad = True)
        alpha, beta, gamma = optimized[0]

    MSD, params, smoothed, back_predict, sigma, smoothdiff = _expSmooth(mtype, ts, p, a, b, s[:], alpha, beta, gamma)
    holtwinter_diff = []
    for i in range(len(ts)):
        holtwinter_diff.append(ts[i]-back_predict[i])

    mul = get_sigma_mul(ts, back_predict, sigma, anomaly_cnt)
    holtwinter_c = []
    for i in range(len(ts)):
        if abs(ts[i]-back_predict[i]) > mul * sigma[i % 24]:
            holtwinter_c.append(1)
        else:
            holtwinter_c.append(0)
    predicted = _predictValues(mtype, p, ahead, params, ts)
    #plt.plot(ts)
    #plt.plot(params[0] + params[2])
    #plt.show()
    '''results = []
    for i in ts:
        results.append(i)
    for i in predicted:
        results.append(i)
    plt.plot(results)
    plt.show()'''

    return np.array(holtwinter_diff), np.array(holtwinter_c), np.array(predicted), np.array(smoothdiff), np.array(smoothed), mul, sigma, (params, alpha, beta, gamma)


def get_sigma_mul(ts, back_predict, sigma, expectednum):
    '''To get a suitable multiply of sigma to satisfy the expected ratio of anomaly'''
    '''Choosing from 4 * sigma to 10 * sigma'''

    start = 4
    end = 14
    mid = 9
    def get_cnt(ts, back_predict, sigma, mul):
        sigcnt = 0
        for i in range(len(ts)):
            if (ts[i]-back_predict[i]) > mul * sigma[i%24]:
                sigcnt += 1
            '''else:
                sum = 0
                for j in range(max(0,i-2),min(i+3,len(ts))):
                    sum += (ts[j]-back_predict[j])
                sum /= 5
                if sum > sec_control * sigma[i%24]:
                    sigcnt += 1'''
        return  sigcnt
    for i in range(20):
        if get_cnt(ts, back_predict, sigma, mid) > expectednum:
            start = mid
        else:
            end = mid
        mid = (start + end) / 2
    return mid


def _initValues(mtype, ts, p, sp):
    '''subroutine to calculate initial parameter values (a, b, s) based on a fixed number of starting periods'''

    initSeries = pd.Series(ts[:p*sp])

    def rolling_mean(series, window):
        result=[]
        tempmean=0;
        for i in range(window):
            tempmean=(tempmean*i+series[i])/(i+1)
            result.append(tempmean)
        for i in range(window,len(series)):
            tempmean+=(series[i]-series[i-window])/(window)
            result.append(tempmean)
        return result

    if mtype == 'additive':
        rawSeason  = initSeries - rolling_mean(initSeries, window = p)
        initSeason = [np.nanmean(rawSeason[i::p]) for i in range(p)]
        initSeason = pd.Series(initSeason) - np.mean(initSeason)
        deSeasoned = [initSeries[v] - initSeason[v % p] for v in range(len(initSeries))]
    else:
        rawSeason  = initSeries / rolling_mean(initSeries, window = p)
        initSeason = [np.nanmean(rawSeason[i::p]) for i in range(p)]
        initSeason = pd.Series(initSeason) / math.pow(np.prod(np.array(initSeason)), 1/p)
        deSeasoned = [initSeries[v] / initSeason[v % p] for v in range(len(initSeries))]

    lm = linear_model.LinearRegression()
    lm.fit(pd.DataFrame({'time': [t+1 for t in range(len(initSeries))]}), pd.Series(deSeasoned))
    return float(lm.intercept_), float(lm.coef_), list(initSeason)

def _MSD(tuning, *args):
    '''subroutine to pass to BFGS optimization to determine the optimal (alpha, beta, gamma) values'''

    predicted, diff = [], []
    mtype     = args[0]
    ts, p     = args[1:3]
    Lt1, Tt1  = args[3:5]
    St1       = args[5][:]
    alpha, beta, gamma = tuning[:]

    for t in range(len(ts)):

        if mtype == 'additive':
            Lt = alpha * (ts[t] - St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
            St = gamma * (ts[t] - Lt)         + (1 - gamma) * (St1[t % p])
            predicted.append(Lt1 + Tt1 + St1[t % p])
            diff.append(abs(ts[t] - predicted[t]))
        else:
            Lt = alpha * (ts[t] / St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
            St = gamma * (ts[t] / Lt)         + (1 - gamma) * (St1[t % p])
            predicted.append((Lt1 + Tt1) * St1[t % p])

        Lt1, Tt1, St1[t % p] = Lt, Tt, St
    diff.sort()
    diff = diff[:-5]

    return sum([diff[t]**2 for t in range(len(diff))])/len(diff)

def _expSmooth(mtype, ts, p, a, b, s, alpha, beta, gamma):
    '''subroutine to calculate the retrospective smoothed values and final parameter values for prediction'''

    smoothed = []
    diff = []
    Lt1, Tt1, St1 = a, b, s[:]

    for t in range(len(ts)):

        if mtype == 'additive':
            Lt = alpha * (ts[t] - St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
            St = gamma * (ts[t] - Lt)         + (1 - gamma) * (St1[t % p])
            smoothed.append(Lt1 + Tt1 + St1[t % p])
            diff.append(ts[t]-smoothed[t])
        else:
            Lt = alpha * (ts[t] / St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
            St = gamma * (ts[t] / Lt)         + (1 - gamma) * (St1[t % p])
            smoothed.append((Lt1 + Tt1) * St1[t % p])
            diff.append(ts[t]-smoothed[t])

        Lt1, Tt1, St1[t % p] = Lt, Tt, St

    MSD = sum([((ts[t] - smoothed[t]))**2 for t in range(len(smoothed))])/len(smoothed)
    sigma = []
    temp = []
    for start in range(24):
        for k in range(len(diff)//24):
            temp.append(diff[start + k * 24])
        sigma.append(np.std(temp))
        temp.clear()

    '''Lt = 0
    cnt = 0
    for i in range(len(smoothed)):
        if abs(ts[i]-smoothed[i]) < 3 * sigma[i%24]:
            Lt += smoothed[i]
            cnt += 1
    Lt /= cnt'''

    def BackPredict(length, Lt, Tt, St):

        return [(Lt+(t+1)*Tt+St[(t+len(ts))%p]) for t in range(-length,0)]

    back_predict = BackPredict(len(smoothed), Lt, Tt1, St1)

    return MSD, (Lt, Tt1, St1), smoothed, back_predict, sigma, diff

def _predictValues1(mtype, p, ahead, params, ts, alpha, beta, gamma):
    '''subroutine to generate predicted values @ahead periods into the future'''

    Lt1, Tt1, St1 = params
    results = [ts[len(ts)-1]]
    if mtype == 'additive':
        length = len(ts)
        for t in range(length, length+ahead):
            Lt = alpha * (ts[t-length] - St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
            Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
            St = gamma * (ts[t-length] - Lt)         + (1 - gamma) * (St1[t % p])
            results.append(Lt1 + Tt1 + St1[t % p])
            Lt1, Tt1, St1[t % p] = Lt, Tt, St
        return results
    else:
        return [(Lt + (t+1)*Tt) * St[t % p] for t in range(ahead)]

def _predictValues(mtype, p, ahead, params, ts):

    Lt, Tt, St = params

    if mtype == 'additive':
        return [(Lt+(t+1)*Tt+St[(t+len(ts))%p]) for t in range(ahead)]
    else:
        return [(Lt + (t+1)*Tt) * St[(t+len(ts)) % p] for t in range(ahead)]
def wholepredict(mtype, p, ahead, params,alpha, beta, gamma):
    Lt, Tt, St = params
    predict_result=[]
    if mtype == 'additive':
        for t in range(-len(ts)+1,ahead):
            predict_result.append(Lt+(t+1)*Tt+St[t % p])
        #plt.plot(predict_result)
    else:
        for t in range(-len(ts)+1,ahead):
            predict_result.append((Lt + (t+1)*Tt) * St[t % p])
        #plt.plot(predict_result)

# print out the results to check against R output
#------------------------------------------------
