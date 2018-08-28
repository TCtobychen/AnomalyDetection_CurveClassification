import numpy as np 
import math
import matplotlib.pyplot as plt 
import time
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kurtosis
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fftpack import fft
from scipy.stats import skew

def _3_sigma(ts):
	mean, sigma = np.median(ts), np.std(ts)
	anomaly = []
	for v in ts:
		if abs(v - mean) > 3 * sigma:
			anomaly.append(-1)
		else:
			anomaly.append(1)
	return anomaly

def normalize(value):
	value = np.array(value).reshape(-1, 1)
	scaler = MinMaxScaler().fit(value)
	value = scaler.transform(value).ravel()
	return value

def get_abs_diff(ts, season, trend):
	ans, diff = 0, []
	for i in range(len(ts)):
		diff.append(ts[i] - season[i] - trend)
		ans += abs(diff[i])
	return (ans, diff)

def get_seasonal_diff(ts, T = 288, _normalize = True):
	# return best fitted seasonal_diff, season series, absolute diff and the average trend used. 
	# ----------------------------------------------------------------------------
	ts = np.array(ts)
	ts = pre_process(ts, _normalize = _normalize)
	res = seasonal_decompose(ts, freq = T)
	start, end = min(ts), max(ts)
	season = res.seasonal
	_abs, diff = [], []
	for j in range(100):
		temp0, temp1 = get_abs_diff(ts, season, start + j * (end - start) / 100)
		_abs.append(temp0)
		diff.append(temp1)
	ind, _min = 0, len(ts) * max(ts)
	for i in range(len(_abs)):
		if _abs[i] < _min:
			_min = _abs[i]
			ind = i
	return (diff[ind], season, _abs[ind], start + (ind - 1) * (end - start) / 100)

def process_slice(ts):
	value = ts
	anomaly = _3_sigma(ts)
	start, end, i = 0, 0, 0
	while i < len(ts):
		if anomaly[i] == 1:
			start, end = i, i
			i += 1
		else:
			j = i
			while (j < (len(ts))) and (anomaly[j] == -1):
				j += 1
			if j < len(ts):
				end = j
			normal = (value[start] + value[end])/2
			for k in range(i, j):
				value[k] = normal
			start, end = j+1, j+1
			i = j + 1
	return np.array(value)

def pre_process(ts, slice_len = 200, _normalize = True):
	start = 0
	while start < len(ts) - slice_len:
		ts[start: start + slice_len] = process_slice(ts[start: start + slice_len])
		start = start + slice_len
	ts[start: len(ts)] = process_slice(ts[start: len(ts)])
	if _normalize == True:
		ts = normalize(ts)
	return ts

def Apen(U, m = 2, r = None):
    if r == None:
	    r = 1 * np.std(U)
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    def _phi(m):
        s = time.time()
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        print(time.time() - s)
        s = time.time()
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        print(time.time() - s)
        return (N - m + 1.0)**(-1) * sum(np.log(C))
    N = len(U)
    return abs(_phi(m + 1) - _phi(m))
#print(ApEn([1,2,1,2,1,2,1,2]))

def Spen(L, m = 2, r = None):
    N = len(L)
    A, B = 0.0, 0.0
    if r == None:
    	r = np.std(L)
    xmi = np.array([L[i:i+m] for i in range(N-m)])
    xmj = np.array([L[i:i+m] for i in range(N-m+1)])

    B = np.sum([np.sum(np.abs(xmii-xmj).max(axis=1) <= r)-1 for xmii in xmi])

    m += 1
    xm = np.array([L[i:i+m] for i in range(N-m+1)])
        
    A = np.sum([np.sum(np.abs(xmi-xm).max(axis=1) <= r)-1 for xmi in xm])

    return -np.log(A/B)
#print(sampen([1,2,1,2,1,2,1,2]))

def Skew(x):
	return skew(x)
def Kurtosis(x):
	return kurtosis(x)
def Mean(x):
	return np.mean(x)
def Median(x):
	return np.median(x)
def Std(x):
	return np.std(x)
def Abs_change(x):
	s = 0
	for i in range(len(x)-1):
		s += abs(x[i+1]-x[i])
	return s / (len(x)-1)
def Linear(x):
	N = len(x)
	X = np.linspace(0,N-1,N)
	res = np.polyfit(X, x, 1)
	return res[0]
#print(Linear([1,2,3,4]))
def Zero_cnt(ts, ratio = 0.1):
	N, cnt = len(ts), 0.0
	for i in range(N):
		if ts[i] < ratio:
			cnt += 1
	return cnt / N
def Zero_max(ts, ratio = 0.1):
	N, now, max_ans = len(ts), 0.0, 0.0
	for i in range(N):
		if ts[i] < ratio:
			now += 1
		else:
			max_ans = max(max_ans, now)
			now = 0
	return max_ans / N  
#print(Zero_max([0,0,0.5,0.5,0,0.3]))

def Top_cnt(ts, ratio = 0.9):
	N, cnt = len(ts), 0.0
	for i in range(N):
		if ts[i] > ratio:
			cnt += 1
	return cnt / N

def Top_max(ts, ratio = 0.9):
	N, now, max_ans = len(ts), 0.0, 0.0
	for i in range(N):
		if ts[i] > ratio:
			now += 1
		else:
			max_ans = max(max_ans, now)
			now = 0
	return max_ans / N  

def Auto_correlation(x, t = 288):
	N = len(x)
	sigma = np.std(x)
	y = np.array(x, dtype = float) - sigma
	ans = 0
	for i in range(len(x) - t):
		ans += y[i] * y[i + t]
	return ans / ((N - t) * (sigma * sigma))

def Bf_period(x, t = 288):
	N = len(x) // t
	season = []
	for j in range(t):
		temp = 0
		for i in range(N):
			temp += x[i * t + j]
		temp /= N
		season.append(temp)
	ans = 0
	for i in range(N):
		for j in range(t):
			ans += (x[i * t + j] - season[j]) ** 2
	return ans / (N * t)

feature_dict = {
	'Auto_correlation': Auto_correlation,
	'Bf_period': Bf_period,
	'Spen': Spen,
	'Skew': Skew,
	'Kurtosis': Kurtosis,
	'Mean': Mean,
	'Median': Median,
	'Std': Std,
	'Abs_change': Abs_change,
	'Linear': Linear,
	'Zero_cnt': Zero_cnt,
	'Zero_max': Zero_max,
	'Top_cnt': Top_cnt,
	'Top_max': Top_max,
}

#featurename_list = ['Apen', 'Spen', 'Skew', 'Kurtosis', 'Mean', 'Median', 'Std', 'Abs_change', 'Linear', 'Zero_cnt', 'Zero_max']
#func_list = [Apen, Spen, Skew, Kurtosis, Mean, Median, Std, Abs_change, Linear, Zero_cnt, Zero_max]