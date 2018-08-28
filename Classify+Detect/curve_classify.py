from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import numpy as np 
import time
import sys
import pandas as pd 
import math
import graphviz
import os
import matplotlib.pyplot as plt 
from sklearn import tree
from sklearn.cluster import Birch, KMeans
from sklearn.ensemble import RandomForestClassifier
from feature_method import *
from statsmodels.tsa.seasonal import seasonal_decompose

slice_len = 200

fc_parameters = {
	#'agg_autocorrelation': [{'f_agg': 'mean'}, {'f_agg': 'median'}],
	#'skewness': None,
	#'kurtosis': None,
	#'approximate_entropy': [{'r': 0.1, 'm': 2}, {'r': 0.3, 'm': 2}, {'r': 0.5, 'm': 2}, {'r': 0.7, 'm': 2}, {'r': 0.9, 'm': 2}],
	#'binned_entropy': [{'max_bins': 10}],
	'sample_entropy': None,
	#'variance': None,
	#'mean': None,
	#'median': None,
	#'fft_aggregated': [{'aggtype': 'centroid'}, {'aggtype': 'variance'}, {'aggtype': 'skew'}, {'aggtype': 'kurtosis'}],
	#'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'mean'},{'attr': 'slope', 'chunk_len': 5, 'f_agg': 'mean'}],
	#'absolute_sum_of_changes': None,
}

weights1 = [20,1,1,4,4,1,1,1,1,1,1,1,1,1,0.5,1,0,1,1] # For Kmeans
weights2 = [10,1,1,4,4,1,1,1,1,1,1,1,1,1,0.5,1,0,1,1] # For Birch
weights = [100,10,100,1,10000,1,1,1,100,1,10,100,10,100]

def visual(filename_list, pred, cluster_num, index = 1, Length = 3000):
	classindex = []
	for c in range(cluster_num):
		temp = []
		for i in range(len(pred)):
			if pred[i] == c:
				temp.append(i)
		classindex.append(temp)

	ind = 0
	for i in range(len(classindex)):
		plt.figure()
		plt.suptitle('Class: "{}"'.format(i))
		L = classindex[i]
		N = math.sqrt(len(L)) + 1
		for i in range(len(L)):
			data = pd.read_csv(filename_list[L[i]])
			if len(data) > Length:
				data = data[:Length]
			value = np.array(data[data.columns[index]])
			value = pre_process(value)
			plt.subplot(N, N, i + 1)
			plt.xticks([])
			plt.yticks([])
			plt.plot(value)
			#plt.plot(normalize(holtWinters_forclass(value)))
		#plt.savefig('Figure_for_Class{}'.format(ind), dpi = 600)
		plt.show()
		ind += 1

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
'''
def get_feature(filename, index = 1, Length = 3000):
	print(filename)
	data = pd.read_csv(filename)
	if len(data) > Length :
		data = data[:Length]
	ts = np.array(data[data.columns[index]])
	ts = pre_process(ts, slice_len)
	data_input = pd.DataFrame()
	data_input['id'] = np.full(len(ts), 1)
	data_input['value'] = ts
	features = extract_features(data_input, column_id = 'id', default_fc_parameters=fc_parameters)
	impute(features)
	features['value__absolute_sum_of_changes'] = features['value__absolute_sum_of_changes'] / len(data)
	features['value_zero_cnt'], features['value_zero_max'] = zero_feature(ts)
	return features 
'''
def get_feature(filename, index = 1, Length = 3000):
	print("Processing :", filename)
	data = pd.read_csv(filename)
	if len(data) > Length:
		data = data[:Length]
	ts = np.array(data[data.columns[index]])
	ts = pre_process(ts)
	feature = pd.DataFrame()
	feature['id'] = [filename]
	print("Starting to calculate\n")
	_s = time.time()
	for featurename in sorted(feature_dict):
		func = feature_dict[featurename]
		feature[featurename] = [func(ts)]
	print("All time used: ", time.time() - _s)
	return feature

def add_feature(feature_path, featurename, func):
	features = pd.read_csv(feature_path)
	new_feature = []
	cnt = 0
	for filename in features['id']:
		print("Processing {}, {}\n".format(filename, cnt))
		data = pd.read_csv(filename)
		value = np.array(data['value'])
		value = pre_process(value)
		new_feature.append(func(value))
		cnt += 1
	features[featurename] = new_feature
	features.to_csv('features.csv')

def feature_list(filename_list, feature_path, index = 1):
	feature = pd.DataFrame()
	i, num = 1, len(filename_list)
	for filename in filename_list:
		feature_temp = get_feature(filename, index)
		feature = pd.concat([feature, feature_temp])
		feature.to_csv(feature_path)
		print('{}/{}\n'.format(i, num))
		i += 1
	#for i in range(len(feature.columns)):
	#	feature[feature.columns[i]] = normalize(feature[feature.columns[i]])
	return feature

def combine(feature, weights):
	if len(feature) != len(weights):
		print("Danger!!! weights' length is not right!\n\n")
		return
	ans = []
	for i in range(len(weights)):
		if weights[i] != 0:
			ans.append(weights[i] * feature[i])
	return ans

def train(feature, weights, cluster_num, feature_path = None):
	if feature_path != None:
		feature = pd.read_csv(feature_path)
	X, Y = [], []
	print("Training...\n")
	for i in range(len(feature[feature.columns[0]])):
		f = np.array(feature.iloc[i][1:])
		f_w = combine(feature.iloc[i][1:], weights)
		print(f)
		print(f_w)
		X.append(f_w)
		Y.append(f)
	clf = Birch(n_clusters = cluster_num)
	clf = KMeans(n_clusters = cluster_num)
	clf.fit(X)
	pred = clf.predict(X)
	joblib.dump(clf, 'curve_model_KMeans.pkl')
	rdf = RandomForestClassifier()
	rdf.fit(Y, pred)
	joblib.dump(rdf, 'rforest_model.pkl')
	print(pred)
	return pred

def visual_self(filename_list, pred, index = 1, Length = 3000):
	classindex = []
	for c in range(cluster_num):
		temp = []
		for i in range(len(pred)):
			if pred[i] == c:
				temp.append(i)
		classindex.append(temp)

	for i in range(len(classindex)):
		plt.figure()
		plt.suptitle('Class: "{}"'.format(i))
		L = classindex[i]
		N = math.sqrt(len(L)) + 1
		for i in range(len(L)):
			data = pd.read_csv(filename_list[L[i]])
			if len(data) > Length:
				data = data[:Length]
			value = data[data.columns[index]]
			plt.subplot(N, N, i + 1)
			plt.plot(value)
	plt.show()

def scratch(filename):
	data = pd.read_csv(filename)
	value = data[data.columns[1]]
	value = np.array(value)
	plt.figure()
	plt.subplot(211)
	plt.plot(value, label = 'real_value')
	plt.subplot(212)
	plt.plot(pre_process(value, slice_len), label = 'smooth_value')
	legend = plt.legend(loc='upper left', shadow=True, fontsize='small')
	plt.show()

def all_graph():
	filename_list = filename_list[3:]
	for i in range(0,40):
		plt.figure()
		for j in range(1,26):
			print(i * 25 + j)
			data = pd.read_csv(filename_list[i * 25 + j - 1])
			print(data)
			value = np.array(data['value'])
			plt.subplot(5, 5, j)
			plt.plot(value)
			plt.plot(pre_process(value, slice_len))
		plt.show()

def get_class(filename, curve_model_path = ['curve_model_Birch.pkl', 'curve_model_KMeans.pkl'], 
	weights = [weights1, weights2], value_index = 1):
	model_1 = joblib.load('curve_model_Birch.pkl')
	model_2 = joblib.load('curve_model_KMeans.pkl')
	#data = pd.read_csv(filename)
	#value = np.array(data[data.columns[value_index]])
	#if len(feature) == 0:
	feature = get_feature(filename)
	print(feature)
	feature = np.array(feature.iloc[0])
	print(feature)
	feature1 = combine(feature, weights[0])
	feature2 = combine(feature,weights[1])
	print(feature1)
	c1 = model_1.predict([feature1])
	c2 = model_2.predict([feature2])
	c = 0
	if c1 == 1 or c1 == 6:
		if c2 == 2 or c2 == 7:
			c = 1
	return c

def get_abs_diff(ts, season, trend):
	ans, diff = 0, []
	for i in range(len(ts)):
		diff.append(ts[i] - season[i] - trend)
		ans += abs(diff[i])
	return (ans, diff)

def visual_diff(filename_list, pred, cluster_num, index = 1, Length = 3000):
	# This is for painting the diff
	print("Saving graphs...\n")
	classindex = []
	for c in range(cluster_num):
		temp = []
		for i in range(len(pred)):
			if pred[i] == c:
				temp.append(i)
		classindex.append(temp)

	ind = 0
	for i in range(len(classindex)):
		print("Graph {}\n".format(i))
		plt.figure()
		plt.suptitle('Class: "{}"'.format(i))
		L = classindex[i]
		N = math.sqrt(len(L)) + 1
		for i in range(len(L)):
			#print(filename_list[L[i]])
			data = pd.read_csv(filename_list[L[i]])
			if len(data) > Length:
				data = data[:Length]
			value = np.array(data[data.columns[index]])
			diff, season, abs_value, trend = get_seasonal_diff(value)
			plt.subplot(N, N, i + 1)
			plt.plot(diff)
		plt.savefig('Figure_diff_for_Class{}'.format(ind), dpi = 600)
		plt.close()
		ind += 1

def visual(filename_list, pred, cluster_num, index = 1, Length = 3000):
	# This is for painting the whole ts
	print("Saving graphs...\n")
	classindex = []
	for c in range(cluster_num):
		temp = []
		for i in range(len(pred)):
			if pred[i] == c:
				temp.append(i)
		classindex.append(temp)

	ind = 0
	for i in range(len(classindex)):
		print("Graph {}\n".format(i))
		plt.figure()
		plt.suptitle('Class: "{}"'.format(i))
		L = classindex[i]
		N = math.sqrt(len(L)) + 1
		for i in range(len(L)):
			#print(filename_list[L[i]])
			data = pd.read_csv(filename_list[L[i]])
			if len(data) > Length:
				data = data[:Length]
			value = np.array(data[data.columns[index]])
			plt.subplot(N, N, i + 1)
			plt.xticks([])
			plt.yticks([])
			plt.plot(value)
		plt.savefig('Figure_for_Class{}'.format(ind), dpi = 600)
		plt.close()
		ind += 1

def visual_feature(feature_path, featurename):
	data = pd.read_csv(feature_path)
	feature = np.array(data[featurename])
	plt.scatter(feature, np.full(len(feature), 1), s = 0.01)
	plt.show()
	plt.close()

def train_classify(feature_path, model_path, bf_index = 2):
	feature = pd.read_csv(feature_path)
	X, flst = [], []
	print("Training...\n")
	for i in range(len(feature[feature.columns[0]])):
		f = np.array(feature.iloc[i][1:])
		key = f[bf_index]
		if key > 0.008:
			flst.append(feature.iloc[i][0])
			f_w = combine(feature.iloc[i][1:], weights)
			X.append(f_w)
	clf = joblib.load(model_path)
	pred = clf.predict(X)
	c = []
	print(pred)
	for i in range(len(pred)):
		t = 2
		if pred[i] in [1,2,7,12,15,16,20,25,33,38,49]: # big keng
			t = 1
		if pred[i] in [5,8,9,10,14,23,24,26,28,29,30,37,42,47]: # period
			t = 0
		c.append(t)
	X = []
	for i in range(len(feature[feature.columns[0]])):
		f = np.array(feature.iloc[i][1:])
		key = f[bf_index]
		if key > 0.008:
			f_w = feature.iloc[i][1:]
			X.append(f_w)
	rdf = tree.DecisionTreeClassifier(max_depth = 7)
	rdf = rdf.fit(X, c)
	dot_data = tree.export_graphviz(rdf, out_file="tree.dot")  
	graph = graphviz.Source(dot_data)
	joblib.dump(rdf, 'curve_model.pkl')
	pred = rdf.predict(X)
	print(pred)
	visual(flst, pred, 3)

def recal(feature_path):
	data = pd.read_csv(feature_path)
	filename_list = np.array(data['id'])
	features = feature_list(filename_list, feature_path = 'feature1.csv')
	features.to_csv('featuresss.csv')

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Please input the number of clusters\n")
		sys.exit()
	cluster_num = int(sys.argv[1])
	feature_path = 'features.csv'
	flst = os.listdir()
	filename_list = []
	for filename in flst:
		if filename[-4:] == '.csv':
			filename_list.append(filename)
	features = feature_list(filename_list, feature_path = feature_path)
	#filename_list = os.listdir()[2:-1]
	#features = feature_list(filename_list, feature_path = feature_path)
	#features.to_csv(feature_path)
	'''
	f = get_feature('_0_6_34_71685.csv')
	for i in range(len(f.columns)):
		print(f[f.columns[i]])
	'''
	#add_feature(feature_path, 'Bf_period', Bf_period)

	#visual_feature('feature.csv', 'Bf_period')
	#train_classify(feature_path, 'curve_model_Birch.pkl')
	#recal(feature_path)
	pred = train(features, weights = weights, cluster_num = cluster_num)
	data = pd.read_csv(feature_path)
	filename_list = np.array(data['id'])
	index = np.random.randint(0,len(filename_list),len(filename_list)//5)
	flst, p =[], []
	for i in range(len(index)):
		flst.append(filename_list[index[i]])
		p.append(pred[index[i]])
	visual(flst, p, cluster_num)
	'''
	name = pd.read_csv(feature_path)
	filename_list = np.array(name['id'])
	for filename in filename_list:
		data = pd.read_csv(filename)
		plt.figure()
		plt.plot(data['value'])
		plt.savefig(filename[:-4], dpi = 400)
		plt.close()
	'''