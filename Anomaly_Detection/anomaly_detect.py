from HoltWinters import holtWinters, naive_sigma, sigma_diff
from MA import moving_average, get_fluc
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib
import sys
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


#value, anomaly = np.array(data['value']), np.array(data['anomaly_score'])

# Here starts main functions used for predictions. 
# --------------------------------------------------------

# This is the function to get features for the datasets. 
# And the features will be used to train IForests or RandomForests. 

def _Unsupervised_nolabel(value_train, anomaly_ratio = 0.004):
	N = len(value_train)
	mean, sigma = np.mean(value_train), np.std(value_train)

	print("\nCalculating features...\n")

	MA1, MA2, MA3, MA4, MA5, MA6, Dif1, Dif2, Dif3, Dif4, Dif5, Dif6 = moving_average(value_train)
	feature_3_sigma = naive_sigma(value_train, 3, mean, sigma)
	feature_4_sigma = naive_sigma(value_train, 4, mean, sigma)
	feature_5_sigma = naive_sigma(value_train, 5, mean, sigma)
	feature_mean_diff = sigma_diff(value_train, mean)
	feature_holtwinter_diff, feature_holtwinter, predicted_holtwinter, hw_smoothdiff, hw_smooth, hw_mul, hw_sigma, hw_params = holtWinters(
		value_train, anomaly_cnt = len(value_train) * anomaly_ratio, ahead = 0)
	feature_fluc = get_fluc(value_train)
	params = (mean, sigma, hw_mul, hw_sigma, len(value_train))

	print("Combining features...\n")
	
	features1 = np.r_['1,2,0',
	#MA1, MA2, MA3, MA4, MA5, MA6, Dif1, Dif2, Dif3, Dif4, Dif5, Dif6, 
	feature_3_sigma, feature_4_sigma, feature_5_sigma, feature_mean_diff, 
	feature_holtwinter_diff, feature_holtwinter, hw_smoothdiff, hw_smooth, feature_fluc]

	features = feature_3_sigma[50:]
	#features = MA1[50:]
	for f in [
	#MA2, MA3, MA4, MA5, MA6, Dif1, Dif2, Dif3, Dif4, Dif5, Dif6, feature_3_sigma,
	feature_4_sigma, feature_5_sigma, feature_mean_diff, 
	feature_holtwinter_diff, feature_holtwinter, hw_smoothdiff, hw_smooth, feature_fluc]:
		features = np.r_['1,2,0', features, f[50:]]
		

	print("Training...\n")

	clf = IsolationForest(contamination = anomaly_ratio, 
		bootstrap = False, max_samples = 'auto'
		)
	#clf = LocalOutlierFactor(contamination = anomaly_ratio)
	clf.fit(features)
	y = clf.predict(features1)
	#y = clf.fit_predict(features1)

	outlierx, outliery = _getanomaly_graph(y, value_train, key = -1)
	print("Find Anomaly: ", len(outlierx))
	legend = plt.legend(loc='upper left', shadow=True, fontsize='small')
	plt.plot(value_train, label = 'real_value')
	plt.scatter(outlierx, outliery, s = 30, c = 'red', marker = 'x', label = 'detected_anomaly')
	legend = plt.legend(loc='upper left', shadow=True, fontsize='small')

	return clf, y

def _Unsupervised_withlabel(value_train, anomaly_train = None, anomaly_ratio = 0.004):
	N = len(value_train)
	Stdanomcnt = _InitialGraph(value_train, anomaly_train)
	mean, sigma = np.mean(value_train), np.std(value_train)

	#MA1, MA2, MA3, MA4, MA5, MA6, Dif1, Dif2, Dif3, Dif4, Dif5, Dif6 = moving_average(value_train)
	feature_3_sigma = naive_sigma(value_train, 3, mean, sigma)
	feature_4_sigma = naive_sigma(value_train, 4, mean, sigma)
	feature_5_sigma = naive_sigma(value_train, 5, mean, sigma)
	feature_mean_diff = sigma_diff(value_train, mean)
	feature_holtwinter_diff, feature_holtwinter, predicted_holtwinter, hw_smoothdiff, hw_smooth, hw_mul, hw_sigma, hw_params = holtWinters(
		value_train, anomaly_cnt = len(value_train) * anomaly_ratio, ahead = 0)
	params = (mean, sigma, hw_mul, hw_sigma, len(value_train))

	print("Combining features...\n")
	
	features1 = np.r_['1,2,0',
	feature_3_sigma, feature_4_sigma, feature_5_sigma, feature_mean_diff, 
	feature_holtwinter_diff, feature_holtwinter, hw_smoothdiff, hw_smooth]

	features = feature_3_sigma[50:]
	for f in [
	feature_4_sigma, feature_5_sigma, feature_mean_diff, 
	feature_holtwinter_diff, feature_holtwinter, hw_smoothdiff, hw_smooth]:
		features = np.r_['1,2,0', features, f[50:]]
		

	print("Training...\n")

	clf = IsolationForest(contamination = anomaly_ratio, bootstrap = True, max_samples = 'auto')
	clf.fit(features)
	y = clf.predict(features1)
	misscnt, misrcnt = 0, 0
	for i in range(len(y)):
		if y[i] == -1:
			if anomaly_train[i] == 0:
				misrcnt += 1
		else:
			if anomaly_train[i] == 1:
				misscnt += 1

	outlierx, outliery = _getanomaly_graph(y, value_train, key = -1)
	print("All to find: ", Stdanomcnt)
	print("Find Anomaly: ", len(outlierx))
	print("Missing: ", misscnt)
	print("Missreport: ", misrcnt)
	legend = plt.legend(loc='upper left', shadow=True, fontsize='small')
	plt.subplot(212)
	plt.plot(value_train, label = 'real_value')
	plt.scatter(outlierx, outliery, s = 30, c = 'red', marker = 'x', label = 'detected_anomaly')
	legend = plt.legend(loc='upper left', shadow=True, fontsize='small')

	return clf, y

def _Supervised(value, anomaly, saved_ratio = 0.1):
	print("Processing Data\n")
	_InitialGraph(value, anomaly)
	N, SavedSamples = len(value), int(len(value) * saved_ratio)
	value_train, anomaly_train = value[:-SavedSamples], anomaly[:-SavedSamples]
	value_test, anomaly_test = value[-SavedSamples:], anomaly[-SavedSamples:]
	model, predict, params, hw_params = _train(value_train, anomaly_train, SavedSamples)
	_test(value_test, anomaly_test, predict, model, params, hw_params)
	return model


# The graph function at the beginning for a big picture. Return the number of anomaly
def _InitialGraph(value, anomaly):
	Stdanox, Stdanoy = _getanomaly_graph(anomaly, value)
	plt.subplot(211)
	plt.scatter(Stdanox, Stdanoy, s = 20, c = 'red', marker = 'x', label = 'real_anomaly')
	plt.plot(value, label = 'real_value')
	return len(Stdanox)



# The train function
def _train(value_train, anomaly_train, SavedSamples):
	x, y = _getanomaly(anomaly_train)
	anomaly_cnt = len(y)
	mean, sigma = np.mean(value_train), np.std(value_train)
	MA1, MA2, MA3, MA4, MA5, MA6, Dif1, Dif2, Dif3, Dif4, Dif5, Dif6 = moving_average(value_train)
	feature_3_sigma = naive_sigma(value_train, 3, mean, sigma)
	feature_4_sigma = naive_sigma(value_train, 4, mean, sigma)
	feature_5_sigma = naive_sigma(value_train, 5, mean, sigma)
	feature_mean_diff = sigma_diff(value_train, mean)
	feature_holtwinter_diff, feature_holtwinter, predicted_holtwinter, hw_smoothdiff, hw_smooth, hw_mul, hw_sigma, hw_params = holtWinters(
		value_train, anomaly_cnt, ahead = SavedSamples)
	params = (mean, sigma, hw_mul, hw_sigma, len(value_train))

	print("Combining features...\n")
	features1 = np.r_['1,2,0',
	#MA1, MA2, MA3, MA4, MA5, MA6, 
	#Dif1, Dif2, Dif3, Dif4, Dif5, Dif6,
	feature_3_sigma, feature_4_sigma, feature_5_sigma, feature_mean_diff, 
	feature_holtwinter_diff, feature_holtwinter, hw_smoothdiff, hw_smooth]

	features = feature_3_sigma[50:]
	for f in [#MA2, MA3, MA4, MA5, MA6, 
	#Dif2, Dif3, Dif4, Dif5, Dif6,
	feature_4_sigma, feature_5_sigma, feature_mean_diff, 
	feature_holtwinter_diff, feature_holtwinter, hw_smoothdiff, hw_smooth]:
		features = np.r_['1,2,0', features, f[50:]]
		
	'''
	results = np.r_[value_train, predicted_holtwinter]
	plt.plot(results)

	anomaly_x, anomaly_y = _getanomaly(anomaly_train, upshift = 100)
	holtwinter_x, holtwinter_y = _getanomaly(feature_holtwinter, upshift = 120)
	plt.scatter(anomaly_x, anomaly_y, s = 10, c = 'red', marker = 'x')
	plt.scatter(holtwinter_x, holtwinter_y, s = 10, c = 'orange', marker = 'x')
	plt.show()
	'''
	print("Training...\n")


	clf = RandomForestClassifier(n_estimators = 100, max_depth = 10)
	clf.fit(features, anomaly_train[50:])
	print(clf.feature_importances_)
	plt.plot(np.concatenate((value_train, predicted_holtwinter)), label = 'prediction')
	legend = plt.legend(loc='upper left', shadow=True, fontsize='small')



	return clf, predicted_holtwinter, params, hw_params


# Test function
def _test(value_test, anomaly_test, predict, model, params, hw_params):
	outlier, test_features, test_smooth = [], [], []
	misscnt, misreportcnt, p = 0, 0, 288
	mean, sigma, hw_mul, hw_sigma, base = params
	Lt1, Tt1, St1 = hw_params[0]
	alpha, beta, gamma = hw_params[1:4]
	for i in range(len(value_test)):
		if abs(value_test[i] - mean) > 3 * sigma:
			test_features.append(1)
		else:
			test_features.append(0)
		if abs(value_test[i] - mean) > 4 * sigma:
			test_features.append(1)
		else:
			test_features.append(0)
		if abs(value_test[i] - mean) > 5 * sigma:
			test_features.append(1)
		else:
			test_features.append(0)
		test_features.append(value_test[i] - mean)
		test_features.append(value_test[i] - predict[i])
		if abs(value_test[i] - predict[i]) > hw_mul * hw_sigma[(base + i) % 24]:
			test_features.append(1)
		else:
			test_features.append(0)
		Lt = alpha * (value_test[i] - St1[(i + base) % p]) + (1 - alpha) * (Lt1 + Tt1)
		Tt = beta * (Lt - Lt1) + (1 - beta) * (Tt1)
		St = gamma * (value_test[i] - Lt) + (1 - gamma) * (St1[(i + base) % p])
		Lt1, Tt1, St1[(i + base) % p] = Lt, Tt, St
		test_smooth.append(Lt1 + Tt1 + St1[(i + base) % p])
		test_features.append(value_test[i] - test_smooth[i])
		test_features.append(Lt1 + Tt1 + St1[(i + base) % p])

		'''
		dif = []
		for j in [5,10,20,30,40,50]:
			sum = 0
			for k in range(max(0,i-j), i+1):
				sum += value_test[k]
			sum /= (i + 1 - max(0,i-j))
		#	test_features.append(sum)
			dif.append(value_test[i]-sum)
		for item in dif:
			test_features.append(item)
		'''
		res = model.predict([test_features])
		outlier.append(res)
		if res == 1:
			if anomaly_test[i] == 0:
				misreportcnt += 1
		else:
			if anomaly_test[i] == 1:
				misscnt += 1
		test_features.clear()

# Here is a pure Holt-winter test to see if Randomforest does a better job.
# -----------------------------------------------------------------------------

	holt_x, holt_y = [], []
	hw_misreportcnt, hw_missingcnt = 0, 0
	for i in range(len(value_test)):
		if abs(value_test[i] - predict[i]) > hw_mul * hw_sigma[(base + i) % 24]:
			holt_x.append(i)
			holt_y.append(150)
			if anomaly_test[i] == 0:
				hw_misreportcnt += 1
		else:
			if anomaly_test[i] == 1:
				hw_missingcnt += 1
	outlierx, outliery = _getanomaly_graph(outlier, value_test)
	anomalyx, anomalyy = _getanomaly_graph(anomaly_test, value_test)
	print("Pure Holt test results: \n")
	print("All to find:  ", len(anomalyx))
	print("Find in test:  ", len(holt_x))
	print("Missing:  ", hw_missingcnt)
	print("Misreport:  ", hw_misreportcnt)

# The pure holt test ends here
# --------------------------------------

	print("\n\nRandomforest test results: \n")
	print("All to find:  ", len(anomalyx))
	print("Find in test:  ", len(outlierx))
	print("Missing:  ", misscnt)
	print("Misreport:  ", misreportcnt)
	
	plt.subplot(212)
	plt.scatter(outlierx, outliery, s = 40, c = 'red', marker = 'x', label = 'detected_anomaly')
	plt.scatter(anomalyx, anomalyy, s = 20, c = 'blue', marker = 'o', label = 'real_anomaly')
	plt.plot(value_test, label = 'real_value', c = 'green')
	legend = plt.legend(loc='upper left', shadow=True, fontsize='small')


# Given anomaly as (0, 0, 1, 1, 0, 1 ... ) and output only 1's 

def _getanomaly(anomaly, base = 0, upshift = 0, key = 1):	
	# base is used to locate the starting point on the final graph. upshift is used to make the final graph more clear.  
	# key is the label of anomaly, default set to 1. 
	anomaly_x = []
	anomaly_y = []
	for i in range(base, base + len(anomaly)):
		if anomaly[i] == key:
			anomaly_x.append(i)
			anomaly_y.append(anomaly[i]+upshift)
	return anomaly_x, anomaly_y


# Given anomaly and value, output the datasets suitable for drawing on the graph. 

def _getanomaly_graph(anomaly, value, key = 1):	
	# base is used to locate the starting point on the final graph. upshift is used to make the final graph more clear.  
	anomaly_x = []
	anomaly_y = []
	for i in range(0, len(anomaly)):
		if anomaly[i] == key:
			anomaly_x.append(i)
			anomaly_y.append(value[i])
	return anomaly_x, anomaly_y



if __name__ == "__main__":
	'''
	data = pd.read_csv('GOOGresults.csv')
	value = np.array(data[data.columns[1]])
	anomaly = np.array(data[data.columns[2]])
	_Supervised(value, anomaly, 0.2)
	'''
	if len(sys.argv) < 4:
		print("Please input filename(.csv), whether or not have labels for anomaly(0/1), anomaly_ratio(float)\n")
		sys.exit()
	filename = sys.argv[1]
	t = int(sys.argv[2])
	r = float(sys.argv[3])
	data = pd.read_csv(filename)
	value = np.array(data[data.columns[1]])
	if t == 1:
		anomaly = np.array(data[data.columns[2]])
		model, pred = _Unsupervised_withlabel(value, anomaly, r)
		plt.savefig(filename[:-4]+'_unsupervised', dpi = 500)
		plt.close()
		_Supervised(value, anomaly, 0.2)
		plt.savefig(filename[:-4]+'_supervised', dpi = 500)
		plt.close()
	else:
		_Unsupervised_nolabel(value, r)
		plt.savefig(filename[:-4]+'_unsupervised', dpi = 500)
		plt.close()