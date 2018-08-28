from sklearn.externals import joblib
import numpy as np 
import time
import sys
import pandas as pd 
import math
import os
from feature_method import *
from curve_classify import *

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Please input the name of the new file\n")
		sys.exit()
	file_path = sys.argv[1]
	model = joblib.load('rforest_model.pkl')
	feature = get_feature(file_path)
	c = model.predict([np.array(feature.iloc[0][1:])])
	print(c[0])