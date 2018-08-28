import numpy as np 

def moving_average(value):
	ANS = []
	Diff = []
	for n in [5,10,20,30,40,50]:
		temp = np.cumsum(value, dtype = float)
		temp[n:] = temp[n:] - temp[:-n]
		for i in range(n):
			temp[i] /= (i+1)
		L = temp
		Dif = value - temp
		Diff.append(Dif)
		ANS.append(L)
	return ANS + Diff

def get_fluc(value):
	N = len(value)
	Dif = []
	for i in range(N-1):
		Dif.append(abs(value[i+1]-value[i]))
	Dif.append(Dif[-1])
	temp, fluc = 0, []
	for i in range(5):
		temp += Dif[i]
		fluc.append(temp)
	for i in range(5, N):
		temp += Dif[i]
		temp -= Dif[i-5]
		fluc.append(temp)
	for i in range(5):
		fluc[i] = fluc[5]
	return fluc
