import numpy, random

"""
This file contains some general helper functions.
"""

def recover_predictions(M,X_true,X_pred):
	(I,J) = M.shape
	actual_vs_predicted = []
	for i in range(0,I):
		for j in range(0,J):
			if M[i][j] == 0:
				actual_vs_predicted.append((X_true[i][j],X_pred[i][j]))
	return (actual_vs_predicted)

def generate_M(I,J,fraction):
	M = numpy.ones([I,J])
	values = random.sample(range(0,I*J),int(I*J*fraction))
	for v in values:
		M[v / J][v % J] = 0
	return M

def calc_inverse_M(M):
	(I,J) = M.shape
	M_inv = numpy.ones([I,J])
	for i in range(0,I):
		for j in range(0,J):
			if M[i][j] == 1:
				M_inv[i][j] = 0
	return M_inv