import sys, random, numpy
import matplotlib.pyplot as plt

import generate_example, recover_example
sys.path.append("../../.")
from variational_pmf.code.variational_pmf import VariationalPMF


"""
In this example we randomly generate a IxJ matrix X using a KxI matrix U and KxJ
matrix V. Of this 100x80 matrix, we randomly leave a tenth of the values out, and use 
the variational PMF module to predict these values. We then plot the predicted vs the
real values, and observe the trend.
"""

if __name__ == "__main__":
	I = 100
	J = 80
	K = 2

	M = numpy.ones([I,J])
	# Randomly set a tenth of the values to 0 (unknown)
	values = random.sample(range(0,I*J),(I*J)/10)
	for v in values:
		M[v / I][v % J] = 0

	generated = "generated.txt"
	recovered = "recovered.txt"

	generate_example.generate(I,J,K,generated)
	recover_example.recover(M,generated,recovered,20)

	(_,_,_,original_X,original_U,original_V) = recover_example.read_X_U_V(generated)
	(_,_,_,predicted_X,predicted_U,predicted_V) = recover_example.read_X_U_V(recovered)

	unknown_values_indices = []
	actual_vs_predicted = []
	for i in range(0,I):
		for j in range(0,J):
			if M[i][j] == 0:
				unknown_values_indices.append((i,j))
				actual_vs_predicted.append((original_X[i][j],predicted_X[i][j]))
	
	(x,y) = zip(*actual_vs_predicted)
	plt.figure()
	plt.xlabel('actual')
	plt.ylabel('predicted')
	plt.scatter(x,y)
	plt.plot([-10,10],[-10,10]) #y=x line
	plt.show()