import sys, random, numpy
import matplotlib.pyplot as plt

import generate_example, recover_example
sys.path.append("../../.")
from variational_pmf.code.load_store_matrices import load_X_U_V, store_X_U_V
from variational_pmf.code.variational_pmf import VariationalPMF
from variational_pmf.code.run_different_k import recover_predictions, generate_M


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

	M = generate_M(I,J,0.1)

	generated = "generated.txt"
	recovered = "recovered.txt"

	generate_example.generate_X(I,J,K,generated)
	recover_example.recover(M,generated,recovered,20)

	(_,_,_,original_X,original_U,original_V) = load_X_U_V(generated)
	(_,_,_,predicted_X,predicted_U,predicted_V) = load_X_U_V(recovered)

	actual_vs_predicted = recover_predictions(M,original_X,predicted_X)

	(x,y) = zip(*actual_vs_predicted)
	plt.figure()
	plt.xlabel('actual')
	plt.ylabel('predicted')
	plt.scatter(x,y)
	plt.plot([-10,10],[-10,10]) #y=x line
	plt.show()