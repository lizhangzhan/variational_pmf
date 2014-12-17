import sys
import matplotlib.pyplot as plt
sys.path.append("../../.")
from variational_pmf.code.variational_pmf import VariationalPMF
from variational_pmf.code.load_store_matrices import store_X_U_V
from variational_pmf.code.run_different_k import recover_predictions, generate_M
from run_real_data import read_data

"""
Run the Variational PMF algorithm for the best value of K found by running
run_real_data.py, and store the matrices U, V, and the predictions X in a file.
"""


if __name__ == "__main__":
	X = read_data("gi50_no_missing.txt")
	(I,J) = X.shape
	fraction_unknown = 0.1
	M = generate_M(I,J,fraction_unknown)
	K = 3
	outputfile = "recovered_matrices.txt"

	PMF = VariationalPMF(X,M,K)
	PMF.run(iterations=20,updates=1)

	predicted_U = PMF.U
	predicted_V = PMF.V
	predicted_X = PMF.predicted_X

	# Store the predicted matrix X with U and V
	store_X_U_V(outputfile,predicted_X,predicted_U,predicted_V)

	# Now we plot the predictions vs the true values
	actual_vs_predicted = recover_predictions(M,X,predicted_X)

	(x,y) = zip(*actual_vs_predicted)
	plt.figure()
	plt.xlabel('actual')
	plt.ylabel('predicted')
	plt.scatter(x,y)
	plt.plot([0,10],[0,10]) #y=x line
	plt.show()

	plt.figure()
	plt.xlabel('actual - all')
	plt.ylabel('predicted - all')
	plt.scatter(X,predicted_X)
	plt.plot([0,10],[0,10]) #y=x line
	plt.show()