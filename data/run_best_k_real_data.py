import sys, numpy, math
import matplotlib.pyplot as plt
sys.path.append("../../.")
from variational_pmf.code.variational_pmf import VariationalPMF
from variational_pmf.code.load_store_matrices import store_X_U_V
from variational_pmf.code.run_different_k import recover_predictions, generate_M, compute_RMSE
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
	iterations = 10

	PMF = VariationalPMF(X,M,K)
	PMF.run(iterations=iterations,updates=1)

	predicted_U = PMF.U
	predicted_V = PMF.V
	predicted_X = PMF.predicted_X

	# Store the predicted matrix X with U and V
	store_X_U_V(outputfile,predicted_X,predicted_U,predicted_V)

	# Now we plot the predictions vs the true values
	actual_vs_predicted = recover_predictions(M,X,predicted_X)
	(actual,predicted) = zip(*actual_vs_predicted)

	plt.figure()
	plt.xlabel('values - actual vs predicted')
	plt.ylabel('no.')
	bins = [v * 0.1 for v in range(0,10*10)]
	plt.hist(actual,bins,alpha=0.5,label='actual')
	plt.hist(predicted,bins,alpha=0.5,label='predicted')
	plt.show()

	# Also do a histogram plot of the differences between the actual and prediction
	differences = [math.fabs(v1-v2) for (v1,v2) in actual_vs_predicted]
	plt.figure()
	plt.xlabel('difference between actual and predicted')
	plt.ylabel('no.')
	bins = [v * 0.1 for v in range(0,10*5)]
	plt.hist(differences,bins)
	plt.show()

	differences = [math.fabs(v1-v2) for (v1,v2) in zip(numpy.ndarray.flatten(X),numpy.ndarray.flatten(predicted_X))]
	plt.figure()
	plt.xlabel('difference between actual and predicted - all')
	plt.ylabel('no.')
	bins = [v * 0.1 for v in range(0,10*5)]
	plt.hist(differences,bins)
	plt.show()

	# Plot the predictions - only the "unknown" values (1 entries in M)
	plt.figure()
	plt.xlabel('actual')
	plt.ylabel('predicted')
	plt.scatter(actual,predicted)
	plt.plot([-2,10],[-2,10]) #y=x line
	plt.show()

	# Plot all the values, including the ones that are already known
	plt.figure()
	plt.xlabel('actual - all')
	plt.ylabel('predicted - all')
	plt.scatter(X,predicted_X)
	plt.plot([-2,10],[-2,10]) #y=x line
	plt.show()