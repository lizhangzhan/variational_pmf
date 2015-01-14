import sys, numpy, math
import matplotlib.pyplot as plt
sys.path.append("../../../.")
from variational_pmf.code.variational_pmf import VariationalPMF
from variational_pmf.code.load_store_matrices import store_X_U_V
from variational_pmf.code.helpers import recover_predictions, generate_M, calc_inverse_M
from run_different_k_real_data import read_data

"""
Run the Variational PMF algorithm for the best value of K found by running
<run_different_k_real_data.py>, and store the matrices U, V, and the predictions X in a file.
"""


def compute_RMSE(actual_vs_predicted):
	RMSE = sum([(y_t-y_p)**2 for (y_t,y_p) in actual_vs_predicted])/len(actual_vs_predicted)
	return RMSE


if __name__ == "__main__":
	X = read_data("./../../data/gi50_no_missing.txt")
	(I,J) = X.shape
	fraction_unknown = 0.1

	M = generate_M(I,J,fraction_unknown)
	M_inv = calc_inverse_M(M)
	K = 3

	outputfile = "recovered_matrices.txt"
	iterations = 10

	PMF = VariationalPMF(X,M,K)
	PMF.run(iterations=iterations,updates=1,calc_predictions=True,M_inv=M_inv)

	predicted_U = PMF.U
	predicted_V = PMF.V
	predicted_X = PMF.predicted_X

	# Store the predicted matrix X with U and V
	store_X_U_V(outputfile,predicted_X,predicted_U,predicted_V)

	# Now we plot the predictions vs the true values
	actual_vs_predicted = recover_predictions(M,X,predicted_X)
	(actual,predicted) = zip(*actual_vs_predicted)

	RMSE_predictions = compute_RMSE(actual_vs_predicted)
	RMSE_training = PMF.RMSE

	print "RMSE of predictions: %s" % RMSE_predictions
	print "RMSE of training data: %s" % RMSE_training

	bins_predictions = plt.figure(1)
	plt.title('Histogram of values - true values, predictions')
	plt.xlabel('values - actual vs predicted')
	plt.ylabel('no.')
	bins = [v * 0.1 for v in range(0,10*10)]
	plt.hist(actual,bins,alpha=0.5,label='actual')
	plt.hist(predicted,bins,alpha=0.5,label='predicted')
	
	# Also do a histogram plot of the differences between the actual and prediction
	differences = [math.fabs(v1-v2) for (v1,v2) in actual_vs_predicted]
	bins_differences_unknown = plt.figure(2)
	plt.title('Histogram of differences between the predictions and actual values')
	plt.xlabel('difference between actual and predicted')
	plt.ylabel('no.')
	bins = [v * 0.1 for v in range(0,10*5)]
	plt.hist(differences,bins)

	differences = [math.fabs(v1-v2) for (v1,v2) in zip(numpy.ndarray.flatten(X),numpy.ndarray.flatten(predicted_X))]
	bins_differences_all = plt.figure(3)
	plt.title('Histogram of differences between the predictions and actual values - including training data')
	plt.xlabel('difference between actual and predicted - all')
	plt.ylabel('no.')
	bins = [v * 0.1 for v in range(0,10*5)]
	plt.hist(differences,bins)

	# Plot the predictions - only the "unknown" values (1 entries in M)
	scatter_predictions = plt.figure(4)
	plt.title('Scatter plot of predictions vs true values')
	plt.xlabel('actual')
	plt.ylabel('predicted')
	plt.scatter(actual,predicted)
	plt.plot([-2,10],[-2,10]) #y=x line

	# Plot all the values, including the ones that are already known
	scatter_all = plt.figure(5)
	plt.title('Scatter plot of predictions vs true values - including training data')
	plt.xlabel('actual - all')
	plt.ylabel('predicted - all')
	plt.scatter(X,predicted_X)
	plt.plot([-2,10],[-2,10]) #y=x line

	# Show all the plots
	plt.show()