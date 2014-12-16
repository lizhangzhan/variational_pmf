from variational_pmf import VariationalPMF
from load_store_matrices import load_X_U_V, store_X_U_V, write_matrix, write_list
import numpy, random

"""
Methods for running the Variational PMF algorithm <no_per_k> times for each of
the values for K in <k_range>, storing the variational lower bounds F(q) in a 
file <fout>. This allows us to plot how good different values of K are for a 
specific problem, and determine which value to use for our final inference.
Each time we randomly initialize M, with <fraction> of the values missing.
"""

def try_different_k(filename,X,K_range,no_per_K,fraction=0.1,iterations=50,updates=10):
	(I,J) = X.shape
	variational_lower_bounds = []
	RMSEs = []
	for K in K_range:
		bounds = []
		errors = []
		for i in range(1,no_per_K+1):
			print "K=%s, i=%s" % (K,i)

			# Generate M
			M = generate_M(I,J,fraction)

			# Recover the matrices U,V
			PMF = VariationalPMF(X,M,K)
			PMF.run(iterations,updates)

			predicted_X = PMF.predicted_X
			U = PMF.U
			V = PMF.V

			actual_vs_predicted = recover_predictions(M,X,predicted_X)
			RMSE = compute_RMSE(actual_vs_predicted)

			bounds.append(PMF.F_q)
			errors.append(RMSE)

		RMSEs.append(errors)
		variational_lower_bounds.append(bounds)

	# We then return: the best of each K value, the avr, and all of the values
	variational_avr = [sum(l)/len(l) for l in variational_lower_bounds]
	variational_best = [max(l) for l in variational_lower_bounds]
	RMSEs_avr = [sum(l)/len(l) for l in RMSEs]
	RMSEs_best = [min(l) for l in RMSEs]

	# We now store this info in the file <filename>
	fout = open(filename,'w')
	write_matrix(fout,"Variational - all",variational_lower_bounds)
	write_list(fout,"Variational - average",variational_avr)
	write_list(fout,"Variational - best",variational_best)
	write_matrix(fout,"RMSE - all",RMSEs)
	write_list(fout,"RMSE - average",RMSEs_avr)
	write_list(fout,"RMSE - best",RMSEs_best)
	fout.close()
	
	return


def compute_RMSE(actual_vs_predicted):
	RMSE = sum([(y_t-y_p)**2 for (y_t,y_p) in actual_vs_predicted])/len(actual_vs_predicted)
	return RMSE


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
		M[v / I][v % J] = 0
	return M