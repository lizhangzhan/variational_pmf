from variational_pmf import VariationalPMF
from load_store_matrices import load_X_U_V, store_X_U_V, write_matrix, write_list
from helpers import generate_M, calc_inverse_M
import numpy, random, math

"""
Methods for running the Variational PMF algorithm <no_per_k> times for each of
the values for K in <k_range>, storing the variational lower bounds F(q) in a 
file <fout>. This allows us to plot how good different values of K are for a 
specific problem, and determine which value to use for our final inference.
Each time we randomly initialize M, with <fraction> of the values missing.
"""

def try_different_k(filename,X,K_range,no_per_K,fraction=0.1,iterations=50,updates=10):
	(I,J) = X.shape
	variational_all = []
	RMSE_all = []
	NRMSE_all = []
	for K in K_range:
		variational_lower_bounds = []
		RMSEs_predict = []
		NRMSEs_predict = []
		for i in range(1,no_per_K+1):
			print "K=%s, i=%s" % (K,i)

			# Generate M
			M = generate_M(I,J,fraction)
			M_inv = calc_inverse_M(M)

			# Recover the matrices U,V
			PMF = VariationalPMF(X,M,K)
			PMF.run(iterations,updates,True,M_inv)

			predicted_X = PMF.predicted_X
			U = PMF.U
			V = PMF.V

			# Retrieve the variational lower bound and (N)RMSE from this run, and store it
			RMSE_predict = PMF.RMSE_predict
			NRMSE_predict = PMF.NRMSE_predict
			
			variational_lower_bounds.append(PMF.F_q)
			RMSEs_predict.append(RMSE_predict)
			NRMSEs_predict.append(NRMSE_predict)

		# Add the variational lower bounds and (N)RMSE values for this value of K to the lists
		variational_all.append(variational_lower_bounds)
		RMSE_all.append(RMSEs_predict)
		NRMSE_all.append(NRMSEs_predict)

	# We then return: the best of each K value, the avr, and all of the values
	def averages(matrix):
		return [sum(l)/len(l) for l in matrix]
	def bests(matrix):
		return [max(l) for l in matrix]

	variational_avr = averages(variational_all)
	variational_best = bests(variational_all)
	RMSE_avr = averages(RMSE_all)
	RMSE_best = bests(RMSE_all)
	NRMSE_avr = averages(NRMSE_all)
	NRMSE_best = bests(NRMSE_all)

	# We now store this info in the file <filename>
	fout = open(filename,'w')

	def print_performances(name,all_values,avr,best):
		write_matrix(fout,"%s - all" % name,all_values)
		write_list(fout,"%s - average" % name,avr)
		write_list(fout,"%s - best" % name,best)

	print_performances("Variational",variational_all,variational_avr,variational_best)
	print_performances("RMSE predictions",RMSE_all,RMSE_avr,RMSE_best)
	print_performances("NRMSE predictions",NRMSE_all,NRMSE_avr,NRMSE_best)

	fout.close()
	
	return