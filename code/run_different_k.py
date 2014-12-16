from variational_pmf import VariationalPMF
from load_store_matrices import load_X_U_V, store_X_U_V
import numpy

"""
Methods for running the Variational PMF algorithm <no_per_k> times for each of
the values for K in <k_range>, storing the variational lower bounds F(q) in a 
file <fout>. This allows us to plot how good different values of K are for a 
specific problem, and determine which value to use for our final inference.
"""

def try_different_k(X,M,k_range,no_per_k):
	variational_lower_bounds = []
	for k in k_range:
		bounds = []
		for i in range(0,no_per_k):
			# Recover the matrices U,V
