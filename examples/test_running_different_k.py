import sys

import generate_example, recover_example
sys.path.append("../../.")
from variational_pmf.code.run_different_k import try_different_k
from variational_pmf.code.load_store_matrices import load_X_U_V

"""
In this example we use the matrices in the file generated.txt to try different
values of K for recovering U and V, and store how good the results are (storing
the RMSE and variational lower bound in a file performances.txt).
"""

if __name__ == "__main__":
	outputfile = "performances.txt"
	K_range = range(1,20+1)
	no_per_K = 10
	fraction_unknown = 0.1
	iterations = 50
	updates = 10

	generated = "generated.txt"
	(I,J,K,X,U,V) = load_X_U_V(generated)

	try_different_k(filename=outputfile,X=X,K_range=K_range,no_per_K=no_per_K,
					fraction=fraction_unknown,iterations=iterations,updates=updates)
	