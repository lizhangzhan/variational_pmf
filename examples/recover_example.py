import sys
sys.path.append("../../.")
from variational_pmf.code.variational_pmf import VariationalPMF
from variational_pmf.code.load_store_matrices import load_X_U_V, store_X_U_V

import numpy

"""
Uses the Variational PMF method to recover the original matrices U, V from the observed
X in <source> with known values specified by 1's in M, and stores these together with the 
predicted X=U^T*V in <target>.
"""

def recover(M,source,target,iterations):
	(I,J,K,X,U,V) = load_X_U_V(source)

	PMF = VariationalPMF(X,M,K)
	PMF.run(iterations,updates=10)

	predicted_U = PMF.U
	predicted_V = PMF.V
	predicted_X = PMF.predicted_X

	# Write predicted_X, U, V to output file
	store_X_U_V(target,predicted_X,predicted_U,predicted_V)

	return


if __name__ == "__main__":
	recover(numpy.ones([100,80]),"generated.txt","recovered.txt",20)