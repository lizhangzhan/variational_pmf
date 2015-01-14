import sys
sys.path.append("../../../.")
from variational_pmf.code.load_store_matrices import store_X_U_V

import numpy, random, math

"""
Generates an example matrix of dimensions (I,J), with rank K, and stores the matrix
in a file <target>. Format of the file:
I<tab>J<tab>K<\n>
X<\n>
X11<tab>X12<tab>...<tab>X1J<\n>
X21<tab> ...
U<\n>
similar to X
V<\n>
similar to X
"""

def generate_X(I,J,K,target):
	alpha = numpy.ones(I)
	beta = numpy.ones(J)
	tau = 1
	
	U = numpy.array([
		[random.normalvariate(0,math.sqrt(alpha[k])) for i in range(0,I)]
		for k in range(0,K)
	])
	V = numpy.array([
		[random.normalvariate(0,math.sqrt(beta[k])) for j in range(0,J)]
		for k in range(0,K)
	])
	X = numpy.array([
		[
			random.normalvariate(element,math.sqrt(tau))
			for element in row
		]
		for row in numpy.dot(U.transpose(),V)
	])

	# Write to file
	store_X_U_V(target,X,U,V)

	return


if __name__ == "__main__":
	I = 100
	J = 80
	K = 2
	target = "generated.txt"
	generate_X(I,J,K,target)