import sys, numpy
sys.path.append("../../.")
from variational_pmf.code.run_different_k import try_different_k
from variational_pmf.code.load_store_matrices import read_matrix

"""
This file loads in the data in gi50_no_missing.txt, and then runs the 
Variational PMF algorithm for a range of values for K on the matrix. We store
the RMSE and variational lower bounds in a file performances.txt, which we can
plot to show the best values of K for this data set.

The data set has the following format:

NSC			\t	<cell line name>\t	<cell line name>\t	...	\r\n
<NSC number>\t	drug sensitivity\t	drug sensitivity\t	...	\r\n
<NSC number>\t	drug sensitivity\t	drug sensitivity\t	...	\r\n
...

So we get rid of the first line, and the first value.
"""

def read_data(filename):
	# Read in the data from the file into X
	fin = open(filename,"r")
	fin.readline()
	X = numpy.array([
		[float(v) for v in line.split("\n")[0].split("\r")[0].split("\t")[1:]]
		for (_,line) in enumerate(fin)
	])
	return X



if __name__ == "__main__":
	outputfile = "performances.txt"
	K_range = range(1,10+1)
	no_per_K = 5
	fraction_unknown = 0.1
	iterations = 10
	updates = 5

	X = read_data("gi50_no_missing.txt")	

	try_different_k(filename=outputfile,X=X,K_range=K_range,no_per_K=no_per_K,
					fraction=fraction_unknown,iterations=iterations,updates=updates)
	