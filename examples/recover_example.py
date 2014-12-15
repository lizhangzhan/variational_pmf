import sys
sys.path.append("../../.")
from variational_pmf.code.variational_pmf import VariationalPMF
from generate_example import write
import numpy

"""
Uses the Variational PMF method to recover the original matrices U, V from the observed
X in <source> with known values specified by 1's in M, and stores these together with the 
predicted X=U^T*V in <target>.
"""

def recover(M,source,target,iterations):
	(I,J,K,X,U,V) = read_X_U_V(source)

	PMF = VariationalPMF(X,M,K)
	PMF.run(iterations,updates=10)

	predicted_U = PMF.U
	predicted_V = PMF.V
	predicted_X = numpy.dot(predicted_U.transpose(),predicted_V)

	# Write predicted_X, U, V to output file
	fout = open(target,"w")
	fout.write("%s\t%s\t%s\n" % (I,J,K))
	write(fout,"X",predicted_X)
	write(fout,"U",predicted_U)
	write(fout,"V",predicted_V)

	return


def read_X_U_V(filename):
	fin = open(filename,"r")
	(I,J,K) = [int(val) for val in fin.readline().split("\n")[0].split("\t")]

	def read_matrix(f,rows,cols):
		matrix = numpy.empty([rows,cols])
		f.readline()
		for r in range(0,rows):
			row = f.readline().split("\n")[0].split("\t")
			for c in range(0,cols):
				matrix[r][c] = row[c]
		return matrix

	X = read_matrix(fin,I,J)
	U = read_matrix(fin,K,I)
	V = read_matrix(fin,K,J)

	return (I,J,K,X,U,V)


if __name__ == "__main__":
	recover(numpy.ones([100,80]),"generated.txt","recovered.txt",20)
