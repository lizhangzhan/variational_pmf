import numpy

"""
This file contains methods for storing or reading the matrices X, U, V, as well
as the dimensions I, J, K.
"""

def load_X_U_V(filename):
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


def store_X_U_V(filename,X,U,V):
	(I,J) = X.shape
	(K,_) = U.shape

	fout = open(filename,'w')
	fout.write("%s\t%s\t%s\n" % (I,J,K))
	write(fout,"X",X)
	write(fout,"U",U)
	write(fout,"V",V)

	return


# Helper for write_X_U_V
def write(fout,name,matrix):
	fout.write("%s\n" % name)
	text = "\n".join(["\t".join([str(element) for element in row]) for row in matrix])+"\n"
	fout.write(text)
	return