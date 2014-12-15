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

def generate(I,J,K,target):
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
	f = open(target,'w')
	f.write("%s\t%s\t%s\n" % (I,J,K))
	write(f,"X",X)
	write(f,"U",U)
	write(f,"V",V)

	return


def write(fout,name,matrix):
	fout.write("%s\n" % name)
	text = "\n".join(["\t".join([str(element) for element in row]) for row in matrix])+"\n"
	fout.write(text)


if __name__ == "__main__":
	I = 100
	J = 80
	K = 2
	target = "generated.txt"
	generate(I,J,K,target)