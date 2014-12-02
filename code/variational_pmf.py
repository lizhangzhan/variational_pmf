import numpy, random
from scipy.stats import invgamma


class VariationalPMF:

	def __init__(self,X,M,K):
		# K is the order of the matrix decomposition, X is a num.py matrix.
		# M shows which entries are known (M_ij = 1 if known, 0 otherwise).
		self.X = X
		self.M = M
		self.K = K
		self.I = X.shape[0]
		self.J = X.shape[1]
		self.alpha = numpy.empty([K])
		self.beta = numpy.empty([K])
		self.tau = 0
		return


	def run(self,iterations):
		# Initialize a (K,I) matrix U, S_U, (K,J) matrix V, S_V, (I,J) matrix R
		self.initialize()

		# self.omega_U will store an array of tuples Mi, containing the indices
		# j for which Mij is 1. Similarly for self.omega_V, Mj, indices i; and
		# self.omega, indices (i,j)
		self.compute_omega()

		# Initialize the hyperparameters alpha_k, beta_k, tau
		self.update_hyperparameters()

		# Then repeatedly: update U, update V, update hyperparams
		for i in range(0,iterations):
			self.update_U()
			self.update_V()
			self.update_hyperparameters()

		return


	def initialize(self):
		# InvGamma RV with beta=1, alpha=1
		IG = invgamma(a=1)

		self.U = numpy.empty([self.K,self.I])
		self.S_U = numpy.empty([self.K,self.I])

		for k in range(0,self.K):
			for i in range(0,self.I):
				self.U[k][i] = random.normalvariate(0,1)
				self.S_U[k][i] = IG.rvs()

		self.V = numpy.empty([self.K,self.J])
		self.S_V = numpy.empty([self.K,self.J])

		for k in range(0,self.K):
			for j in range(0,self.J):
				self.V[k][j] = random.normalvariate(0,1)
				self.S_V[k][j] = IG.rvs()

		self.R = numpy.empty([self.I,self.J])
		for i in range(0,self.I):
			for j in range(0,self.J):
				if self.M[i][j] == 1:
					self.R[i][j] = self.X[i][j] - numpy.dot(self.U[:,i].transpose(),self.V[:,j])
				else: 
					self.R[i][j] = 0

		#print "U: ", self.U
		#print "S_U: ", self.S_U
		#print "V: ", self.V
		#print "S_V: ", self.S_V
		#print "R: ", self.R

		return


	def compute_omega(self):
		self.omega = []
		for i in range(0,self.I):
			for j in range(0,self.J):
				if self.M[i,j] == 1:
					self.omega.append((i,j))
		self.omega = numpy.array(self.omega)

		self.omega_I = numpy.empty([self.I],object) # the object is a little hack to allow an array of different sized arrays
		for i in range(0,self.I):
			self.omega_I[i] = [j for (j,m_ij) in enumerate(self.M[i]) if m_ij == 1]

		self.omega_J = numpy.empty([self.J],object)
		for j in range(0,self.J):
			self.omega_J[j] = [i for (i,m_ij) in enumerate(self.M[:,j]) if m_ij == 1]

		#print "Omega: ", self.omega
		#print "Omega_I: ", self.omega_I
		#print "Omega_J: ", self.omega_J

		return


	def update_hyperparameters(self):
		for k in range(0,self.K):
			self.alpha[k] = sum([(self.U[k][i])**2 + self.S_U[k][i] for i in range(0,self.I)]) / float(self.I)
			self.beta[k] = sum([(self.V[k][j])**2 + self.S_V[k][j] for j in range(0,self.J)]) / float(self.J)

		#print "Alpha: ", self.alpha
		#print "Beta: ", self.beta

		# Compute E_ij, then sum to give E
		E = 0.0
		for (i,j) in self.omega:
			E_ij = self.R[i][j]**2 + sum([((self.U[k][i]**2)*self.S_V[k][j] + (self.V[k][j]**2)*self.S_U[k][i] + self.S_U[k][i]*self.S_V[k][j]) for k in range(0,self.K)])
			E += E_ij
			#print self.R[i][j],self.U[k][i],self.V[k][j]
			#print "E_ij: ",i,j,E_ij

			#print "E:", i, j, E_ij

		self.tau = E / float(len(self.omega))

		#print "Tau: ", self.tau

		return


	def update_U(self):
		for i in range(0,self.I):
			for k in range(0,self.K):
				old_U_ki = self.U[k][i]
				self.S_U[k][i] = 1.0 / ((1.0/self.alpha[k]) + (1.0/self.tau) * sum([(self.V[k][j]**2+self.S_V[k][j]) for j in self.omega_I[i]]))
				self.U[k][i] = self.S_U[k][i] * ((1.0/self.tau) * sum([(self.R[i][j]-self.U[k][i]*self.V[k][j])*self.V[k][j] for j in self.omega_I[i]]))
				for j in self.omega_I[i]:
					self.R[i][j] = self.R[i][j] - (self.U[k][i]-old_U_ki) * self.V[k][j]
					
		#print
		#print "U: ", self.U
		#print "S_U: ", self.S_U

		#print "R: ", self.R
		#MSE = sum([sum([(self.X[i][j] - self.R[i][j])**2 for j in range(0,self.J)]) for i in range(0,self.I)])
		#print "MSE: ",MSE

		return


	def update_V(self):
		for j in range(0,self.J):
			for k in range(0,self.K):
				old_V_kj = self.V[k][j]
				self.S_V[k][j] = 1.0 / ((1.0/self.beta[k]) + (1.0/self.tau) * sum([(self.U[k][i]**2+self.S_U[k][i]) for i in self.omega_J[j]]))
				self.V[k][j] = self.S_V[k][j] * ((1.0/self.tau) * sum([(self.R[i][j]-self.V[k][j]*self.U[k][i])*self.U[k][i] for i in self.omega_J[j]]))
				for i in self.omega_J[j]:
					self.R[i][j] = self.R[i][j] - (self.V[k][j]-old_V_kj) * self.U[k][i]
		
		#print
		#print "V: ", self.V
		#print "S_V: ", self.S_V

		#print "R: ", self.R

		return


if __name__ == "__main__":
	# Original matrices:
	# U = [[1,2,3,4][1,2,3,4]]
	# V = [[5,6,7,8,9],[5,6,7,8,9]]
	#X = numpy.array([
	#	[1,	2,	3,	4,	5	],
	#	[99,3,	4,	5,	99	],
	#	[3,	4,	99,	6,	7	],
	#	[4,	5,	6,	7,	8	]
	#])

	#X = numpy.array([
	#	[0.01,0.012,0.014,0.016,0.018],
	#	[0.02,0.024,0.028,0.032,0.036],
	#	[0.03,0.036,0.042,0.048,0.054],
	#	[0.04,0.048,0.056,0.064,0.072]
	#])
	X = numpy.array([
		[10,12,14,16,18],
		[20,24,28,32,36],
		[30,36,42,48,54],
		[40,48,56,64,72]
	])
	M = numpy.array([
		[1,	1,	1,	1,	1	],
		[0, 1,	1,	1,	0	],
		[1,	1,	0,	1,	1	],
		[1,	1,	1,	1,	1	]
	])
	K = 2
	
	PMF = VariationalPMF(X,M,K)
	PMF.run(10)

	print
	print "tau, alpha, beta:", PMF.tau, PMF.alpha, PMF.beta
	print "U: ", PMF.U
	print "V: ", PMF.V
	print "S_U: ", PMF.S_U
	print "S_V: ", PMF.S_V
	print "R: ", PMF.R

	print "X:",numpy.dot(PMF.U.transpose(),PMF.V)