import numpy, random, math
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
		self.RMSE = 0
		self.F_q = 0
		return


	def run(self,iterations):
		# Initialize a (K,I) matrix U, S_U, (K,J) matrix V, S_V, (I,J) matrix R
		#self.initialize()
		self.U = numpy.array([
			[ 2.20293498, -0.12139367,  0.67800237,  2.27199757],
			[-1.85369011, -0.32077793, -0.11329014,  0.40277643]
		])
		self.S_U = [[1,1,1,1],[2,2,2,2]]
		self.V = numpy.array([
			[ 1.23343862, -3.47311768, -4.43899942,  1.18449626, -2.20931011],
			[-0.07824337,  3.10808451, -1.37756161,  0.66224362,  0.04654989]
		])
		self.S_V = [[3,3,3,3,3],[2,2,2,2,2]]
		self.R = self.X - numpy.dot(self.U.transpose(),self.V)

		# self.omega_U will store an array of tuples Mi, containing the indices
		# j for which Mij is 1. Similarly for self.omega_V, Mj, indices i; and
		# self.omega, indices (i,j)
		self.compute_omega()

		# Initialize the hyperparameters alpha_k, beta_k, tau
		#self.update_hyperparameters()
		self.alpha = [1,2]
		self.beta = [3,2]
		self.tau = 2

		# Then repeatedly: update U, update V, update hyperparams
		for i in range(0,iterations):
			#self.update_U()
			#self.update_V()
			self.update_hyperparameters()
			
			self.calc_statistics()
			print self.F_q

		# Calculate the RMSE and variational lower bound F(q)
		self.calc_statistics()

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

		self.R = self.X - numpy.dot(self.U.transpose(),self.V)


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


	def calc_statistics(self):
		# See Kim and Choi for these definitions
		self.RMSE = sum([sum([self.R[i][j]**2 for j in range(0,self.J)])for i in range(0,self.I)]) / (self.I * self.J)
		F = 0.0
		for i in range(0,self.I):
			for j in range(0,self.J):
				E_ij = self.R[i][j]**2 + sum([
					self.U[k][i]**2 * self.S_V[k][j] + self.V[k][j]**2 * self.S_U[k][i] + self.S_U[k][i]*self.S_V[k][j]
					for k in range(0,self.K)])
				F_ij = -(self.tau/2.0)*E_ij - (1.0)/(2.0)*math.log(2*math.pi*self.tau) # natural log
				F += F_ij
		for i in range(0,self.I):
			for k in range(0,self.K):
				F_U_ki = -1.0/(2.0*self.alpha[k]) * (self.U[k][i]**2 + self.S_U[k][i]) + math.log(self.S_U[k][i]*self.alpha[k])/2.0 + 1.0/2.0
				F += F_U_ki
		for j in range(0,self.J):
			for k in range(0,self.K):
				F_V_kj = -1.0/(2.0*self.beta[k]) * (self.V[k][j]**2 + self.S_V[k][j]) + math.log(self.S_V[k][j]*self.beta[k])/2.0 + 1.0/2.0
				F += F_V_kj
		self.F_q = F
		return


if __name__ == "__main__":
	# Original matrices:
	alpha = [1,2]
	beta = [3,2]
	tau = 2
	#original_U = numpy.array([
	#	[random.normalvariate(0,alpha[0]) for i in range(0,4)],
	#	[random.normalvariate(0,alpha[1]) for i in range(0,4)]
	#])
	#original_V = numpy.array([
	#	[random.normalvariate(0,beta[0]) for i in range(0,5)],
	#	[random.normalvariate(0,beta[1]) for i in range(0,5)]
	#])
	#X = numpy.array([
	#	[
	#		random.normalvariate(element,tau)
	#		for element in row
	#	]
	#	for row in numpy.dot(original_U.transpose(),original_V)
	#])
	original_U = numpy.array([
		[ 2.20293498, -0.12139367,  0.67800237,  2.27199757],
		[-1.85369011, -0.32077793, -0.11329014,  0.40277643]
	])
	original_V = numpy.array([
		[ 1.23343862, -3.47311768, -4.43899942,  1.18449626, -2.20931011],
		[-0.07824337,  3.10808451, -1.37756161,  0.66224362,  0.04654989]
	])
	X = numpy.array([
		[  2.32431992, -11.7913782,   -7.74228389,   2.59442851,  -5.51584412],
 		[  0.85785318,  -3.06949111,   0.72351582,   0.43589775,   0.13710034],
 		[  1.51937854,   0.01188825,  -4.88318157,  -1.06202971,  -2.360093  ],
 		[  1.28276217,  -5.64957929,  -9.51851513,   3.5672679,   -6.33356818]
	])

	M = numpy.array([
		[1,	1,	1,	1,	1	],
		[1, 1,	1,	1,	1	],
		[1,	1,	1,	1,	1	],
		[1,	1,	1,	1,	1	]
	])
	K = 2
	
	PMF = VariationalPMF(X,M,K)
	PMF.run(100)

	X_predicted = numpy.dot(PMF.U.transpose(),PMF.V)

	print
	print "Original U: ", original_U
	print "Original V: ", original_V
	print "X: ", X
	print "tau, alpha, beta:", PMF.tau, PMF.alpha, PMF.beta
	print "U: ", PMF.U
	print "V: ", PMF.V
	print "S_U: ", PMF.S_U
	print "S_V: ", PMF.S_V
	print "R: ", PMF.R

	print "X_predicted:",X_predicted