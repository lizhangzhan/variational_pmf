import numpy, random, math
from scipy.stats import invgamma
from variational_pmf_helpers import calc_F_q, calc_RMSE, calc_NRMSE, calc_performance

"""
This file contains the implementation of the variational approach to the
Probabilistic Matrix Factorization algorithm.
"""

class VariationalPMF:

	def __init__(self,X,M,K):
		# K is the order of the matrix decomposition, X is a num.py matrix.
		# M shows which entries are known (M_ij = 1 if known, 0 otherwise).
		self.X = X
		self.M = M
		self.K = K
		(self.I,self.J) = X.shape
		self.alpha = numpy.empty([K])
		self.beta = numpy.empty([K])
		self.X_max = None
		self.X_min = None


	# Run the variational inference updates <iterations> times, providing an update of F_q,
	# (N)RMSE_train (RMSE of training data), and (N)RMSE_predict (RMSE of test data) if 
	# <calc_predictions> is True (in which case <M_inv> is a matrix storing which values are
	# to be predicted - M_ij = 1 if so)
	def run(self,iterations,updates=10,calc_predictions=False,M_inv=[]):
		# Initialize U, S_U, V, S_V, R
		self.initialize()
		self.compute_omega()

		# Initialize the hyperparameters alpha_k, beta_k, tau
		self.update_hyperparameters()
		self.calc_statistics(0,calc_predictions,M_inv)
		
		# Then repeatedly: update U, update V, update hyperparams
		i = 0
		for i in range(1,iterations+1):
			self.update_U()
			self.update_V()
			self.update_hyperparameters()

			if (updates > 0 and i % updates == 0):
				self.calc_statistics(i,calc_predictions,M_inv)

		# Calculate the RMSE and variational lower bound F(q), if we haven't done so already
		if (updates == 0 or i % updates != 0):
			self.calc_statistics(i,calc_predictions,M_inv)

		self.predicted_X = numpy.dot(self.U.transpose(),self.V)


	def initialize(self):
		# InvGamma RV with beta=1, alpha=1
		IG = invgamma(1)

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

		return


	def update_U(self):
		for i in range(0,self.I):
			for k in range(0,self.K):
				old_U_ki = self.U[k][i]
				self.S_U[k][i] = 1.0 / (self.alpha[k] + self.tau * sum([(self.V[k][j]**2+self.S_V[k][j]) for j in self.omega_I[i]]))
				self.U[k][i] = self.S_U[k][i] * (self.tau * sum([(self.R[i][j]+self.U[k][i]*self.V[k][j])*self.V[k][j] for j in self.omega_I[i]]))
				for j in self.omega_I[i]:
					self.R[i][j] = self.R[i][j] - (self.U[k][i]-old_U_ki) * self.V[k][j]
			
		return


	def update_V(self):
		for j in range(0,self.J):
			for k in range(0,self.K):
				old_V_kj = self.V[k][j]
				self.S_V[k][j] = 1.0 / (self.beta[k] + self.tau * sum([(self.U[k][i]**2+self.S_U[k][i]) for i in self.omega_J[j]]))
				self.V[k][j] = self.S_V[k][j] * (self.tau * sum([(self.R[i][j]+self.V[k][j]*self.U[k][i])*self.U[k][i] for i in self.omega_J[j]]))
				for i in self.omega_J[j]:
					self.R[i][j] = self.R[i][j] - (self.V[k][j]-old_V_kj) * self.U[k][i]
	
		return


	def update_hyperparameters(self):
		for k in range(0,self.K):
			self.alpha[k] = float(self.I) / sum([(self.U[k][i])**2 + self.S_U[k][i] for i in range(0,self.I)])
			self.beta[k] = float(self.J) / sum([(self.V[k][j])**2 + self.S_V[k][j] for j in range(0,self.J)])

		# Compute E_ij, then sum to give E
		E = 0.0
		for (i,j) in self.omega:
			E_ij = self.R[i][j]**2 + sum([((self.U[k][i]**2)*self.S_V[k][j] + (self.V[k][j]**2)*self.S_U[k][i] + self.S_U[k][i]*self.S_V[k][j]) for k in range(0,self.K)])
			E += E_ij
		self.tau = float(len(self.omega)) / E

		return


	# This function calculates both the variational lower bound, F_q, and the RMSE/NRMSE for
	# the training data values. If <calc_predictions> is true, it will also calculate the
	# performance of the predictions. All these values are printed to the command line.
	def calc_statistics(self,iteration,calc_predictions,M_inv):
		if self.X_min is None:
			self.X_min = min([self.X[i][j] for (i,j) in self.omega])
		if self.X_max is None:
			self.X_max = max([self.X[i][j] for (i,j) in self.omega])

		self.RMSE_train = calc_RMSE(self.R,self.omega)
		self.NRMSE_train = calc_NRMSE(self.RMSE_train,self.X_min,self.X_max)

		self.F_q = calc_F_q(self.U,self.V,self.S_U,self.S_V,self.R,self.omega,self.tau,self.alpha,self.beta,self.I,self.J,self.K)

		print "Iteration %s, RMSE training: %s, NRMSE training: %s, F_q: %s" % (iteration,self.RMSE_train,self.NRMSE_train,self.F_q)
		
		# Calculate the performance ((N)RMSE) on the test data
		if calc_predictions:
			predicted_X = numpy.dot(self.U.transpose(),self.V)
			(self.RMSE_predict,self.NRMSE_predict) = calc_performance(predicted_X,self.X,self.I,self.J,M_inv,self.X_min,self.X_max)
			print "RMSE predictions: %s, NRMSE predictions: %s" % (self.RMSE_predict,self.NRMSE_predict)

		return