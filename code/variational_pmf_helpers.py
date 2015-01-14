import math

"""
This file contains helper methods for the PMF algorithm, like computing the variational
lower bound F_q, and the (N)RMSE.
"""

def calc_F_q(U,V,S_U,S_V,R,omega,tau,alpha,beta,I,J,K):
	# See Kim and Choi for these definitions
	F = 0.0
	for (i,j) in omega:
		E_ij = R[i][j]**2 + sum([
			U[k][i]**2 * S_V[k][j] + V[k][j]**2 * S_U[k][i] + S_U[k][i]*S_V[k][j]
			for k in range(0,K)])
		F_ij = -tau/2.0*E_ij - (1.0)/(2.0)*math.log(2*math.pi/tau) # natural log
		F += F_ij
	for i in range(0,I):
		for k in range(0,K):
			F_U_ki = -alpha[k]/2.0 * (U[k][i]**2 + S_U[k][i]) + math.log(S_U[k][i]*alpha[k])/2.0 + 1.0/2.0
			F += F_U_ki
	for j in range(0,J):
		for k in range(0,K):
			F_V_kj = -beta[k]/2.0 * (V[k][j]**2 + S_V[k][j]) + math.log(S_V[k][j]*beta[k])/2.0 + 1.0/2.0
			F += F_V_kj
	return F


def calc_RMSE(R,omega):
	return math.sqrt(sum([R[i][j]**2 for (i,j) in omega]) / float(len(omega)))
		

def calc_NRMSE(RMSE,X_min,X_max):
	return RMSE / float(X_max - X_min)


def calc_performance(predicted_X,true_X,I,J,M_inv,X_min,X_max):
	predict_indices = []
	for i in range(0,I):
		for j in range(0,J):
			if M_inv[i][j] == 1:
				predict_indices.append((i,j))

	square_errors = [(true_X[i][j] - predicted_X[i][j])**2 for (i,j) in predict_indices]
	
	RMSE_predict = math.sqrt(sum(square_errors) / len(predict_indices))
	NRMSE_predict = RMSE_predict / float(X_max - X_min)
	return (RMSE_predict,NRMSE_predict)


def calc_omega():
	self.omega = []
	for i in range(0,self.I):
		for j in range(0,self.J):
			if self.M[i,j] == 1:
				self.omega.append((i,j))
	self.omega = numpy.array(self.omega)

def calc_omega_I():
	self.omega_I = numpy.empty([self.I],object) # the object is a little hack to allow an array of different sized arrays
	for i in range(0,self.I):
		self.omega_I[i] = [j for (j,m_ij) in enumerate(self.M[i]) if m_ij == 1]

def calc_omega_J():
	self.omega_J = numpy.empty([self.J],object)
	for j in range(0,self.J):
		self.omega_J[j] = [i for (i,m_ij) in enumerate(self.M[:,j]) if m_ij == 1]
