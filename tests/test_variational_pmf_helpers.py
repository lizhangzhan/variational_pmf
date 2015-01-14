from variational_pmf.code.variational_pmf_helpers import calc_F_q, calc_RMSE, calc_NRMSE, calc_performance
from variational_pmf.code.variational_pmf import VariationalPMF
import numpy, math


def test_calc_RMSE():
	R = numpy.array([[1,2],[3,4]])
	omega = [(0,0),(0,1),(1,1)]

	expected_RMSE = math.sqrt(7)
	RMSE = calc_RMSE(R,omega)

	assert RMSE == expected_RMSE


def test_calc_NRMSE():
	RMSE = 7
	X_min = 5
	X_max = 10

	expected_NRMSE = 1.4
	NRMSE = calc_NRMSE(RMSE,X_min,X_max)

	assert NRMSE == expected_NRMSE


def test_calc_performance():
	predicted_X = numpy.array([[1,2,3],[4,5,6]])
	true_X = numpy.array([[6,5,4],[3,2,1]])
	I = 2
	J = 3
	M_inv = numpy.array([[1,0,1],[1,1,0]])
	X_min = 1
	X_max = 5

	expected_RMSE = 3
	expected_NRMSE = 0.75
	(RMSE,NRMSE) = calc_performance(predicted_X,true_X,I,J,M_inv,X_min,X_max)

	assert RMSE == expected_RMSE
	assert NRMSE == expected_NRMSE


def test_calc_F_q():
	X = numpy.array([[1,2],[3,4]])
	M = numpy.array([[1,0],[0,1]])
	K = 2
	I = 2
	J = 2

	omega = [(0,0),(1,1)]

	alpha = numpy.array([0.1,0.2])
	beta = numpy.array([0.3,0.4])
	tau = 0.5
	U = numpy.array([[0.6,0.7],[0.8,0.9]])
	S_U = numpy.array([[1.0,1.1],[1.2,1.3]])
	V = numpy.array([[1.4,1.5],[1.6,1.7]])
	S_V = numpy.array([[1.8,1.9],[2.0,2.1]])
	R = numpy.array([[(1-(0.6*1.4)-(0.8*1.6)),	(2-(0.6*1.5)-(0.8*1.7))],
					 [(3-(0.7*1.4)-(0.9*1.6)),	(4-(0.7*1.5)-(0.9*1.7))]])
	
	E_00 = R[0][0]**2 + U[0][0]**2 * S_V[0][0] + V[0][0]**2 * S_U[0][0] + S_U[0][0] * S_V[0][0] + \
		   U[1][0]**2 * S_V[1][0] + V[1][0]**2 * S_U[1][0] + S_U[1][0] * S_V[1][0]
	F_00 = -0.5*tau*E_00 - 0.5*math.log(2*math.pi/tau)

	E_11 = R[1][1]**2 + U[0][1]**2 * S_V[0][1] + V[0][1]**2 * S_U[0][1] + S_U[0][1] * S_V[0][1] + \
		   U[1][1]**2 * S_V[1][1] + V[1][1]**2 * S_U[1][1] + S_U[1][1] * S_V[1][1]
	F_11 = -0.5*tau*E_11 - 0.5*math.log(2*math.pi/tau)

	F_U_00 = -0.5*alpha[0]*(U[0][0]**2+S_U[0][0])+0.5*math.log(S_U[0][0]*alpha[0])+0.5
	F_U_01 = -0.5*alpha[0]*(U[0][1]**2+S_U[0][1])+0.5*math.log(S_U[0][1]*alpha[0])+0.5
	F_U_10 = -0.5*alpha[1]*(U[1][0]**2+S_U[1][0])+0.5*math.log(S_U[1][0]*alpha[1])+0.5
	F_U_11 = -0.5*alpha[1]*(U[1][1]**2+S_U[1][1])+0.5*math.log(S_U[1][1]*alpha[1])+0.5

	F_V_00 = -0.5*beta[0]*(V[0][0]**2+S_V[0][0])+0.5*math.log(S_V[0][0]*beta[0])+0.5
	F_V_01 = -0.5*beta[0]*(V[0][1]**2+S_V[0][1])+0.5*math.log(S_V[0][1]*beta[0])+0.5
	F_V_10 = -0.5*beta[1]*(V[1][0]**2+S_V[1][0])+0.5*math.log(S_V[1][0]*beta[1])+0.5
	F_V_11 = -0.5*beta[1]*(V[1][1]**2+S_V[1][1])+0.5*math.log(S_V[1][1]*beta[1])+0.5

	expected_F_q = F_00 + F_11 + F_U_00 + F_U_01 + F_U_10 + F_U_11 + F_V_00 + F_V_01 + F_V_10 + F_V_11
	F_q = calc_F_q(U,V,S_U,S_V,R,omega,tau,alpha,beta,I,J,K)

	assert F_q == expected_F_q