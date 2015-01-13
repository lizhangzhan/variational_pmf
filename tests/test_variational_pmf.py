from variational_pmf.code.variational_pmf import VariationalPMF
import numpy


def test_compute_omega():
	X = numpy.array([[1,2],[3,4],[5,6]])
	M = numpy.array([[0,1],[1,0],[1,1]])
	K = 2

	PMF = VariationalPMF(X,M,K)
	PMF.compute_omega()

	expected_omega = [(0,1),(1,0),(2,0),(2,1)]
	expected_omega_I = [[1],[0],[0,1]]
	expected_omega_J = [[1,2],[0,2]]

	assert numpy.array_equal(PMF.omega,expected_omega)
	assert numpy.array_equal(PMF.omega_I,expected_omega_I)
	assert PMF.omega_J[0] == expected_omega_J[0] and PMF.omega_J[1] == expected_omega_J[1]


def test_updates():
	# We create a simple test case by overwriting the random initialization,
	# and then test each of the updates.
	X = numpy.array([[-1,2],[3,4]])
	M = numpy.array([[0,1],[1,1]])
	K = 2

	# Reset values
	r_alpha = numpy.array([0.1,0.2])
	r_beta = numpy.array([0.3,0.4])
	r_tau = 0.5
	r_U = numpy.array([[0.6,0.7],[0.8,0.9]])
	r_S_U = numpy.array([[1.0,1.1],[1.2,1.3]])
	r_V = numpy.array([[1.4,1.5],[1.6,1.7]])
	r_S_V = numpy.array([[1.8,1.9],[2.0,2.1]])
	r_R = numpy.array([[0,						(2-(0.6*1.5)-(0.8*1.7))],
					   [(3-(0.7*1.4)-(0.9*1.6)),(4-(0.7*1.5)-(0.9*1.7))]])


	PMF = VariationalPMF(X,M,K)
	PMF.run(0)


	def reset():
		# Overwrite everything for the test case
		PMF.alpha[:] = r_alpha
		PMF.beta[:] = r_beta
		PMF.tau = r_tau
		PMF.U[:] = r_U
		PMF.S_U[:] = r_S_U
		PMF.V[:] = r_V
		PMF.S_V[:] = r_S_V
		PMF.R[:] = r_R


	# Test updating the hyperparameters
	reset()
	PMF.update_hyperparameters()

	expected_alpha = numpy.array([
		((0.6**2+1.0)+(0.7**2+1.1)) / 2.0,
		((0.8**2+1.2)+(0.9**2+1.3)) / 2.0
	])
	expected_beta = numpy.array([
		((1.4**2+1.8)+(1.5**2+1.9)) / 2.0,
		((1.6**2+2.0)+(1.7**2+2.1)) / 2.0
	])
	# tau = E_21 + E_12 + E_22 (because M_11=0)
	E_12 = (2-(0.6*1.5+0.8*1.7))**2 + (0.6**2*1.9+1.5**2*1.0+1.0*1.9) + (0.8**2*2.1+1.7**2*1.2+2.1*1.2)
	E_21 = (3-(0.7*1.4+0.9*1.6))**2 + (0.7**2*1.8+1.4**2*1.1+1.1*1.8) + (0.9**2*2.0+1.6**2*1.3+2.0*1.3)
	E_22 = (4-(0.7*1.5+0.9*1.7))**2 + (0.7**2*1.9+1.5**2*1.1+1.1*1.9) + (0.9**2*2.1+1.7**2*1.3+2.1*1.3)
	expected_tau = (E_21 + E_12 + E_22) / 3.0

	assert numpy.array_equal(PMF.alpha,expected_alpha)
	assert numpy.array_equal(PMF.beta,expected_beta)
	assert PMF.tau == expected_tau


	# Test updating U. Testing U itself is hard because the values of R_ij change,
	# so to verify you would have to run the program itself anyways.
	reset()
	PMF.update_U()
	expected_S_U = numpy.array([
		[
			1.0/(1.0/0.1 + (1/0.5) * (1.5**2+1.9)),
			1.0/(1.0/0.1 + (1/0.5) * (1.4**2+1.8 + 1.5**2+1.9))
		],
		[
			1.0/(1.0/0.2 + (1/0.5) * (1.7**2+2.1)),
			1.0/(1.0/0.2 + (1/0.5) * (1.6**2+2.0 + 1.7**2+2.1))
		]
	])
	assert numpy.array_equal(PMF.S_U,expected_S_U)

	assert True