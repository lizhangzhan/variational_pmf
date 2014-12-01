Variational implementation of Probabilistic Matrix Factorization
===============

This class implements the scalable variational probabilistic matrix factorization algorithm presented in Kim and Choi (2013).

We try to deconstruct the (potentially sparse) observed matrix X:(I,J) into matrices U:(K,I) and V:(K,J), such that X =~ (U)T*V. (T = transpose). The (I,J) matrix M tells us which values in X are known (M_ij = 1 if know, 0 if unknown).

We assume U and V are subjected to Gaussian noise, u_ki ~ N(0,alpha_k) and v_kj ~ N(0,beta_k) (row-dependent). Similarly for each entry x_ij ~ N((u_i)T*v_j,tau).

For the variational approximation we assume each individual element in U and V is independently distributed as N(u_ki,s^u_ki) and N(v_kj,s^v_kj).

We also maintain a residual error matrix R to speed up computations, where:
	R_ij = X_ij - (sum over k)[(u_ki)T*v_kj]

Initialisation:
- We draw each s^u_ki and s^v_kj from an Inverse Gaussian distribution with parameters shape=1, scale=1 (this is the conjugate informative prior for the normal distribution)
- u_ki ~ N(0,1)
- v_kj ~ N(0,1)
- alpha_k, beta_k, tau are all initialized using the derived updates