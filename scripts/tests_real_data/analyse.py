import matplotlib.pyplot as plt

"""
File containing analyses for:
 - 	The results of running PMF for different K values on the real data, and making scatter 
	plots of the variational lower bound versus the NRMSE on the training data
  
"""


def scatter_plot_F_q_vs_NRMSE(performances_file):
	# Data is stored in <performances.txt>
	fin = open(performances_file,"r")

	# Read variational lower bounds
	l = fin.readline()
	l = fin.readline()
	variational_averages = []
	while l != "Variational - average\n":
		variational_averages.append([float(s) for s in l.split("'n")[0].split("\t")])
		l = fin.readline()

	# Read NRMSE values
	while l != "NRMSE predictions - all\n":
		l = fin.readline()
	l = fin.readline()
	NRMSE_averages = []
	while l != "NRMSE predictions - average\n":
		NRMSE_averages.append([float(s) for s in l.split("'n")[0].split("\t")])
		l = fin.readline()

	scatter_predictions = plt.figure()
	plt.title('Scatter plot of variational lower bound vs NRMSE')
	plt.xlabel('F_q')
	plt.ylabel('NRMSE')
	colours = [str(float(i)/len(variational_averages)) for i in range(0,len(variational_averages))]
	print colours
	for i in range(0,len(variational_averages)):
		plt.scatter(variational_averages[i],NRMSE_averages[i],c=colours[i],label=i+1)
	plt.legend(loc='upper left');
	plt.show()



if __name__ == "__main__":
	scatter_plot_F_q_vs_NRMSE("performances.txt")
	#scatter_plot_F_q_vs_NRMSE("../scripts/tests_fake_data/performances.txt")