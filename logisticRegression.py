# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import random
import numpy as np
import sys
from numpy.linalg import inv
# ==============logistic Regression==================================
#Logistic Regression
#Extracts coordinates, labels from classification.txt
#Finds best fit using the logisctic regression algo given in class
def main():
	# load txt file
	f = open("classification.txt", 'r')
	result_matrix = []
	for line in f.readlines():
	    values_as_strings = line.split(',')
	    arr = np.array(map(float, values_as_strings))
	    result_matrix.append(arr)
	X = np.array(result_matrix)

	coordinates = np.array(X[:,0:3])
	labels = np.array(X[:,4])
	#pad coordinates to make N x d+1 dimensionality (2,000 x 4)
	padded_coordinates = np.insert(coordinates, 0, 1, axis=1)

	#randomly choose weights having dx1 (4,)dimentionality
	weights = np.random.uniform(low=-0.01, high=0.01, size=(len(padded_coordinates[0])))
	print "Random weights", weights

	print 'Best fit (weights) for data in classification.txt using Logistic Regression is: \n', logisticRegression(padded_coordinates, labels, weights)

#Runs 7000 times and updates weights by subtracting the product of
#error gradient and eta(learning rate) from weights.
#Returns the weights updated in 7000 iterations
def logisticRegression(x, y, w):
	#Max number of iterations given by assignment
	iterations = 7000
	#eta should be "small" enough
	eta = 0.01
	N = len(x)
	while iterations > 0:
		ein_gradient = gradient_calc(N,y,w,x)
		w = np.subtract(w, eta*ein_gradient)
		iterations -= 1
	return w

#Calculates the gradient using the formula derived in class
#Formala: -1/N * summation from 1 to N (1/1+e^(y * w * x)  * y * x)
def gradient_calc(N,y,w,x):
	ein = 0
	for i in range(N):
		ein += (1/(1+np.exp(y[i]*(np.dot(w.T,x[i]))))) * (y[i]*x[i])
	return ((-1/N)*ein)


if __name__ == "__main__":
    main()