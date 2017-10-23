# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import random
import numpy as np
import sys
from numpy.linalg import inv
# ==============Linear Regression==================================
#Linear Regression
#Extracts coordinates, labels from linear-regression.txt
#Finds the optimized weight using the simplified matrix formula derived in class
def main():
	# load txt file
	f = open("linear-regression.txt", 'r')
	result_matrix = []
	for line in f.readlines():
	    values_as_strings = line.split(',')
	    arr = np.array(map(float, values_as_strings))
	    result_matrix.append(arr)
	X = np.array(result_matrix)

	coordinates = np.array(X[:,0:2])
	labels = np.array(X[:,2])
	#pad coordinates to make N x d+1 dimensionality (2,000 x 4)
	padded_coordinates = np.insert(coordinates, 0, 1, axis=1)

	print 'w_opt for Linear Regression is: \n', linearRegression(padded_coordinates, labels)

#Linear Regression using formula:
#(D.T*D)^-1*Dy where y is the dependent variable
def linearRegression(D, y):
	w_opt = np.dot(inv(np.dot(D.T,D)),np.dot(D.T,y))
	return w_opt

if __name__ == "__main__":
    main()