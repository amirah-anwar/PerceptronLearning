# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import random
import numpy as np
import sys

# ==============Perceptron Algorithm==================================
#Linear classification
#Extracts coordinates, labels from classification.txt
#Randomly chooses weights for each coordinate
#Finds final weights using perceptron algorithm
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
	labels = np.array(X[:,3])

	#pad coordinates to make d+1 dimensionality
	padded_coordinates = np.insert(coordinates, 0, 1, axis=1)

	#randomly choose weights for each coordinate having same dimensioanolity
	weights = np.random.uniform(low=0.0, high=0.5, size=(len(padded_coordinates),len(padded_coordinates[0])))

	#find final weights using perceptron algorithm
	np.set_printoptions(threshold='nan')
	print 'Calculating final weights using perceptron algorithm...'
	print 'Calculated weights', perceptron(padded_coordinates, weights, labels)

#Calculates violated constraints
#Updates respective weights of coordinates which violate constraints
#Keeps on updating weights untill there are no more violated constraints
def perceptron(padded_coordinates, weights, labels):
	learning_rate = 0.1

	#find violated constraints
	violations = calculateViolations(padded_coordinates, weights, labels)

	#while there are violated constraints keep on changing weights
	while len(violations['+ve']) != 0 or len(violations['-ve']) != 0:

		#add learning rate in wieght if (w.T).coordinate < 0 and label is +ve
		if len(violations['+ve']) != 0:
			index = int(violations['+ve'][0][1])
			weights[index] = np.add(weights[index], learning_rate*padded_coordinates[index])

		#subtract learning rate in wieght if (w.T).coordinate > 0 and label is -ve
		elif len(violations['-ve']) != 0:
			index = int(violations['-ve'][0][1])
			weights[index] = np.subtract(weights[index], learning_rate*padded_coordinates[index])

		violations = calculateViolations(padded_coordinates, weights, labels)

	return weights

#Stores the result that violates constraints
def calculateViolations(coordinates, weights, labels):
	index = 0
	violations = {}
	pos_viol = []
	neg_viol = []

	for coordinate in coordinates:
		result = np.dot((weights[index].T),coordinate)
		if result < 0 and labels[index] == 1:
			pos_viol.append(np.array([result, index]))
		elif result > 0 and labels[index] == -1:
			neg_viol.append(np.array([result, index]))
		index += 1
	
	violations['+ve'] = np.array(pos_viol)
	violations['-ve'] = np.array(neg_viol)
	return violations

if __name__ == "__main__":
    main()