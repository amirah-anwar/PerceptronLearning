# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import random
import numpy as np
import sys
import matplotlib.pyplot as plt

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

	#pad coordinates to make N x d+1 dimensionality (2,000 x 4)
	padded_coordinates = np.insert(coordinates, 0, 1, axis=1)
	# print padded_coordinates, padded_coordinates.shape

	#randomly choose weights for each coordinate having dx1 (4,)dimentionality
	weights = np.random.uniform(low=0.0, high=0.4, size=(len(padded_coordinates[0])))
	#print weights, weights.shape
	wcalc =   perceptronLearning(padded_coordinates, weights, labels)
	print wcalc
	# plot_costs(padded_coordinates[:,1], padded_coordinates[:,2], labels)
	# plot_weight_line(wcalc, 0.0, 0.9)
	# plt.title('Results for Perceptron Learning')
	# plt.show()
	#print labels, labels.shape
	#find final weights using perceptron algorithm
	#np.set_printoptions(threshold='nan')
	#print 'Calculating final weights using perceptron algorithm...'
	#print 'Calculated weights', perceptron(padded_coordinates, weights, labels)
	#perceptron(padded_coordinates, weights, labels)
def perceptronLearning(x, w, y):
	alpha = 0.1
	constBroken = True
	while constBroken:
		constBroken = False
		for i in range(len(x)):
			y_calc = np.dot(w, x[i])
			if(y_calc <= 0 and y[i] == 1):
				constBroken = True
				w = np.add(w, alpha*x[i])
			elif(y_calc >= 0 and y[i] == -1):
				constBroken = True
				w = np.subtract(w, alpha*x[i])
	return w
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
def plot_costs(x0, x1, y):
    for i, c in enumerate(y):  
        plt.scatter(x0[i], x1[i], marker='+' if c==1 else '$-$',
                    c='b', s=50)
def plot_weight_line(weights, x0, x1):
    def eq(w, x):
        """ convert w0 + w1*x + w2*y into y = mx + b"""
        return (-w[1]*x - w[0]) / w[2]
    plt.plot([x0, x1], [eq(weights, x0), eq(weights, x1)], ls='--', color='g')

if __name__ == "__main__":
    main()