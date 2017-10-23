# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import random
import numpy as np
import sys
import matplotlib.pyplot as plt

# ==============Pocket Algorithm==================================
#Linear classification
#Extracts coordinates, labels from classification.txt
#Randomly chooses weights for each coordinate
#Finds the least number of broken constraints since this breaks the 
#"promise problem"
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

	#randomly choose weights for each coordinate having dx1 (4,)dimentionality
	weights = np.random.uniform(low=0.0, high=0.4, size=(len(padded_coordinates[0])))

	brokenConstraints = pocketAlgorithm(padded_coordinates, weights, labels)
	print 'Plotting...'
	plot_missclass(brokenConstraints, 7000)
	

# Implements the percetron learning algorithm from class
# As long as there is a constraint, either add or subtract alpha*x(i)
# Converge or stop the algorithm where the number of constriants == 0
# Return the weight vector
def pocketAlgorithm(x, weight, y):
	#Max number of iterations given by assignment
	iterations = 7000
	#Alpha should be "small" enough
	alpha = 0.01
	#The constraints broken count
	constraint = 1
	#The weight vector, currently all randomized 
	w = weight

	brokenConstraints = []
	#While there exist a constraint run the PLA
	while iterations > 0:
		constraint = 0
		#Randomly choose indexes from our data x
		for i in np.random.permutation(range(len(x))):
			y_calc = np.dot(w, x[i])
			#Check wheter w*x violates a constraint
			if(y_calc < 0 and y[i] == 1):
				constraint += 1
				w = np.add(w, alpha*x[i])
				
			elif(y_calc > 0 and y[i] == -1):
				constraint += 1
				w = np.subtract(w, alpha*x[i])
		brokenConstraints.append(constraint)
		iterations -= 1
	return brokenConstraints

#Plots Misclassified points vs 7000 Iterations
def plot_missclass(missMat, maxIter):
	#plt.figure(figsize=(15,2))
	plt.plot(missMat)
	# plt.xticks(np.arange(0, maxIter, 100))
	plt.xlim(0,maxIter)
	plt.xlabel('Iterations')
	plt.ylabel('Misclassifications')
	plt.title('Misclassified points vs Iterations of Pocket Algorithm')
	plt.show()
if __name__ == "__main__":
    main()