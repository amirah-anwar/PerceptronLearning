# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import numpy as np
from sklearn.linear_model import perceptron
 
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

# Create the model
net = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
net.fit(coordinates,labels)
 
# Print the results
print "Prediction " + str(net.predict(coordinates))
print "Actual     " + str(labels)
print "Accuracy   " + str(net.score(coordinates, labels)*100) + "%"

# Output the values
print "Coefficients " + str(net.coef_)
print "Intercept " + str(net.intercept_)
 
