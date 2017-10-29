# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
 
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

# Create the model
logreg = linear_model.LogisticRegression()
logreg.fit(coordinates,labels)
 
# Output the values
print "Coefficients " + str(logreg.coef_[0])
print "Intercept " + str(logreg.intercept_)
 
