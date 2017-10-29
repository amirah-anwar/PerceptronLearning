# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
 
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

# Create the model
lm = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
lm.fit(coordinates,labels)
 
# Output the values
print "Coefficients " + str(lm.coef_)
print "Intercept " + str(lm.intercept_)
 
