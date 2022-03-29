import sys
import numpy
from sklearn import preprocessing

#http://www.scipy.org/
try:
	from numpy import dot, array, mean
	from numpy.linalg import norm

except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))


def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	vector1 = numpy.array(vector1)
	vector2 = numpy.array(vector2)
	return numpy.dot(vector1,vector2) / (numpy.linalg.norm(vector1) * numpy.linalg.norm(vector2))


def euclidean(vector1, vector2):
	# query vector / document vector
	vector1 = array(vector1)
	vector2 = array(vector2)
	# X = numpy.asarray([vector1, vector2], dtype=numpy.float)

	# X_normalized = preprocessing.normalize(X, norm='l2')
	# vector1 = X_normalized[0]
	# vector2 = X_normalized[1]
	eu = numpy.linalg.norm(vector1-vector2)
	return eu
