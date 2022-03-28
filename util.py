import sys
import numpy as np

#http://www.scipy.org/
try:
	from numpy import dot, array
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
	return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))


def euclidean(vector1, vector2):
	vector1 = array(vector1)
	vector2 = array(vector2)
	return float(np.sqrt(sum(pow(vector1-vector2,2))))


