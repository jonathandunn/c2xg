#---------------------------------------------------------------------------------------------#
#INPUT: Dictionary of pairs and their co-occurrences, unit index and frequency lists ---------#
#OUTPUT: Dataframe with co-occurrence data (a, b, c) for each pair ---------------------------#
#---------------------------------------------------------------------------------------------#
def check_distance(current_association_dictionary,
				   previous_association_dictionary
					):
	
	from scipy import spatial
	import numpy as np
	
	pk = []
	qk = []
	
	for key in current_association_dictionary:
	
		pk.append(current_association_dictionary[key])
		
		try:
			qk.append(previous_association_dictionary[key])
		
		except:
			qk.append(0)
	
	pk = np.asarray(pk)
	qk = np.asarray(qk)
	
	distance_measure = spatial.distance.cosine(pk, qk)

	return distance_measure
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#