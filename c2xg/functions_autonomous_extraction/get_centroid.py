#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_centroid(vector_file_list, delete_temp):
	
	import pandas as pd
	import numpy as np
	from functions_candidate_extraction.read_candidates import read_candidates
	pd.set_option('precision',12)
	
	#The centroid is the expected usage of a given feature across the whole dataset.
	#This is calculated as the inverse probability of a text containing the feature.
	
	vector_file_list = vector_file_list[0]
	centroid_list = []
	
	#First merge all intermediate centroid files#
	print("Merging centroids from individual input files.")
	for vector_file in vector_file_list:
		
		temp_vector = read_candidates(vector_file)
		centroid_list.append(temp_vector)
	
	#Now concat and sum to get total texts each feature occurs in#
	full_vector = pd.concat(centroid_list, axis = 1)
	summed_vector = full_vector.sum(axis = 1)
	
	#Get total instances, drop instances and length as no longer necessary#
	total_instances = summed_vector.loc["Instances"]
	summed_vector.drop('Instances', axis = 0, inplace = True)
	print("Total instances represented: " + str(total_instances))
	
	#Get probability of each feature occur in text#
	summed_vector = summed_vector.div(total_instances, level = None, fill_value = 0, axis = 0)

	#Transpose to allow row by row manipulations during feature extraction#
	centroid_df = pd.DataFrame(summed_vector)
	centroid_df = centroid_df.T

	#---Possibly delete temp centroids---#
	if delete_temp == True:
		import os
		
		for file in vector_file_list:
			os.remove(file)
	#------------------------------------#
	
	return  centroid_df
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
