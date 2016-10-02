#------------------------------------------------------------------------#
#-- Prune candidates with low coverage on training data -----------------#
#-- And merge into single dataframe -------------------------------------#
#------------------------------------------------------------------------#
def merge_coverage_association(coverage_dataframe, full_vector_dataframe, coverage_threshold, training_files):

	import pandas as pd
	import numpy as np
	
	print("")
	print("Starting state: ")
	print("\tNumber of candidates in coverage dictionary: " + str(len(coverage_dataframe)))
	print("\tNumber of candidates in full vector df: " + str(len(full_vector_dataframe)))
	print("")
	
	#Reformat candidate names in DataFrame to tuples#
	candidate_list = list(full_vector_dataframe.loc[:,"Candidate"].values)
	candidate_list = [tuple(eval(x)) for x in candidate_list]
	candidate_list = np.asarray(candidate_list)
	full_vector_dataframe.loc[:,"Candidate"] = candidate_list
	
	#Make DataFrame from coverage_dictionary and merge the two#
	coverage_dataframe["Candidate"] = coverage_dataframe.index
	full_vector_dataframe = pd.merge(full_vector_dataframe, coverage_dataframe, on='Candidate')
	
	#Clean coverage column names#
	counter = 0
	for filename in training_files:
		counter += 1
		full_vector_dataframe = full_vector_dataframe.rename(columns = {filename: "Coverage" + str(counter)})
		
	#Make query to include all training files in coverage threhsold: must pass at least one#
	query_string = ""
	first_flag = 1
	for i in range(1,len(training_files)+1):

		if first_flag == 0:
			query_string += " or "
		query_string += "(Coverage" + str(i) + " >= " + str(coverage_threshold) + ")"
		first_flag = 0

	full_vector_dataframe = full_vector_dataframe.query(query_string, parser='pandas', engine='numexpr')
	
	print("")
	print("Ending state: ")
	print("\tNumber of candidates in full vector df: " + str(len(full_vector_dataframe)))
	print("")
	
	del coverage_dataframe

	return full_vector_dataframe
#------------------------------------------------------------------------#