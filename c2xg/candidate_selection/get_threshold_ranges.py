#--------------------------------------------------------------#
def get_threshold_ranges(feature_list, full_vector_df, number_thresholds, max_candidate_length):

	threshold_values = {}
	
	for feature in feature_list:
		
		if feature == "Directional_Categorical_Unweighted":
		
			threshold_list = [x for x in range(1, max_candidate_length)]
			threshold_values[feature] = threshold_list
		
		else:
		
			#Set upper and lower bounds on thresholds, and find increment#
			min_value = full_vector_df.loc[:,feature].min(skipna = True)
			max_value = full_vector_df.loc[:,feature].max(skipna = True)
			increment = (max_value - min_value) / float(number_thresholds)
			
			#Initialize for search#
			current_threshold = max_value - 0.0001
			
			threshold_list = [current_threshold]
			
			for i in range(number_thresholds):
				current_threshold = current_threshold - increment
				threshold_list.append(current_threshold)
				
			threshold_values[feature] = threshold_list
			
	return	threshold_values
#-------------------------------------------------------------#