#--------------------------------------------------------------#
#--Take single feature, learn optimum threshold ---------------#
#--------------------------------------------------------------#
def learn_thresholds(feature_name, full_vector_df):

	from functions_candidate_pruning.get_grammar import get_grammar
	
	import cytoolz as ct
	
	#Set upper and lower bounds on thresholds, and find increment#
	min_value = full_vector_df.loc[:,feature_name].min(skipna = True)
	max_value = full_vector_df.loc[:,feature_name].max(skipna = True)
	increment = (max_value - min_value) / float(1000)
	
	#Initialize for search#
	current_threshold = max_value
	threshold_scores = {}
	deviation_list = []
	
	#Until threshold falls below lower bound, evaluate possible thresholds#
	for i in range(1000):
	
		#Get DataFrame containing coverage for current threshold#
		query_string = "(" + feature_name + " > " + str(current_threshold) + ")"
		metric_list = get_grammar(full_vector_df, query_string)
					
		#Analyze cross-validated coverage after loop through training sets#
		threshold_scores[current_threshold] = sum(metric_list) / float(len(metric_list))
		temp_range = max(metric_list) - min(metric_list)
		deviation_list.append(temp_range)
				
		#Update threshold#
		current_threshold = current_threshold - increment

	#Now, find smallest threshold in dictionary#
	if len(threshold_scores.keys()) > 2:
		best_threshold = min(threshold_scores, key = threshold_scores.get)
		query_string = "(" + feature_name + " > " + str(best_threshold) + ")"
		temp_df = full_vector_df.query(query_string, parser = "pandas", engine = "numexpr")
		temp_size = len(temp_df)
		
		avg_range = sum(deviation_list) / float(len(deviation_list))
		print("\tDone with " + feature_name  + ", Threshold: " + str(best_threshold) + ", Candidates: " + str(temp_size) + " with avg. metric range of " + str(avg_range) + " across training sets")
	
	else:
		print("Empty: No optimum threshold")
		best_threshold = max_value
		
	return	{feature_name: best_threshold}
#-------------------------------------------------------------#