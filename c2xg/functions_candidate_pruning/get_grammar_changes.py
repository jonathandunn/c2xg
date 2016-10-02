#-----------------------------------------------#
#--Get list of possible new feature settings ---#
#-----------------------------------------------#
def get_grammar_changes(feature_name, grammar_dict, full_vector_df):

	current_state = grammar_dict[feature_name]["State"]
	current_threshold = grammar_dict[feature_name]["Threshold"]
	
	change_list = []
	
	#First, find increment to change feature#
	max_value = full_vector_df.loc[:,feature_name].max()
	min_value = full_vector_df.loc[:,feature_name].min()
	increment = (max_value - min_value) / float(1000)
		
	#For features that are currently on, get 50 up and 50 down#
	if current_state == "On":
	
		new_threshold = current_threshold
		
		#50 steps above current#
		for i in range(50):
			new_threshold += increment
			change_list.append((feature_name, new_threshold))
			
		new_threshold = current_threshold
		
		#50 steps below current#
		for i in range(50):
			new_threshold -= increment
			change_list.append((feature_name, new_threshold))
			
	#For features that are current off, get 50 from top and 50 from bottom#
	elif current_state == "Off":
	
		new_threshold = max_value
		
		#50 steps down from top#
		for i in range(50):
			new_threshold -= increment
			change_list.append((feature_name, new_threshold))
			
		new_threhsold = min_value
			
		#50 steps up from bottom#
		for i in range(50):
			new_threshold += increment
			change_list.append((feature_name, new_threshold))

	return change_list