#--------------------------------------------------------------#
#--Processing function to search through potential grammars, --#
#-- evaluate each, and return optimum grammar found------------#
#--------------------------------------------------------------#
def learn_feature_set(feature_name,
						full_vector_df,
						threshold_dict
						):

	from functions_candidate_pruning.grammar_generator import grammar_generator
	from functions_candidate_pruning.grammar_evaluator import grammar_evaluator
	from functions_candidate_pruning.get_grammar import get_grammar
	
	import cytoolz as ct
	
	on_counter = 0
	off_counter = 0
	
	#Separate current feature from threshold dictionary#
	current_threshold = threshold_dict.pop(feature_name, None)
	current_query = " | (" + feature_name + " > " + str(current_threshold) + ")"
	
	#Generate X number of random feature combinations#
	#For each, compare with current feature on and off#
	#Return best state#
	for i in range(1000):
	
		current_grammar_off = grammar_generator(threshold_dict)
		current_grammar_on = current_grammar_off + current_query
		
		#Get grammar qualities with and without current feature#
		off_metric_list = get_grammar(full_vector_df, current_grammar_off)
		on_metric_list = get_grammar(full_vector_df, current_grammar_on)
		
		off_score = sum(off_metric_list) / float(len(off_metric_list))
		on_score = sum(on_metric_list) / float(len(on_metric_list))
		
		if off_score > on_score:
			off_counter += 1
			
		elif on_score > off_score:
			on_counter += 1
	
	#Now set state and return#
	if off_counter > on_counter:
		state = "Off"
		
	elif on_counter > off_counter:
		state = "On"
		
	elif on_counter == off_counter:
		state = "On"
			
	print("\tFinished with " + feature_name + ": " + str(state) + "; " + str(off_counter) + " off and " + str(on_counter) + " on.")

	return	{feature_name: state}
#-------------------------------------------------------------#