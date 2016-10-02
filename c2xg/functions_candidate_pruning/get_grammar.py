#------------------------------------------------------------#
def get_grammar(full_vector_df, query_string):

	from functions_candidate_pruning.grammar_evaluator import grammar_evaluator
	import pandas as pd

	#Limitation on number of conditions for query; ensure safety#
	double_flag = 0
	items = query_string.count(",")
	
	if items > 0:
		query_list = query_string.split(",")
		query_string = query_list[0]
		query_string2 = query_list[1]
		query_string2 = query_string2.replace(" | ","", 1).replace(",","")
		double_flag = 1	
	#End on safety check for overly long queries#
	
	training_list = [x for x in full_vector_df.columns.tolist() if x[0:8] == "Coverage"]
	
	if double_flag == 0:
		temp_df = full_vector_df.query(query_string, parser = "pandas", engine = "numexpr")
		
	elif double_flag == 1:
		temp_df = full_vector_df.query(query_string, parser = "pandas", engine = "numexpr")

		if len(temp_df) > 1 and "(" in query_string2:
			temp_df = temp_df.query(query_string2, parser = "pandas", engine = "numexpr")

	if len(temp_df) > 1:
			
		temp_size = len(temp_df)

		#Get coverage metric of current threshold on each training set#
		metric_list = []
		for coverage_column in training_list:
				
			temp_coverage = temp_df.loc[:,coverage_column].sum()
			temp_metric = grammar_evaluator(temp_size, temp_coverage)
			metric_list.append(temp_metric)
				
		return metric_list
#-------------------------------------------------------------------#