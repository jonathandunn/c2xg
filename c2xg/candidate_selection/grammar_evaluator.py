#--------------------------------------------------------------#
def grammar_evaluator(grammar_df, all_indexes):

	import math 

	if len(grammar_df) > 2:
	
		TOP_LEVEL_ENCODING = 0.301

		#L1 is the encoding size of the grammar: only the cost of encoding the current construction grammar-------#
		#Regret (residual unencoded units) is included in L3 -----------------------------------------------------#
		
		#Sum cost of constructions in current grammar#
		mdl_l1 = grammar_df.loc[:,"Cost"].sum()
		
		#L2 contains, first, the cost of encoding all constructions#
		num_constructions_encoded = grammar_df.loc[:,"Encoded"].sum()
		num_constructions = len(grammar_df)
		
		cost_per_construction = -(math.log(1/float(num_constructions))) + TOP_LEVEL_ENCODING
		
		mdl_l2_constructions = cost_per_construction * num_constructions_encoded
		
		#Find indexes not encoded by a construction, from grammar_df#
		#Combine and find set of all tuples in "Indexes" #
		encoded_indexes = set([item for sublist in grammar_df.loc[:,"Indexes"].tolist() for item in sublist])
		unencoded_indexes = set(all_indexes) - set(encoded_indexes)
		
		#L2 contains, second, the regret of all unencoded indexes#
		number_unencoded = len(list(unencoded_indexes))
		unencoded_cost = -(math.log(1/float(number_unencoded))) + TOP_LEVEL_ENCODING
		
		mdl_l2_unencoded = number_unencoded * unencoded_cost
		
		#Sum final MDL metric#
		mdl_l2 = mdl_l2_constructions + mdl_l2_unencoded
		mdl_full = mdl_l1 + mdl_l2
		
	else:
	
		mdl_l1 = 100000000000000000
		mdl_l2 = 100000000000000000
		mdl_full = 100000000000000000000000
	
	return mdl_l1, mdl_l2, mdl_full
#-------------------------------------------------------------#