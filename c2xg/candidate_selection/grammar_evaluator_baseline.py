#-----------------------------------------------------------#
def grammar_evaluator_baseline(test_df):

	import math
	import pandas as pd
	
	TOP_LEVEL_ENCODING = 0.301
	
	#Limit test file to only original units (no complex constituents)#
	test_df = test_df.loc[test_df.loc[:,"Alt"] == 0]
	
	#Calculate encoding size#
	number_of_units = len(test_df)
	unit_cost = -(math.log(1/number_of_units)) + TOP_LEVEL_ENCODING
	total_unencoded_size = unit_cost * number_of_units
	
	return total_unencoded_size
#----------------------------------------------------------#