#---------------------------------------------------------------------------------------------#
#INPUT: Current template and DataFrame -------------------------------------------------------#
#OUTPUT: DataFrame with matches to template --------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def find_template_matches(current_df, 
							template, 
							frequency_threshold_constructions_perfile
							):
    
	import pandas as pd
	import time
	from functions_candidate_extraction.get_column_shift import get_column_shift
	from functions_candidate_extraction.get_column_list import get_column_list
	from functions_candidate_extraction.get_candidate_count import get_candidate_count
	from functions_candidate_extraction.memory import memory
	from functions_candidate_extraction.get_query import get_query
	from functions_candidate_extraction.get_query_zero import get_query_zero
	from functions_candidate_extraction.create_shifted_df import create_shifted_df
	from functions_candidate_extraction.create_shifted_template_df import create_shifted_template_df
	
	start_all = time.time()
	
	#Create shifted alt-only dataframe for length of template#
	alt_columns = []
	alt_columns_names = []
	for i in range(len(template)):
		alt_columns.append(1)
		alt_columns_names.append("c" + str(i))
	
	alt_dataframe = create_shifted_df(current_df, 1, alt_columns)
	alt_dataframe.columns = alt_columns_names
	query_string = get_query(alt_columns_names)
	row_mask_alt = alt_dataframe.eval(query_string)
	del alt_dataframe

	#Create shifted sent-only dataframe for length of template#
	sent_columns = []
	sent_columns_names = []
	for i in range(len(template)):
		sent_columns.append(0)
		sent_columns_names.append("c" + str(i))
	
	sent_dataframe = create_shifted_df(current_df, 0, sent_columns)
	sent_dataframe.columns = sent_columns_names
	query_string = get_query(sent_columns_names)
	row_mask_sent = sent_dataframe.eval(query_string)
	del sent_dataframe
	
	#Create and shift template-specific dataframe#
	temp_list = get_column_list(template)
	column_list = temp_list[0]
	column_names = temp_list[1]
	current_df = create_shifted_template_df(current_df, column_list)
	current_df.columns = column_names
	
	current_df = current_df.loc[row_mask_sent & row_mask_alt,]
	del row_mask_sent
	del row_mask_alt
	
	#Remove NaNS and change dtypes#
	current_df.fillna(value=0, inplace=True)
	column_list = current_df.columns.values.tolist()
	current_df = current_df[column_list].astype(int)

	#Remove zero valued indexes#
	column_list = current_df.columns.values.tolist()
	query_string = get_query_zero(column_list)
	current_df = current_df.query(query_string, parser='pandas', engine='numexpr')
	
	#Find duplicated rows within same sentence and remove those which are duplicated#
	column_list = current_df.columns.values.tolist()
	row_mask2 = current_df.duplicated(subset=column_list, keep="first")
	current_df = current_df.loc[~row_mask2,]
	del row_mask2
	
	#Find unique rows and remove them#
	current_df = current_df.drop('c1', 1)
	column_list = current_df.columns.values.tolist()
	row_mask1 = current_df.duplicated(subset=column_list, keep=False)
	
	current_df = current_df.loc[row_mask1,]
	del row_mask1
	
	#Sort DataFrame to get similar candidates from same sentences together#
	column_list = current_df.columns.values.tolist()
	current_df = current_df[column_list].astype(int)
	current_df = current_df.sort_values(by=column_list, axis=0, ascending=True, inplace=False, kind="mergesort")
	
	#Count remaining candidates and remove those below frequency threshold#
	candidate_count_dictionary = get_candidate_count(current_df, frequency_threshold_constructions_perfile)
	del current_df
	
	print("Number found: " + str(len(candidate_count_dictionary)) + ": ", end="")
	
	end_all = time.time()

	return candidate_count_dictionary
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#