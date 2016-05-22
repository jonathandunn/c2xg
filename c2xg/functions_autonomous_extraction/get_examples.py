#---------------------------------------------------------------------------------------------#
#INPUT: DataFrame with current construction matches and  lemma, pos, category index lists ----#
#OUTPUT: list readable examples to write -----------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_examples(search_df, 
					candidate, 
					lemma_list, 
					pos_list, 
					category_list, 
					current_length
					):
	
	import pandas as pd
	
	candidate_id = ""
	example_list = []
	
	for i in range(len(candidate)):
	
		current_col = candidate[i][0]
		current_index = candidate[i][1]
		
		if current_col == "Lem":
			index_list = lemma_list
		
		elif current_col == "Pos":
			index_list = pos_list
		
		elif current_col == "Cat":
			index_list = category_list
			
		current_unit = index_list[current_index]
		
		candidate_id += str(current_col) + ":" + str(current_unit) + " "
		
	example_list.append(candidate_id)
	
	#Start loop through rows#
	column_list = [2]
	previous = 2
	
	for i in range(1,current_length):
		column_list.append(previous+3)
		previous = previous + 3
		
	for row in search_df.itertuples():
	
		current_row = ""
		
		for column in column_list:
			
			if column == 2:
				current_row += str(lemma_list[row[column]])
			
			else:
				current_row += " " + str(lemma_list[row[column]])
			
		example_list.append(current_row)
		
	return example_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#