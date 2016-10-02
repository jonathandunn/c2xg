#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_type_examples(data_file,
						output_file,
						rule_dict,
						constituent_len_dictionary,
						length_list,
						pos_list,
						encoding_type,
						semantic_category_dictionary,
						word_list,
						lemma_list,
						lemma_dictionary,
						pos_dictionary,
						category_dictionary,
						delete_temp
						):

	import pandas as pd
	import cytoolz as ct
	import time
		
	from functions_input.pandas_open import pandas_open
	from functions_constituent_reduction.find_unit_index import find_unit_index
	from functions_constituent_reduction.get_search_df_expansion import get_search_df_expansion
	from functions_constituent_reduction.get_expansion_query import get_expansion_query
	from functions_constituent_reduction.constituents_reduce import constituents_reduce
		
	time_start = time.time()
	
	#Loop through files to support out-of-memory datasets#
	current_df = pandas_open(data_file, 
								encoding_type,
								semantic_category_dictionary,
								word_list,
								lemma_list,
								pos_list,
								lemma_dictionary,
								pos_dictionary,
								category_dictionary,
								save_words = True,
								write_output = False,
								delete_temp = delete_temp
								)
	print("")
	
	import codecs
	
	fw = codecs.open(output_file, "w", encoding = "utf-8")
	
	print("\tFinding constituent matches.")	
	#Loop through constituents by length, creating only 1 search_df for each length#
	for length in length_list:
	
		print("Starting length: " + str(length) + " with " + str(len(constituent_len_dictionary[length])) + " sequences.")
		
		#Generate initial search DF#
		copy_df = current_df.copy("Deep")
		search_df = get_search_df_expansion(copy_df, length)
		
		#Loop through constituents of current length#
		for constituent in constituent_len_dictionary[length]:
		
			direction = rule_dict[constituent]

			if direction == "L":
				current_head = constituent[0]
			
			elif direction == "R":
				current_head = constituent[-1]
			
			try:
				head_write = pos_list[current_head]
			except:
				head_write = "na"
			
			constituent_write = ""
			
			for thing in constituent:
				try:
					constituent_write += str(pos_list[thing]) + " "
				except:
					constituent_write += "na "

			#Find constituents#
			query_string = get_expansion_query(constituent)
			match_df = search_df.query(query_string, parser='pandas', engine='numexpr')
			
			mas_values = match_df.loc[:,"Mas"].values
			
			for start_mas in mas_values:
				
				fw.write(head_write)
				fw.write(",")
				fw.write(constituent_write)
				fw.write(",")
				
				fw.write(str(current_df.loc[current_df['Mas'] == start_mas, "Word"].values))

				sent_id = current_df.loc[current_df['Mas'] == start_mas, "Sent"].values
				sent_id = sent_id[0]
				
				for k in range(1, length):
					new_mas = start_mas + k
					
					fw.write(" ")
					fw.write(str(current_df.loc[current_df['Mas'] == new_mas, "Word"].values))
				
				fw.write(",")
				
				query_string = "(Sent == " + str(sent_id) + ")"
				sent_df = current_df.query(query_string, parser='pandas', engine='numexpr')
				
				for row in sent_df.itertuples(index = False, name = None):
					fw.write(str(row[2]))
					fw.write(" ")
					
				fw.write("\n")

			
		#Done looping through constituents of current length#
	#Done looping through constituents by length#			
	
	print("Time for finding examples by type: " + str(time.time() - time_start))
	
	return 
#---------------------------------------------------------------------------------------------#