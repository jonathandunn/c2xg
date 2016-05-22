#---------------------------------------------------------------------------------------------#
#INPUT: Formatted Line and its length --------------------------------------------------------#
#OUTPUT: List of reductions of head-first phrases --------------------------------------------#
#Take formatted line and return versions with head first phrases reduced ---------------------#
#---------------------------------------------------------------------------------------------#
def process_learned_constituents(single_df, 
									pos_list, 
									lemma_list, 
									phrase_constituent_dictionary, 
									direction,
									action,
									counter,
									encoding_type = "",
									examples_file = ""																		
									):

	from functions_constituent_reduction.find_unit_index import find_unit_index
	from functions_constituent_reduction.get_search_df_expansion import get_search_df_expansion
	from functions_constituent_reduction.get_expansion_query import get_expansion_query
	from functions_constituent_reduction.constituents_reduce import constituents_reduce
	import pandas as pd
	import cytoolz as ct
	
	remove_dictionary = {}
	constituent_list = []
	sentence_reductions_list = []
	dependence_dictionary = {}
	
	#If writing to file, make sure that file is empty here#
	if action == "PRINT":
		fw = open(examples_file, "w", encoding = encoding_type)
		fw.close()
	
	#Loop through each phrase head to initialize remove dictionary#
	#And create list of all constituents#
	for key in phrase_constituent_dictionary.keys():
		remove_dictionary[key] = {}
		constituent_list += phrase_constituent_dictionary[key]
	
	#Evaluate to tuples and sort by length#
	temp_list = []
	for item in constituent_list:
		temp_list.append(eval(item))
		
	constituent_list = temp_list
	del temp_list
	
	constituent_len_dictionary = ct.groupby(len, constituent_list)
	length_list = list(constituent_len_dictionary.keys())
	length_list = sorted(length_list, reverse=False)
	
	print("\tFinding constituent matches.")	
	#Loop through constituents by length, creating only 1 search_df for each length#
	for length in length_list:
	
		#Generate initial search DF#
		copy_df = single_df.copy("Deep")
		search_df = get_search_df_expansion(copy_df, length)
		
		#Loop through constituents of current length#
		for constituent in constituent_len_dictionary[length]:
		
			#Find constituent head unit#
			if direction == "LR":
				current_head = constituent[0]
			
			elif direction == "RL":
				current_head = constituent[-1]

			#Find constituents#
			query_string = get_expansion_query(constituent)
			match_df = search_df.query(query_string, parser='pandas', engine='numexpr')
			
			#Get index information for replacements#
			head_list = match_df.loc[:,'Mas'].values.tolist()
			current_length = length
			
			#Dictionary with head index (master) as key and constituent length as value#
			for i in range(len(head_list)):
				
				#First, add to dictionary specific to this head#
				try:
					if current_length > remove_dictionary[current_head][head_list[i]]:
						remove_dictionary[current_head][head_list[i]] = current_length
						
				except:
					remove_dictionary[current_head][head_list[i]] = current_length
					
		#Done looping through constituents of current length#
	#Done looping through constituents by length#
	
	print("\tReducing complex constituents.")
	#Now begin loop through phrase heads to produce reduced sentences#
	for head in list(remove_dictionary.keys()):
		
		counter += 1
		current_dictionary = remove_dictionary[head]	
	
		#Determine if head is independent#
		if (head,) in constituent_list:
			current_status = "Independent"
			
		else:
			current_status = "Dependent"
		
		dependence_dictionary[head] = current_status
		
		print("\t", end="")
		print(pos_list[head], end="")
		print(": " + str(current_status) + ": ", end="")
		
		#Now create reduced phrases for all constituents of current head#
		copy_df = single_df.copy("Deep")
		current_match_df = constituents_reduce(pos_list, 
												lemma_list, 
												direction, 
												current_dictionary, 
												copy_df, 
												head, 
												current_status, 
												action, 
												encoding_type, 
												examples_file
												)
		
		try:
			current_match_df.loc[:,'Alt'] = counter
			sentence_reductions_list.append(current_match_df)
		
		except:
			print("")
			print("No matches for " + str(head))
			print("")
			
	#End loop through phrase heads#				
	total_match_df = pd.concat(sentence_reductions_list)
	
	return total_match_df, remove_dictionary, dependence_dictionary, counter
#---------------------------------------------------------------------------------------------#