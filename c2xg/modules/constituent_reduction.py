#General modules
import time
import codecs
import pandas as pd
import cytoolz as ct
from operator import itemgetter

#INPUT: Name of debug file, single line, alternate sentence representations with debug info --#
#OUTPUT: Add current alternates to readable debug file ---------------------------------------#
#Take line, debug filename, and alternate representations and write readable list of changes--#

def write_reduction_list(data_files_expanded, 
							word_list, 
							encoding_type
							):

	with open(data_file_reductions, "a", encoding = encoding_type) as fa:
			
		for file in data_files_expanded:
		
			print("Writing readable reductions corpus for: " + str(file))
		
			store = pd.HDFStore(file)
			single_df = store['Table']
			store.close()
			
			reduced_df = single_df['Wor']
			del single_df
			
			#Begin loop through sentences#
			for Sent, Alt in reduced_df.groupby(level=0):
			
				#Begin loop through alternate versions of each sentence#
				for Name, Version in Alt.groupby(level=1):
					
					temp_df = Version.reset_index().drop(['Sent', 'Alt', 'Unit'], axis=1)
					
					for i in range(len(temp_df)):
						temp_index = (int(temp_df.iloc[[i]].values))
						fa.write(str(word_list[temp_index]))
						fa.write(" ")
							
					fa.write("\n")

	return 
#---------------------------------------------------------------------------------------------#
#INPUT: Single sentence dataframe ------------------------------------------------------------#
#OUTPUT: single sentence dataframe with punctuation removed ----------------------------------#
#Take sentence and return dataframe without punctuation, etc. --------------------------------#

def remove_punc(single_df, counter):

	try:
		single_df = single_df[single_df.Pos != 0]
		
		pd.options.mode.chained_assignment = None
		single_df.loc[:,'Alt'] = counter
		pd.options.mode.chained_assignment = "warn"
		
	except:
		single_df = single_df
				
	return single_df
#---------------------------------------------------------------------------------------------#

#INPUT: Line ---------------------------------------------------------------------------------#
#OUTPUT: Alternate sentences -----------------------------------------------------------------#
#Allows parrallel processing of alternate candidate generation -------------------------------#

def process_sentence_expansion(current_df, Grammar):

	single_df = current_df.copy("Deep")
	single_df.reset_index(inplace=True)	
	single_df = single_df.loc[:,['Sent', 'Mas', "Lex", 'Pos', 'Cat']]
	
	alt_list = []
	
	#Counter keeps track of constituent types for assigning "Alt" ids#
	#Alt == 0 is reserved for the original input#
	
	#Remove Head-First constituents#
	if len(Grammar.Constituent_Dict[0].keys()) > 0:

		#print("\tStarting Head-First Constituent Reduction")
		start = time.time()
		total_match_df_lr, remove_dictionary_lr, counter = process_learned_constituents(single_df, 
																						Grammar.POS_List, 
																						Grammar.Lemma_List, 
																						Grammar.Constituent_Dict[0], 
																						direction = "LR", 
																						action = "Reduce", 
																						counter = 1
																						)
		alt_list.append(total_match_df_lr)

		end = time.time()		
		print("\tDone with Head-First phrases: " + str(end - start) + ", Number of alts: " + str(counter))
	
	#Initialize empty remove_dictionary_lr if necessary#
	else:
		remove_dictionary_lr = {}
	
	#Remove Head-Last constituents#
	if len(Grammar.Constituent_Dict[0].keys()) > 0:

		#print("\tStarting Head-Last Constituent Reduction")
		start = time.time()
		total_match_df_rl, remove_dictionary_rl, counter = process_learned_constituents(single_df, 
																						Grammar.POS_List, 
																						Grammar.Lemma_List, 
																						Grammar.Constituent_Dict[1], 
																						direction = "RL", 
																						action = "Reduce", 
																						counter = counter
																						)
		alt_list.append(total_match_df_rl)

		end = time.time()		
		print("\tDone with Head-Last phrases: " + str(end - start) + ", Number of alts: " + str(counter))
	
	#Initialize empty remove_dictionary_rl if necessary#
	else:
		remove_dictionary_rl = {}
											
	#Fully schematic representation#
	# print("")
	# print("\tStarting Fully Schematic Representation")
	# start = time.time()
	# total_schematic_df, counter = process_schematic_representation(single_df, 
																	# Grammar.POS_List, 
																	# Grammar.Lemma_List, 
																	# remove_dictionary_lr, 
																	# remove_dictionary_rl, 
																	# counter
																	# )
	# alt_list.append(total_schematic_df)

	# end = time.time()
	#print("\tDone with Fully-Schematic Representation: " + str(end - start) + ", Number of alts: " + str(counter))
	
	#Call function to combine and reformat alternate sentence DataFrames#
	alternate_sentence_candidates = create_alternate_sentences(single_df, alt_list)

	return alternate_sentence_candidates
#---------------------------------------------------------------------------------------------#
#INPUT: Input DF, pos_list, and directional remove dictionaries ------------------------------#
#OUTPUT: DataFrame with fully schematic versions of sentences --------------------------------#
#Produce full schematic representation with all largest constituents reduced ----------------#

def process_schematic_representation(single_df, 
										pos_list, 
										lemma_list,
										full_remove_dictionary_lr, 
										full_remove_dictionary_rl,
										counter
										):

	counter += 1
	tuple_list = []
	
	full_remove_dictionary = ct.merge(full_remove_dictionary_lr, full_remove_dictionary_rl)
	
	lr_heads = list(full_remove_dictionary_lr.keys())
	rl_heads = list(full_remove_dictionary_rl.keys())
	
	#Get index, length tuples for each head#
	for head in list(full_remove_dictionary.keys()):
		
		#Add (Head, Index, Length) tuples to list for merging#
		for index, length in full_remove_dictionary[head].items():
		
			current_tuple = (head, index, length)
			tuple_list.append(current_tuple)
			
	#Done creating (Head, Index, Length) tuples#
	
	#Sort from longest to shortest constituents#
	sorted_tuples = sorted(tuple_list, key=itemgetter(2))
	
	covered_indexes = []
	largest_constituents = []
	
	#Create collection of largest constituents not contained or overlapping with larger constituents#
	for constituent in sorted_tuples:
	
		current_head = constituent[0]
		current_index = constituent[1]
		current_length = constituent[2]
		
		if current_length > 1:
			current_covered = [x for x in range(current_index, current_index + current_length)]
			
		else:
			current_covered = [current_index]
		
		check = any(True for x in current_covered if x in covered_indexes)
		
		if check == False:
			covered_indexes += current_covered
			largest_constituents.append(constituent)
			
	del covered_indexes

	#Create DataFrame representing this fully schematic version#
	copy_df = single_df.copy("Deep")
	remove_list = []
	
	for constituent in largest_constituents:
		
		current_head = constituent[0]
		current_index = constituent[1]
		current_length = constituent[2]
		
		#Get identity of head index#
		if current_head in lr_heads:
			current_head_index = current_index
			
		elif current_head in rl_heads:
			current_head_index = current_index + current_length - 1
		
		#Get list of indexes constituting current constituent#
		if current_length > 1:
			current_covered = [x for x in range(current_index, current_index + current_length)]
		else:
			current_covered = [current_index]

		current_covered.remove(current_head_index)
		remove_list += current_covered
		
		current_pos_index = current_head
		copy_df.loc[copy_df.Mas == current_head_index, 'Pos'] = current_pos_index
			
	#Done changing phrase head information#
	#Now remove non-head members of constituents#
	copy_df = copy_df[~copy_df.Mas.isin(remove_list)]
	
	copy_df.loc[:,'Alt'] = counter

	return copy_df, counter
#---------------------------------------------------------------------------------------------#
#INPUT: Formatted Line and its length --------------------------------------------------------#
#OUTPUT: List of reductions of head-first phrases --------------------------------------------#
#Take formatted line and return versions with head first phrases reduced ---------------------#

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
		temp_list.append(item)
		
	constituent_list = temp_list
	del temp_list
	
	constituent_len_dictionary = ct.groupby(len, constituent_list)
	length_list = list(constituent_len_dictionary.keys())
	length_list = sorted(length_list, reverse=False)
	
	#print("\tFinding constituent matches.")	
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
	
	#print("\tReducing complex constituents.")
	#Now begin loop through phrase heads to produce reduced sentences#
	for head in list(remove_dictionary.keys()):
		
		counter += 1
		current_dictionary = remove_dictionary[head]	
	
		#Now create reduced phrases for all constituents of current head#
		copy_df = single_df.copy("Deep")
		current_match_df = constituents_reduce(pos_list, 
												lemma_list, 
												direction, 
												current_dictionary, 
												copy_df, 
												head, 
												action, 
												encoding_type, 
												examples_file
												)
		
		try:
			current_match_df.loc[:,'Alt'] = counter
			sentence_reductions_list.append(current_match_df)
		
		except:
			#print("")
			print("No matches for " + str(head))
			#print("")
			
	#End loop through phrase heads#				
	try:
		total_match_df = pd.concat(sentence_reductions_list)
	except:
		total_match_df = pd.DataFrame([])
	
	return total_match_df, remove_dictionary, counter
#---------------------------------------------------------------------------------------------#
#INPUT: DataFrame and current ngram length ---------------------------------------------------#
#OUTPUT: DataFrame modified for ngram search -------------------------------------------------#
#Prepare DataFrame for pos ngram search in sentence expansion --------------------------------#

def get_search_df_expansion(original_df, length):

	ordered_columns = ['Sent', 'Pos']
	column_list = []
	
	#First, create initial one-unit dataframe#
	holder_df = original_df.loc[:,['Mas', 'Sent', 'Pos']]
	column_list.append(holder_df)
	
	del holder_df
	
	#Second, if additional units are required, add sequentially#
	if length > 1:
	
		for i in range(2,length):
			ordered_columns.append(['Sent', 'Pos'])
		
		for i in range(len(ordered_columns)):
			holder_df = original_df.loc[:,ordered_columns[i]]
			column_list.append(holder_df.shift(-i))
			del holder_df
		
	original_df = pd.concat(column_list, axis=1)
	del column_list
		
	column_names = original_df.columns.values.tolist()
	column_names_new = []
	sent_counter = 1
	pos_counter = 1
		
	for column in column_names:
		if column == 'Sent':
			column_string = 'Sent' + str(sent_counter)
			column_names_new.append(column_string)
			sent_counter += 1
				
		elif column == 'Pos':
			column_string = 'Pos' + str(pos_counter)
			column_names_new.append(column_string)
			pos_counter += 1
				
	column_names_new.insert(0, 'Mas')	
	original_df.columns = column_names_new
	
	return original_df
#---------------------------------------------------------------------------------------------#
#INPUT: Dictionary with keys head index and values length of constituent----------------------#
#OUTPUT: List for removal --------------------------------------------------------------------#
#Produce reduction info for head-last phrases ------------------------------------------------#

def get_head_last_list(remove_dictionary):

	#Create list of indexes to be removed#
	remove_list = []
	head_list = []
	
	start_list = list(remove_dictionary.keys())
	
	for i in range(len(start_list)):
	
		current_length = remove_dictionary[start_list[i]]

		for j in range(0, current_length-1):
			remove_list.append(start_list[i] + j)

		head_list.append(start_list[i] + current_length-1)
				
	return remove_list, head_list
#---------------------------------------------------------------------------------------------#
#INPUT: Dictionary with keys head index and values length of constituent----------------------#
#OUTPUT: List for removal --------------------------------------------------------------------#
#Produce reduction info for head-first phrases -----------------------------------------------#

def get_head_first_list(remove_dictionary):

	#Create list of indexes to be removed#
	remove_list = []
	head_list = list(remove_dictionary.keys())
			
	for i in range(len(head_list)):
	
		current_length = remove_dictionary[head_list[i]]
		
		for j in range(1,current_length):
			remove_list.append(head_list[i] + j)
				
	return remove_list, head_list
#---------------------------------------------------------------------------------------------#
#INPUT: Length of current n-gram window and direction flag, pos index to start with ----------#
#OUTPUT: String of query ---------------------------------------------------------------------#

def get_expansion_query(constituent):

	if len(constituent) > 1:
	
		query = "(Pos1 == " + str(constituent[0])
	
		for i in range(1,len(constituent)):
			query += " and Pos" + str(i+1) + " == " + str(constituent[i])
		
		query += ") and (Sent1 "
	
		for i in range(1, len(constituent)):
			query += "== Sent" + str(i)
		
		query += ")"
	
	elif len(constituent) == 1:
		query = "(Pos1 == " + str(constituent[0]) + ")"
	
	return query
#---------------------------------------------------------------------------------------------#
#INPUT: List of units to find, index to find them in -----------------------------------------#
#OUTPUT: List unit indexes -------------------------------------------------------------------#
#Take list of units and list of indexes and return list of indexes of requested units --------#

def find_unit_index(list_of_units, index_list):

	list_of_indexes = list_of_units
	
	for i in range(len(index_list)):
		if index_list[i] in list_of_units:
			location = list_of_units.index(index_list[i])
			list_of_indexes[location] = i
			
	return list_of_indexes
#---------------------------------------------------------------------------------------------#
#INPUT: Data files with list of sentence dictionaries ----------------------------------------#
#OUTPUT: Number of expanded sentences; list of dictionaries saved to file --------------------#
#Take sentences and generate simpler, non-recursive variations and save them to file ---------#

def expand_sentences(data_files, Grammar, write_output):

	if write_output == True:
		print("Start sentence expansion for " + str(data_files))
	
	#Open HDF5 data file#
	if write_output == True:
		data_file_expanded = data_files + ".Expanded"
		store = pd.HDFStore(data_files)
		current_df = store['Table']
		store.close()
		
	elif write_output == False:
		current_df = data_files
	
	temp_dataframe = process_sentence_expansion(current_df, Grammar)
	
	if write_output == True:
		
		print("Done expanding sentences for " + str(data_files) + ", Lines: " + str(len(temp_dataframe)))
		#Save expanded sentence representation for expanded datafile#
		temp_dataframe.to_hdf(data_file_expanded, "Table", format="table", complevel=9, complib="blosc")

		return
		
	elif write_output == False:
		
		return temp_dataframe
#---------------------------------------------------------------------------------------------#
#INPUT: Sentence and list of reductions ------------------------------------------------------#
#OUTPUT: List of reduced sentence dictionaries -----------------------------------------------#
#Take sentence and list of reductions and return reduced alternate sentences, with ids--------#


def create_alternate_sentences(single_df, alt_list):
	
	print("")
	print("Starting to reduce and reform alternate sentence DataFrame.")
	
	#Remove illegal POS values (e.g., punctuation)#
	print("\tRemoving illegal POS")
	alt_list.append(remove_punc(single_df, 0))
		
	temp_dataframe = pd.concat(alt_list)
	temp_dataframe = temp_dataframe.sort_values(by=['Sent', 'Alt', 'Mas'], axis=0, ascending=True, inplace=False, kind='mergesort')
	temp_dataframe = temp_dataframe.loc[:,['Sent', 'Alt', 'Mas', "Lex", 'Pos', 'Cat']]
	#Finished creating and formatting DataFrame#
	
	return temp_dataframe
#---------------------------------------------------------------------------------------------#
#INPUT: Direction, dictionary of longest constituents, DF, and current constituent -----------#
#OUTPUT: Combined DF of alts -----------------------------------------------------------------#
#Reduce constituents in current direction ----------------------------------------------------#

def constituents_reduce(pos_list, 
						lemma_list, 
						direction, 
						remove_dictionary, 
						copy_df, 
						key, 
						action = "REDUCE", 
						encoding_type = "", 
						examples_file = ""
						):

	if direction == "LR":
		remove_list, head_list = get_head_first_list(remove_dictionary)
		
	elif direction == "RL":
		remove_list, head_list = get_head_last_list(remove_dictionary)
	
	#Get list of all sentence involved#
	sentence_df = copy_df.loc[copy_df.Mas.isin(head_list), 'Sent']
	sentence_list = sentence_df.drop_duplicates().values.tolist()
	match_df = copy_df.loc[copy_df.Sent.isin(sentence_list)]
	
	#Remove non-head indexes#
	match_df = match_df[~match_df.Mas.isin(remove_list)]
						
	#Replace head indexes with current phrase type#
	#Replacement depends on head independence status#
	
	#Independent heads: labelled with head part-of-speech, lemma stays the same#
	#Dependent heads: labelled with pos_phrase, lemma changes to pos_phrase#
	
	current_pos_index = key
	match_df.loc[match_df.Mas.isin(head_list), 'Pos'] = current_pos_index
			
	if action == "PRINT":

		original_df = copy_df.loc[copy_df.Sent.isin(sentence_list)]
		constituents_print(pos_list[key], head_list, remove_list, lemma_list, original_df, match_df, direction, examples_file, encoding_type)
			
	print("\t\t" + str(pos_list[key]) + ": " + str(len(head_list)) + " matches.")
			
	return match_df
#---------------------------------------------------------------------------------------------#
#INPUT: Current direction, dictionary of longest constituents, DF, head, examples file -------#
#OUTPUT: List of reductions of head last phrases ---------------------------------------------#
#Take formatted line and return versions with head last phrases reduced ----------------------#

def constituents_print(pos_label, 
						head_list, 
						remove_list, 
						lemma_list, 
						original_df, 
						match_df, 
						direction, 
						examples_file, 
						encoding_type
						):

	print(examples_file)	
	with codecs.open(examples_file, "a", encoding = encoding_type) as fw:
		fw.write(str(pos_label).upper())
		fw.write(str("\n\n"))
		
		sentence_list = original_df.Sent.drop_duplicates().values.tolist()
		
		for sentence in sentence_list:
			original_sentence_df = original_df.query("Sent == @sentence", parser='pandas', engine='numexpr')
			reduced_sentence_df = match_df.query("Sent == @sentence", parser='pandas', engine='numexpr')
			
			fw.write(str("Current Sentence: " + str(sentence) + ":\n\t"))

			for row in original_sentence_df.itertuples():

				if row[5] in head_list:
				
					fw.write(str(row[3]))
					
				elif row[5] in remove_list:
					
					if direction == "LR":
						fw.write(str("_"))
						fw.write(str(row[3]))
					
					elif direction == "RL":
						fw.write(str(" "))
						fw.write(str(row[3]))
						fw.write(str("_"))					
					
				else:
					fw.write(str(" "))
					fw.write(str(row[3]))				
						
			fw.write(str("\n\t"))
			
			for row in reduced_sentence_df.itertuples():
				fw.write(str(" "))
				fw.write(str(row[3]))
				
						
			fw.write(str("\n"))

	return
#---------------------------------------------------------------------------------------------#