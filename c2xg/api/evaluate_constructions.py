#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

#-- High-level function for evaluating construction grammar
#-- Assumes output from the c2xg learning algorithms
#-- Assumes RDRPOS model, word2vec dictionary

def evaluate_constructions(Parameters, eval_type = "", Grammar = "", run_parameter = 0):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
		
		print("")
		print("Starting C2xG.Evaluate_Constructions")
		print("")
		
		import datetime
		import time
		import sys
		import multiprocessing as mp
		import pandas as pd
		from functools import partial
		import cytoolz as ct
		import math

		#Import required script-specific modules#
		import c2xg
		from process_input.annotate_files import annotate_files
		from candidate_extraction.read_candidates import read_candidates
		from candidate_extraction.write_candidates import write_candidates
		from process_input.merge_conll import merge_conll
		from process_input.pandas_open import pandas_open
		from candidate_selection.get_coverage import get_coverage
		from process_input.pandas_open import pandas_open
		from constituent_reduction.expand_sentences import expand_sentences
		from process_input.annotate_files import annotate_files
		from candidate_extraction.create_shifted_df import create_shifted_df
		from candidate_extraction.get_query import get_query
		from candidate_selection.create_shifted_length_df import create_shifted_length_df
		from feature_extraction.get_query_autonomous_zero import get_query_autonomous_zero
		from candidate_selection.construction_cost import construction_cost
		from candidate_selection.grammar_evaluator import grammar_evaluator
		from candidate_selection.grammar_evaluator_baseline import grammar_evaluator_baseline
	
		#Load Grammar object if necessary#
		if eval_type == "Idiom":
			
			try:
				Grammar = read_candidates(Parameters.Data_File_Idioms)
				print("Loaded Idiom Grammar")
				expand_flag = False
				full_cxg = False
				construction_list = []

				for thing in Grammar.Idiom_List:
					idiom_string = thing[0]
					idiom_list = idiom_string.split(" ")
					
					current_construction = []
					
					for unit in idiom_list:
						try:
							current_index = Grammar.Lemma_List.index(unit)
							current_tuple = ("Lex", current_index)
							current_construction.append(current_tuple)
						except:
							print("Skipping " + str(idiom_list))
					
					if current_construction != [[]]:
						construction_list.append(current_construction)
					
				Grammar.Idiom_List = []
			
			except:
				print("Unable to load grammar specified in parameters")
				sys.kill()
				
		if eval_type == "Constituent":
			
			try:
				Grammar = read_candidates(Parameters.Data_File_Constituents)
				print("Loaded Constituent Grammar")
				expand_flag = False
				full_cxg = False
				
				temp_list = []
				
				for direction in Grammar.Constituent_Dict:
					for head in direction:
						for sequence in direction[head]:
							temp_list.append(sequence)
				
				construction_list = []
				for sequence in temp_list:
					current_sequence = []
					for unit in sequence:
						current_tuple = ("Pos", unit)
						current_sequence.append(current_tuple)
					construction_list.append(current_sequence)
				
			
			except:
				print("Unable to load grammar specified in parameters")
				sys.kill()
				
		if eval_type == "Construction":
			
			try:
				Grammar = read_candidates(Parameters.Data_File_Constructions)
				print("Loaded Construction Grammar")
				expand_flag = True
				full_cxg = True
				construction_list = Grammar.Construction_List
			
			except:
				print("Unable to load grammar specified in parameters")
				sys.kill()
		
		start_beginning = time.time()
	
		#1: Annotate plain text input files  ---------------------------------------------------------#
		if Parameters.Run_Tagger == True:
		
			conll_files = []
			
			for input_file in Parameters.Input_Files:
		
				conll_files += annotate_files(input_file, Parameters, Grammar)
			
			Parameters.Input_Files = conll_files
			input_files = conll_files
			
			#Only need to run tagger once#
			Parameters.Run_Tagger = False
			
		#Get input files if tagger not run#
		else:
			input_files = Parameters.Input_Files
			
		#2: Get coverage info -------------------------------------------------------------------------#
		#-----Combine training_testing_files into one file for each restart and combine and testing_files into one file --#
		#-------This makes testing and restarts easier to handle ---------------------------------------------------------#
		
		print("\t\tJoining training-testing and testing files into single file for each restart with a single file for testing.")
		testing_filename = str(Parameters.Temp_Folder + "/" + Parameters.Nickname + "Testing.conll")
		tuple_list = (input_files, testing_filename)
		merge_conll(tuple_list, Parameters.Encoding_Type)


		#----FINISHED CONSOLIDATING TESTING SETS -----------------------------------------------------------------#
		
		#First, evaluate string candidates to lists and sort by length#
		eval_list = ct.groupby(len, construction_list)
		
		try:
			del eval_list[0]
		except:
			print("No zero key errors")
		
		try:
			del eval_list[1]
		except:
			print("No one key errors")

		input_file = testing_filename
		
		print("")
		print("\t\tBeginning MDL-prep for " + str(input_file))
		print("")
			
		input_file_original = input_file
			
		#First, load and expand input file#
		input_df = pandas_open(input_file, 
								Parameters, 
								Grammar,
								write_output = False,
								delete_temp = False
								)
		
		#Save DF for later calculations#										
		total_words = len(input_df)
		encoding_df = input_df										
		
		if expand_flag == True:
			input_df = expand_sentences(input_df, Grammar, write_output = False)
					
		else:
			input_df.loc[:,"Alt"] = 0
			input_df = input_df.loc[:,['Sent', 'Alt', 'Mas', "Lex", 'Pos', 'Cat']]

		#Second, call search function by length#
		result_list = []

		if input_df.empty:
			print("ERROR: Training DataFrame is empty.")
			sys.kill()
	
		for i in eval_list.keys():
					
			current_length = i
			current_list = eval_list[i]
	
			if current_list:
								
				current_df = input_df.copy("Deep")
								
				print("")
				print("\t\t\tStarting constructions of length " + str(i) + ": " + str(len(current_list)))
								
				if current_length > 1:
							
					#Create shifted alt-only dataframe for length of template#
					alt_columns = []
					alt_columns_names = []
					for i in range(current_length):
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
					for i in range(current_length):
						sent_columns.append(0)
						sent_columns_names.append("c" + str(i))
							
					sent_dataframe = create_shifted_df(current_df, 0, sent_columns)
					sent_dataframe.columns = sent_columns_names
					query_string = get_query(sent_columns_names)
					row_mask_sent = sent_dataframe.eval(query_string)
					del sent_dataframe
								
					#Create and shift template-specific dataframe#
					current_df = create_shifted_length_df(current_df, current_length)

					current_df = current_df.loc[row_mask_sent & row_mask_alt,]
					del row_mask_sent
					del row_mask_alt
							
					#Remove NaNS and change dtypes#
					current_df.fillna(value=0, inplace=True)
					column_list = current_df.columns.values.tolist()
					current_df = current_df[column_list].astype(int)
							
				elif current_length == 1:
							
					query_string = "(Alt == 0)"
					current_df = current_df.query(query_string, parser='pandas', engine='numexpr')
					current_df = current_df.loc[:,['Sent', "Lex", 'Pos', 'Cat']]
					current_df.columns = ['Sent', 'Lem0', 'Pos0', 'Cat0']
						
				#Remove zero valued indexes#
				column_list = current_df.columns.values.tolist()
				query_string = get_query_autonomous_zero(column_list)
				current_df = current_df.query(query_string, parser='pandas', engine='numexpr')

				#Now, search for individual sequences within prepared DataFrame#
				#Start multi-processing#
				pool_instance=mp.Pool(processes = Parameters.CPUs_General, maxtasksperchild = None)
				coverage_list = pool_instance.map(partial(get_coverage, 
																current_df = current_df, 
																lemma_list = Grammar.Lemma_List, 
																pos_list = Grammar.POS_List, 
																category_list = Grammar.Category_List,
																total_words = total_words
																), [x for x in current_list], chunksize = 500)
				pool_instance.close()
				pool_instance.join()
				#End multi-processing#

				coverage_list = ct.merge([x for x in coverage_list])
				result_list.append(coverage_list)
	
				del current_df
				
		#Merge and save coverage dictionaries for each training set#
		result_dict = ct.merge([x for x in result_list])

		del result_list

		result_dict_encoded = {}
		result_dict_indexes = {}

		if len(result_dict.keys()) == 0:
			
			print("ERROR: No candidate coverage results for this datset.")
			
		else:
			for key in result_dict:

				result_dict_encoded[key] = result_dict[key]["Encoded"]
				result_dict_indexes[key] = result_dict[key]["Indexes"]
				
			del result_dict

			result_df_encoded = pd.DataFrame.from_dict(result_dict_encoded, orient = "index")
			result_df_encoded.columns = ["Encoded"]
				
			result_df_indexes = pd.DataFrame.from_dict(result_dict_indexes, orient = "index", dtype = "object")
			result_df_indexes.columns = ["Indexes"]
				
			result_df = pd.merge(result_df_encoded, result_df_indexes, left_index = True, right_index = True)
			index_list = [str(list(x)) for x in result_df.index.tolist()]
			result_df.loc[:,"Candidate"] = index_list

			del result_df_encoded
			del result_df_indexes
				
	#NOW GET EVALUATION SCORES#
	#PRE-CALCULATE AS MUCH OF MDL METRIC AS POSSIBLE#
	TOP_LEVEL_ENCODING = 0.301
	
	#Get list of actual atomic units used#
	pos_used_list = []
	lex_used_list = []
	cat_used_list = []
	
	for construction in construction_list:

		for unit in construction:
			if unit[0] == "Lex":
				lex_used_list.append(unit[1])
			
			elif unit[0] == "Pos":
				pos_used_list.append(unit[1])
			
			elif unit[0] == "Cat":
				cat_used_list.append(unit[1])
				
	lex_used_list = list(set(lex_used_list))
	pos_used_list = list(set(pos_used_list))
	cat_used_list = list(set(cat_used_list))
	
	#Calculate base encoding costs#
	pos_units = len(pos_used_list) 
	lex_units = len(lex_used_list) 
	cat_units = len(cat_used_list) 
	
	print("POS: " + str(pos_units) + " / " + str(len(Grammar.POS_List)))
	print("Cat: " + str(cat_units) + " / " + str(len(Grammar.Category_List)))
	print("Lex: " + str(lex_units) + " / " + str(len(Grammar.Lemma_List)))

	if pos_units > 1:
		pos_unit_cost = -(math.log(1/float(pos_units))) + TOP_LEVEL_ENCODING
		
	else:
		pos_unit_cost = 1
	
	if lex_units > 1:
		lex_unit_cost = -(math.log(1/float(lex_units))) + TOP_LEVEL_ENCODING
	
	else:
		lex_unit_cost = 1
	
	if cat_units > 1:
		cat_unit_cost = -(math.log(1/float(cat_units))) + TOP_LEVEL_ENCODING
	
	else:
		cat_unit_cost = 1
	
	#For full CxGs, there are three representation types to distinguish#
	if full_cxg == True:
		SLOT_R_COST = 0.4771
	
	#For other grammars, there is only one representation type without cost#
	else:
		SLOT_R_COST = 0.0000
		
	result_df = construction_cost(result_df, SLOT_R_COST, pos_unit_cost, lex_unit_cost, cat_unit_cost)
	
	encoding_df.loc[:,"Alt"] = 0
	
	#Calculate unencoded baseline#
	num_units = encoding_df.loc[:,"Mas"].max()
	all_indexes = set(range(0,num_units+1))
	baseline_mdl = grammar_evaluator_baseline(encoding_df)
	
	#Get full MDL for comparison against unencoded baseline#
	mdl_l1, mdl_l2, mdl_full = grammar_evaluator(result_df, all_indexes)
	
	
	total_unencoded_size = grammar_evaluator_baseline(encoding_df)
	total_over_baseline = 1 - (mdl_full / (float(total_unencoded_size)))
	
	print("")
	print("\t\tTest MDL (Full): " + str(mdl_full) + "; Unencoded MDL: " + str(total_unencoded_size) + "; Adjusted MDL: " + str(total_over_baseline))
	
	weighted_mdl = total_over_baseline
	
	print("\t\tFinal stability weighted metric: " + str(weighted_mdl))
		
		
	return
#-----------------------------------------------------------------------------------------------#