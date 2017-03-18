#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

#-- High-level function for learning a list of idioms
#-- Assumes output from get_indexes

def learn_idioms(Parameters, Grammar = "", run_parameter = 0):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
		
		print("")
		print("Starting C2xG.Learn_Idioms")
		print("")
		
		import time
		import sys
		import multiprocessing as mp
		import pandas as pd
		from functools import partial
		import cytoolz as ct

		#Import required script-specific modules#
		import c2xg
		from process_input.write_conll_raw import write_conll_raw
		from api.get_indexes import get_indexes
		from process_input.get_index_lists import get_index_lists
		from candidate_extraction.read_candidates import read_candidates
		from candidate_extraction.write_candidates import write_candidates
		from api.get_candidates import process_get_candidates
		from api.get_association import get_association
		from api.get_mdl_grammar import get_mdl_grammar
		from process_input.fold_split import fold_split
		from candidate_selection.merge_and_save import merge_and_save
		
		start_beginning = time.time()
		
		if Parameters.Input_Files == []:
			print("ERROR: No input files specified.")
			sys.kill()
		
		#1: Annotate plain text input files  ---------------------------------------------------------#
		#---Idiom converts raw text to CoNLL without tagging and gets temporary lexical indexes --------#
		if Parameters.Run_Tagger == True:

			input_files = [str(Parameters.Input_Folder + "/" + x) for x in Parameters.Input_Files]

			pool_instance=mp.Pool(processes = Parameters.CPUs_General, maxtasksperchild = None)
			conll_files = pool_instance.map(partial(write_conll_raw, 
											Parameters = Parameters,
											), input_files, chunksize = 1)
			pool_instance.close()
			pool_instance.join()
			
			conll_files = [item for sublist in conll_files for item in sublist]
			input_files = conll_files
		
		else:
			input_files = Parameters.Input_Files
			
		#Create temporary lexical indexes -------------------------------------------------------------#
		if Grammar == "":
			Grammar = c2xg.Grammar()
		
		Grammar = get_indexes(Parameters, Grammar, input_files = input_files, idiom_check = True)
		Grammar.Type = "Idiom_Indexes"
		
		fold_file_dict = fold_split(Parameters, input_files)
	#---ITERATE ACROSS FOLDS ----------------------------------------------------------------------------#
		fold_results = []
		
		for fold in fold_file_dict:
		
			print("")
			print("Starting Idiom fold number " + str(fold))
			print("")
			
			training_files = fold_file_dict[fold]["Training_Candidates"]
			training_testing_files = fold_file_dict[fold]["Training_Search"]
			testing_files = fold_file_dict[fold]["Testing"]
		
			#2: Get Word-Only sequence candidates ----------------------------------------------------------#
						
			#Start multi-processing for file processing#
			pool_instance=mp.Pool(processes = Parameters.CPUs_General, maxtasksperchild = None)
			pool_instance.map(partial(process_get_candidates, 
											Parameters = Parameters,
											Grammar = Grammar,
											expand_check = False,
											file_extension = ".Candidates.Idioms",
											annotation_types = ["Lex"],
											max_candidate_length = Parameters.Max_Candidate_Length_Idioms,
											frequency_threshold_perfile = Parameters.Freq_Threshold_Idioms_Perfile
											), training_files, chunksize = 1)
			pool_instance.close()
			pool_instance.join()
			
			initial_flag = False
			#End multi-processing for file processing#
			
			#3: Get association vectors for candidates ----------------------------------------------------#
			association_df, candidate_list_formatted = get_association(Parameters, Grammar, training_files, ".Candidates.Idioms", Parameters.Freq_Threshold_Idioms, "Pass")
			
			#4: MDL grammar learning with association vectors----------------------------------------------#
			Idiom_grammar, Idiom_mdl = get_mdl_grammar(Parameters, 
													Grammar, 
													training_testing_files,
													testing_files,
													association_df, 
													Parameters.Max_Candidate_Length_Idioms,
													candidate_list_formatted, 
													expand_check = False,
													atomic_unit_types = ["Lex"]
													)
			
			#5: Save learned grammar ---------------------------------------------------------------------#
			print("")
			print("")
			print("Saving fold results.")
			filename = Parameters.Temp_Folder + Parameters.Nickname + ".Idiom.Fold=" + str(fold) + ".p"
			fold_results.append(filename)
			write_candidates(filename, (Idiom_grammar, Idiom_mdl))
			
			#6: Clean up this fold's candidate and MDL temp files ----------------------------------------#
			if Parameters.Delete_Temp == True:
				
				print("\tDeleting temp files.")
				from process_input.check_data_files import check_data_files
				
				data_files = []
				for i in range(1,Parameters.Restarts+1):
					data_files += [Parameters.Temp_Folder + "/" + Parameters.Nickname + ".Restart." + str(i) + ".conll"]
					data_files += [Parameters.Temp_Folder + "/" + Parameters.Nickname + ".Restart." + str(i) + ".conll.MDL"]
					data_files += [Parameters.Temp_Folder + "/" + Parameters.Nickname + ".Restart." + str(i) + ".conll.Training"]
				
				data_files += [Parameters.Temp_Folder + "/" + Parameters.Nickname + ".Testing.conll"]
				data_files += [Parameters.Temp_Folder + "/" + Parameters.Nickname + ".Testing.conll.MDL"]
				data_files += [Parameters.Temp_Folder + "/" + Parameters.Nickname + ".Testing.conll.Training"]

				for file in data_files:
					check_data_files(file)			
			
			print("")
			end_beginning = time.time()
			print("\tTotal time for fold: " + str(end_beginning - start_beginning))
			
		print("")
		print("Finished with cross-fold evaluation for Idioms")
		print("")
		
		Grammar = merge_and_save("Idiom", fold_results, Parameters, Grammar, input_files)	
		
		return Grammar
#----------------------------------------------------------------------------------------------------#