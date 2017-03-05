#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

#-- High-level function for learning a constituency grammar
#-- Assumes RDRPOS model and output from get_indexes

def learn_constituents(Parameters, Grammar = "", run_parameter = 0):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
		
		print("")
		print("Starting C2xG.Learn_Constituents")
		print("")
		
		import time
		import sys
		import multiprocessing as mp
		import pandas as pd
		from functools import partial
		import cytoolz as ct

		#Import required script-specific modules#
		import c2xg
		from process_input.annotate_files import annotate_files
		from candidate_extraction.read_candidates import read_candidates
		from candidate_extraction.write_candidates import write_candidates
		from api.get_indexes import get_indexes
		from api.get_candidates import process_get_candidates
		from api.get_association import get_association
		from api.get_mdl_grammar import get_mdl_grammar
		from process_input.fold_split import fold_split
		from candidate_selection.merge_and_save import merge_and_save

		#Load Grammar object if necessary#
		if Grammar == "Load":
			
			try:
				Grammar = read_candidates(Parameters.Data_File_Idioms)
				print("Loaded Idiom Grammar")
			except:
					print("Unable to load Idiom grammar specified in parameters")
					sys.kill()

			if Grammar.Type == "Unlearned":
				print("Error: Wrong grammar type: " + str(Grammar.Type))
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
			
		#Now, run get_indexes#
		Grammar = c2xg.Grammar()
		Grammar = get_indexes(Parameters, Grammar)
		
		fold_file_dict = fold_split(Parameters, input_files)
			
		#---ITERATE ACROSS FOLDS ----------------------------------------------------------------------------#
		fold_results = []
		
		for fold in fold_file_dict:
		
			training_files = fold_file_dict[fold]["Training_Candidates"]
			training_testing_files = fold_file_dict[fold]["Training_Search"]
			testing_files = fold_file_dict[fold]["Testing"]
			
			print("")
			print("Starting Constituent fold number " + str(fold))
			print("")
			
			#2: Get POS-Only sequence candidates ----------------------------------------------------------#
			
			#Start multi-processing for file processing#
			pool_instance=mp.Pool(processes = Parameters.CPUs_General, maxtasksperchild = None)
			pool_instance.map(partial(process_get_candidates, 
											Parameters = Parameters,
											Grammar = Grammar,
											expand_check = False,
											file_extension = ".Candidates.Constituents",
											annotation_types = ["Pos"],
											max_candidate_length = Parameters.Max_Candidate_Length_Constituents,
											frequency_threshold_perfile = Parameters.Freq_Threshold_Constituents_Perfile
											), training_files, chunksize = 1)
			pool_instance.close()
			pool_instance.join()
			
			initial_flag = False
			#End multi-processing for file processing#
			
			#3: Get association vectors for candidates ----------------------------------------------------#
			print("")
			print("Starting to create association vectors for candidates.")
			print("")
			
			association_df, candidate_list_formatted = get_association(Parameters, Grammar, training_files, ".Candidates.Constituents", Parameters.Freq_Threshold_Constituents, "Pass")
			
			#4: MDL grammar learning with association vectors----------------------------------------------#
			constituent_grammar, constituent_mdl = get_mdl_grammar(Parameters, 
																	Grammar,
																	training_testing_files,
																	testing_files,										
																	association_df, 
																	Parameters.Max_Candidate_Length_Constituents,
																	candidate_list_formatted, 
																	expand_check = False,
																	atomic_unit_types = ["Pos"]
																	)
				
			#5: Save learned grammar ---------------------------------------------------------------------#
			print("")
			print("")
			print("Saving fold results.")
			filename = Parameters.Temp_Folder + Parameters.Nickname + ".Constituent.Fold=" + str(fold) + ".p"
			fold_results.append(filename)
			write_candidates(filename, (constituent_grammar, constituent_mdl))
			
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
		print("Finished with cross-fold evaluation for Constituents")
		print("")
		
		Grammar = merge_and_save("Constituent", fold_results, Parameters, Grammar)
		
		print("")
		print("Finished with identifying constituent heads.")
		print("")
		
	return Grammar
#----------------------------------------------------------------------------------------------------#