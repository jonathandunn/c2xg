#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

#-- This function takes a model of constituents and constructions and returns a model with usage info

def learn_usage(Grammar, Parameters, run_parameter = 0):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
		
				print("")
		print("Starting C2xG.Get_Vectors")
		print("")
		
		#IMPORT DEPENDENCIES -------------------------------------------------------------------------#

		import time
		import sys
		import csv
		import cytoolz as ct
		import multiprocessing as mp
		from functools import partial
		import pandas as pd
		
		from get_vectors import process_get_vectors

		#Import required script-specific modules#
		from process_input.annotate_files import annotate_files
		from candidate_extraction.read_candidates import read_candidates
		
		start_all = time.time()
		
		#Load Grammar object if necessary#
		if Grammar == "Load":
			
			if TFIDF == True:
				try:
					Grammar = read_candidates(Parameters.Data_File_Usage)
					print("Loaded Idiom Grammar")
				except:
					print("Unable to load Idiom grammar specified in parameters")
					sys.kill()
					
			elif TFIDF == False:
				try:
					Grammar = read_candidates(Parameters.Data_File_Constructions)
					print("Loaded Idiom Grammar")
				except:
					print("Unable to load Idiom grammar specified in parameters")
					sys.kill()

		#Fix elements which throw off ARFF exporting#
		try:
			comma_index = Grammar.Lemma_List.index(",")
			Grammar.Lemma_List[comma_index] = "<COMMA>"

		except:
			comma_index = "n/a"
	
		try:
			quote_index = Grammar.Lemma_List.index('"')
			Grammar.Lemma_List[quote_index] = "<QUOTE>"

		except:
			quote_index = "n/a"
			
		#Evaluate string candidates to lists and sort by length#
		eval_list = []
	
		for construction in Grammar.Candidate_List:
			eval_list.append(eval(construction))

		eval_list = ct.groupby(len, eval_list)
		Grammar.Candidate_List = eval_list		

		#1: Annotate plain text input files  ---------------------------------------------------------#

		if Parameters.Run_Tagger == True:
		
			conll_files = []
				
			for input_file in Parameters.Input_Files:
			
				conll_files += annotate_files(input_file, Parameters, Grammar)
				
			Parameters.Input_Files = conll_files
			input_files = conll_files
				
		else:
		
			input_files = Parameters.Input_Files
		#--------------------------------------------------------------------------------------------#
		vector_file_list = []
		
		#Now, multi-process for input files#
		pool_instance=mp.Pool(processes = number_of_cpus_extract, maxtasksperchild = None)
		vector_file_list.append(partial(process_get_vectors, 
										Grammar = Grammar,
										Parameters = Parameters,
										metadata_dictionary = "",
										frequency_type = Parameters.Frequency,
										vector_type = "CxG+Units",
										write_output = False
										), input_files, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for input files#

		centroid_df_full = get_centroid(vector_file_list, delete_temp)
		
		#Now take and add to new grammar and usage model#
		Grammar.Centroid_DF = centroid_df_full
		write_candidates(Parameters.Data_File_Usage, Grammar)	
		
		end_all = time.time()
		print("All files completed in " + str(end_all - start_all) + " seconds.")
		
		return
#---------------------------------------------------------------------------------------------#