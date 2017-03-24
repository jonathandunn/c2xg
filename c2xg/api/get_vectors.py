#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

#-- High-level function for taking a C2xG grammar and producing a vector of frequencies

def process_get_vectors(input_file, 
							metadata_dictionary,
							Grammar,
							Parameters,
							frequency_type,
							vector_type,
							write_output = True,
							expand_check = True
							):
	
	import time 
	import csv
	import pandas as pd
	import numpy as np
	from process_input.pandas_open import pandas_open
	from constituent_reduction.expand_sentences import expand_sentences
	from feature_extraction.process_extraction import process_extraction
	from feature_extraction.get_coverage_column import get_coverage_column
	from feature_extraction.get_meta_data import get_meta_data
	from candidate_extraction.write_candidates import write_candidates
	
	start_beginning = time.time()

	#INGEST TEST FILE ----------------------------------------------------------------------------#

	print("")
	print("Ingesting input files.")
	input_dataframe = pandas_open(input_file, Parameters, Grammar, save_words = False, write_output = False, delete_temp = False)

	#EXPAND TEST FILE ----------------------------------------------------------------------------#
	
	if vector_type == "CxG" or vector_type == "CxG+Units" and expand_check == True:
		print("")
		print("Expanding sentences to reduce recursive structures.") 
		input_dataframe = expand_sentences(input_dataframe, Grammar, write_output = False)
			
	else:
	
		input_dataframe.loc[:, 'Alt'] = 0
		
	full_sentences = input_dataframe.loc[:,'Sent'].tolist()
	number_of_sentences = full_sentences[-1]	

	print("Finished expanding " + str(number_of_sentences) + " sentences.")
	print("")

	#EXTRACT FEATURES ----------------------------------------------------------------------------#

	start_extraction = time.time()
	print("Begin feature extraction.")
	
	full_vector_df = process_extraction(Grammar.Candidate_List, 
											Parameters.Max_Candidate_Length_Constructions, 
											input_dataframe, 
											Grammar.Lemma_List, 
											Grammar.POS_List, 
											Grammar.Category_List, 
											number_of_sentences,
											frequency_type,
											vector_type
											)
	
	end_extraction = time.time()
	print("Done with feature extraction: " + str(end_extraction - start_extraction))

	#Save vector#
	print("")
	
	if frequency_type == "TFIDF":
	
		from feature_extraction.get_centroid_normalization import get_centroid_normalization
		print("Normalizing by usage-based centroid.")
		full_vector_df = get_centroid_normalization(full_vector_df, Grammar.Centroid_DF)
	
	#Repopulate and add meta-data#
	if Parameters.Use_Metadata == True:
	
		print("Adding meta-data from input file.")
		#metadata_tuples = metadata_dictionary[input_file]
		#full_vector_df = get_meta_data(full_vector_df, metadata_tuples)
		
	#Extract column names and save separately#
	column_metadata = list(full_vector_df.columns)
	column_ids = [i for i in range(len(column_metadata))]
	full_vector_df.columns = column_ids
	
	end_beginning = time.time()
	print("File " + str(input_file) + ": " + str(end_beginning - start_beginning))
	
	#Save vectors to file and return#
	if write_output == True:
	
		from process_input.get_temp_filename import get_temp_filename
		print("Saving vectors to file.")
		
		suffix = vector_type + "." + frequency_type + ".Features"
	
		output_name = get_temp_filename(input_file, suffix)
		write_candidates(output_name + ".Columns", column_metadata) #Save column names#
		full_vector_df.to_hdf(output_name, "Table", format='fixed', complevel=9, complib="blosc")
		
		return output_name
		
	elif write_output == False:
		
		return full_vector_df
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_vectors(Parameters, Grammar = "", TFIDF = False, run_parameter = 0):

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
		
		from api.get_vectors import process_get_vectors

		#Import required script-specific modules#
		from process_input.annotate_files import annotate_files
		from candidate_extraction.read_candidates import read_candidates
		
		start_all = time.time()
		
		#Load Grammar object if necessary#
		if Grammar == "":
			
			if TFIDF == True:
				try:
					Grammar = read_candidates(Parameters.Data_File_Usage)
					print("Loaded Grammar with TF-IDF info")
				except:
					print("Unable to load Idiom grammar specified in parameters")
					sys.kill()
					
			elif TFIDF == False:
				try:
					Grammar = read_candidates(Parameters.Data_File_Constructions)
					print("Loaded Grammar without TF-IDF info")
				except:
					print("Unable to load grammar specified in parameters")
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
	
		for construction in Grammar.Construction_List:
			eval_list.append(construction)

		eval_list = ct.groupby(len, eval_list)
		Grammar.Candidate_List = eval_list		

		#1: Annotate plain text input files  ---------------------------------------------------------#

		if Parameters.Run_Tagger == True:
		
			if Parameters.Use_Metadata == False:
		
				conll_files = []
				
				for input_file in Parameters.Input_Files:
			
					conll_files += annotate_files(input_file, Parameters, Grammar)
				
				Parameters.Input_Files = conll_files
				input_files = conll_files
				
			elif Parameters.Use_Metadata == True:
			
				metadata_dictionary = {}
				updated_input = []
				
				for input_file in Parameters.Input_Files:
				
					input_short = input_file
					output_name = output_folder + "/" + input_short + ".Vectors.hdf5"
					
					#Different return if using meta-data and annotating or not using meta-data#
					if use_metadata == True:
						
						conll_file, metadata_tuples = annotate_files(input_file, Parameters, Grammar, metadata = True, same_size = True)

						updated_input += conll_file
						metadata_dictionary[str(conll_file[0])] = metadata_tuples
			
				input_files = updated_input
				
			#Only need to run tagger once#
			Parameters.Run_Tagger = False
		
		#Get input files if tagger not run#
		else:
		
			if Parameters.Use_Metadata == True:
				print("Must annotate files as part of feature extraction with meta-data.")
				sys.kill()
			
			else:
				input_files = Parameters.Input_Files
		#--------------------------------------------------------------------------------------------#
		
		#DELETE THIS LATER#
		metadata_dictionary = {}
		
		#Now, multi-process for input files#
		pool_instance=mp.Pool(processes = Parameters.CPUs_General, maxtasksperchild = None)
		pool_instance.map(partial(process_get_vectors, 
										Grammar = Grammar,
										Parameters = Parameters,
										metadata_dictionary = metadata_dictionary,
										frequency_type = Parameters.Frequency,
										vector_type = Parameters.Vectors,
										expand_check = Parameters.Expand_Check
										), input_files, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for input files#
	
		end_all = time.time()
		print("All files completed in " + str(end_all - start_all) + " seconds.")
		
		return
#------------------------------------------------------------------------------------------------------------------------#