#-----C2xG, v 1.0 ----------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#---- Copyright, 2015-2016 Jonathan E. Dunn --------------------------------------------------#
#---------- www.jdunn.name -------------------------------------------------------------------#
#---------- jonathan.edwin.dunn@gmail.com ----------------------------------------------------#
#---------- Illinois Institute of Technology, Department of Computer Science -----------------#
#---------------------------------------------------------------------------------------------#
#This function takes a model of constituents and constructions and returns a model with usage-#
#---------------------------------------------------------------------------------------------#


#Define function for multi-processing-------------------------------------------------------#
def process_learn_usage(input_file, 
				output_folder, 
				candidate_list, 
				lemma_list, 
				pos_list, 
				word_list, 
				category_list, 
				semantic_category_dictionary, 
				sequence_list, 
				max_construction_length, 
				annotation_types, 
				pruned_vector_dataframe, 
				encoding_type, 
				phrase_constituent_list, 
				lemma_dictionary, 
				pos_dictionary, 
				category_dictionary,
				full_scope,
				delete_temp = False
				):
	
	import time 
	import csv
	import pandas as pd
	from functions_input.pandas_open import pandas_open
	from functions_constituent_reduction.expand_sentences import expand_sentences
	from functions_autonomous_extraction.process_extraction import process_extraction
	from functions_autonomous_extraction.get_coverage_column import get_coverage_column
	from functions_annotate.annotate_files import annotate_files
	from functions_candidate_extraction.write_candidates import write_candidates
	from functions_input.get_temp_filename import get_temp_filename
	
	start_beginning = time.time()
	
	#Get output name for centroid df#
	output_file = get_temp_filename(input_file, ".Centroid")
	
	#---------------------------------------------------------------------------------------------#
	#INGEST TEST FILE ----------------------------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#
	print("")
	print("Ingesting input files.")
	input_dataframe = pandas_open(input_file, 
									encoding_type, 
									semantic_category_dictionary, 
									word_list, 
									lemma_list, 
									pos_list, 
									lemma_dictionary, 
									pos_dictionary, 
									category_dictionary,
									delete_temp = delete_temp,
									write_output = False
									)

	#---------------------------------------------------------------------------------------------#
	#EXPAND TEST FILE ----------------------------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#

	if full_scope == True:
	
		print("")
		print("Expanding sentences to reduce recursive structures.") 
		input_dataframe = expand_sentences(input_dataframe, 
											lemma_list, 
											pos_list, 
											category_list, 
											encoding_type, 
											write_output = False, 
											phrase_constituent_list = phrase_constituent_list
											)
		
		print("Finished expanding sentences.")
		print("")
		
	else:
	
		input_dataframe.loc[:, 'Alt'] = 0
		
	full_sentences = input_dataframe.loc[:,'Sent'].tolist()
	number_of_sentences = len(set(full_sentences))
		
	#---------------------------------------------------------------------------------------------#
	#EXTRACT FEATURES ----------------------------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#

	start_extraction = time.time()
	print("Begin feature extraction.")
	vector_df = process_extraction(candidate_list, 
						max_construction_length, 
						input_dataframe, 
						lemma_list, 
						pos_list, 
						category_list, 
						number_of_sentences,
						full_scope = full_scope,
						relative_freq = False,
						use_centroid = True
						)
	
	end_extraction = time.time()
	print("Done with feature extraction: " + str(end_extraction - start_extraction))
	
	#Create centroid df#
	instances = len(vector_df)
	centroid_df = vector_df.sum(axis="rows")
	centroid_df["Instances"] = instances
	
	write_candidates(output_file, centroid_df)

	print("")
	print("Vectors saved, returning to main process.")

	#---------------------------------------------------------------------------------------------#
	#PRINT TIME ELAPSED --------------------------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#
	print("")
	end_beginning = time.time()
	print("File " + str(input_file) + ": " + str(end_beginning - start_beginning))
	
	return output_file
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def learn_usage(model_file, 
				input_files, 
				input_folder, 
				output_folder, 
				temp_folder,
				annotate_pos, 
				number_of_cpus_extract, 
				settings_dictionary,
				encoding_type,
				docs_per_file,
				data_file_usage,
				delete_temp,
				use_metadata,
				run_parameter = 0
				):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
		#---------------------------------------------------------------------------------------------#
		#IMPORT DEPENDENCIES -------------------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#

		import time
		import sys
		import csv
		import cytoolz as ct
		import multiprocessing as mp
		from functools import partial
		import pandas as pd
		
		from learn_usage import process_learn_usage

		#Import required script-specific modules#
		from functions_candidate_extraction.read_candidates import read_candidates
		from functions_candidate_extraction.write_candidates import write_candidates
		from functions_autonomous_extraction.get_centroid import get_centroid
		from functions_annotate.annotate_files import annotate_files

		#---------------------------------------------------------------------------------------------#
		#LOAD DATA FROM MODEL FILE -------------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#
	
		start_all = time.time()
	
		print("Loading model file.")
		write_dictionary = read_candidates(model_file)

		candidate_list = write_dictionary['candidate_list']
		lemma_list = write_dictionary['lemma_list']
		pos_list = write_dictionary['pos_list']
		word_list = write_dictionary['word_list']
		category_list = write_dictionary['category_list']
		semantic_category_dictionary = write_dictionary['semantic_category_dictionary']
		sequence_list = write_dictionary['sequence_list']
		max_construction_length = write_dictionary['max_construction_length']
		annotation_types = write_dictionary['annotation_types']
		pruned_vector_dataframe = write_dictionary['pruned_vector_dataframe']
		phrase_constituent_list = write_dictionary['phrase_constituent_list']
	
		lemma_dictionary = write_dictionary['lemma_dictionary']
		pos_dictionary = write_dictionary['pos_dictionary']
		category_dictionary = write_dictionary['category_dictionary']
		emoji_dictionary = write_dictionary['emoji_dictionary']
		

		#Fix elements which throw off ARFF importing#
		try:
			comma_index = lemma_list.index(",")
			lemma_list[comma_index] = "<COMMA>"

		except:
			comma_index = "n/a"
	
		try:
			quote_index = lemma_list.index('"')
			lemma_list[quote_index] = "<QUOTE>"

		except:
			quote_index = "n/a"

		del write_dictionary
		print("Finished loading model file.")
		
		#---------------------------------------------------------------------------------------------#
		#1: Annotate plain text input files  ---------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#
		if annotate_pos == True:
		
			conll_files = []
			
			for input_file in input_files:
		
				if use_metadata == True:
					current_files, metadata_tuples = annotate_files(input_folder, 
																	input_file, 
																	settings_dictionary, 
																	encoding_type, 
																	number_of_cpus_extract, 
																	emoji_dictionary, 
																	docs_per_file,
																	use_metadata
																	)
																	
					conll_files += current_files
			
				elif use_metadata == False:
					conll_files += annotate_files(input_folder, 
												input_file, 
												settings_dictionary, 
												encoding_type, 
												number_of_cpus_extract, 
												emoji_dictionary, 
												docs_per_file,
												use_metadata
												)			
			input_files = conll_files
		#----------------------------------------------------------------------------------------------#
		#FIRST, get centroid_df for  partial scope: only lexical items for baseline vectors#
		print("Creating lexical only centroid.")
		
		vector_file_list = []
		
		#Now, multi-process for input files#
		pool_instance=mp.Pool(processes = number_of_cpus_extract, maxtasksperchild = None)
		vector_file_list.append(pool_instance.map(partial(process_learn_usage, 
									output_folder=temp_folder, 
									candidate_list=candidate_list, 
									lemma_list=lemma_list, 
									pos_list=pos_list, 
									word_list=word_list, 
									category_list=category_list, 
									semantic_category_dictionary=semantic_category_dictionary, 
									sequence_list=sequence_list, 
									max_construction_length=max_construction_length, 
									annotation_types=annotation_types, 
									pruned_vector_dataframe=pruned_vector_dataframe, 
									encoding_type=encoding_type, 
									phrase_constituent_list=phrase_constituent_list, 
									lemma_dictionary=lemma_dictionary, 
									pos_dictionary=pos_dictionary, 
									category_dictionary=category_dictionary,
									full_scope = False,
									delete_temp=delete_temp
									), input_files, chunksize = 1))
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for input files#

		centroid_df_lexical = get_centroid(vector_file_list, delete_temp)
		
		#--------------------------------------------------------#
		#SECOND, get centroid_df for full_scope: complete grammar #
		print("Creating full grammar centroid.")
		
		vector_file_list = []
		
		#Now, multi-process for input files#
		pool_instance=mp.Pool(processes = number_of_cpus_extract, maxtasksperchild = None)
		vector_file_list.append(pool_instance.map(partial(process_learn_usage, 
									output_folder=temp_folder, 
									candidate_list=candidate_list, 
									lemma_list=lemma_list, 
									pos_list=pos_list, 
									word_list=word_list, 
									category_list=category_list, 
									semantic_category_dictionary=semantic_category_dictionary, 
									sequence_list=sequence_list, 
									max_construction_length=max_construction_length, 
									annotation_types=annotation_types, 
									pruned_vector_dataframe=pruned_vector_dataframe, 
									encoding_type=encoding_type, 
									phrase_constituent_list=phrase_constituent_list, 
									lemma_dictionary=lemma_dictionary, 
									pos_dictionary=pos_dictionary, 
									category_dictionary=category_dictionary,
									full_scope = True,
									delete_temp=delete_temp
									), input_files, chunksize = 1))
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for input files#

		centroid_df_full = get_centroid(vector_file_list, delete_temp)
		
		#Now take and add to new grammar and usage model#
		usage_model = data_file_usage

		write_dictionary = {}
	
		write_dictionary['candidate_list'] = candidate_list
		write_dictionary['lemma_list'] = lemma_list
		write_dictionary['pos_list'] = pos_list
		write_dictionary['word_list'] = word_list
		write_dictionary['category_list'] = category_list
		write_dictionary['semantic_category_dictionary'] = semantic_category_dictionary
		write_dictionary['sequence_list'] = sequence_list
		write_dictionary['max_construction_length'] = max_construction_length
		write_dictionary['annotation_types'] = annotation_types
		write_dictionary['pruned_vector_dataframe'] = pruned_vector_dataframe
		write_dictionary['encoding_type'] = encoding_type
		write_dictionary['phrase_constituent_list'] = phrase_constituent_list
		write_dictionary['emoji_dictionary'] = emoji_dictionary
		write_dictionary['lemma_dictionary'] = lemma_dictionary
		write_dictionary['pos_dictionary'] = pos_dictionary
		write_dictionary['category_dictionary'] = category_dictionary
	
		write_dictionary['centroid_df_full'] = centroid_df_full
		write_dictionary['centroid_df_lexical'] = centroid_df_lexical
	
		write_candidates(usage_model, write_dictionary)	
		
		end_all = time.time()
		print("All files completed in " + str(end_all - start_all) + " seconds.")
		
		return
#---------------------------------------------------------------------------------------------#

#Prevent pool workers from starting here#
if __name__ == '__main__':
#---------------------------------------#
	
	#CODE FOR RUNNING FROM COMMAND LINE#
	import sys

	#Get parameters file to use#
	parameters_file = str(sys.argv[1])

	#Import parameter values and global variables#
	#All parameter and global variables must be prefaced with "pm."  ---#

	import importlib
	from learn_usage import learn_usage
	from learn_usage import process_learn_usage

	try:
		pm = importlib.import_module(parameters_file)
	except ImportError:
		print("Error in specified parameters file. Format is 'files_parameters.FILENAME'")
			
	learn_usage(pm.data_file_constructions, 
					pm.input_files, 
					pm.input_folder, 
					pm.output_folder,
					pm.temp_folder,
					pm.annotate_pos, 
					pm.number_of_cpus_extract, 
					pm.settings_dictionary,
					pm.encoding_type,
					pm.docs_per_file,
					pm.data_file_usage,
					pm.delete_temp,
					pm.use_metadata
					)
					
	#END CODE FOR RUNNING FROM COMMAND LINE#