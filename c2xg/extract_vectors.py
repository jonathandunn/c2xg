#-----C2xG, v 1.0 ----------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#---- Copyright, 2015-2016 Jonathan E. Dunn --------------------------------------------------#
#---------- www.jdunn.name -------------------------------------------------------------------#
#---------- jonathan.edwin.dunn@gmail.com ----------------------------------------------------#
#---------- Illinois Institute of Technology, Department of Computer Science -----------------#
#---------------------------------------------------------------------------------------------#
#This script takes a model for autonomous feature extraction and produces a feature vector ---#
#--- for each text in the input file using the supplied model. -------------------------------#
#---------------------------------------------------------------------------------------------#

def process_extract_vectors(input_file, 
							input_folder, 
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
							centroid_df,
							metadata_dictionary,
							delete_temp = False,
							use_centroid = True,
							use_metadata = True,
							relative_freq = False,
							full_scope = True,
							write_output = True
							):
	
	import time 
	import csv
	import pandas as pd
	import numpy as np
	from functions_input.pandas_open import pandas_open
	from functions_constituent_reduction.expand_sentences import expand_sentences
	from functions_autonomous_extraction.process_extraction import process_extraction
	from functions_autonomous_extraction.get_coverage_column import get_coverage_column
	from functions_autonomous_extraction.get_meta_data import get_meta_data
	from functions_candidate_extraction.write_candidates import write_candidates
	
	
	start_beginning = time.time()

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
		
		
		
	else:
	
		input_dataframe.loc[:, 'Alt'] = 0
		
	full_sentences = input_dataframe.loc[:,'Sent'].tolist()
	number_of_sentences = full_sentences[-1]	

	print("Finished expanding " + str(number_of_sentences) + " sentences.")
	print("")
	#---------------------------------------------------------------------------------------------#
	#EXTRACT FEATURES ----------------------------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#

	start_extraction = time.time()
	print("Begin feature extraction.")
	full_vector_df = process_extraction(candidate_list, 
												max_construction_length, 
												input_dataframe, 
												lemma_list, 
												pos_list, 
												category_list, 
												number_of_sentences,
												full_scope,
												relative_freq
												)
	
	end_extraction = time.time()
	print("Done with feature extraction: " + str(end_extraction - start_extraction))

	#Save vector#
	print("")
	
	if use_centroid == True:
	
		from functions_autonomous_extraction.get_centroid_normalization import get_centroid_normalization
		print("Normalizing by usage-based centroid.")
		full_vector_df = get_centroid_normalization(full_vector_df, centroid_df)
	
	#Repopulate and add meta-data#
	if use_metadata == True:
	
		print("Adding meta-data from input file.")
		metadata_tuples = metadata_dictionary[input_file]
		full_vector_df = get_meta_data(full_vector_df, metadata_tuples)
		
	#Extract column names and save separately#
	column_metadata = list(full_vector_df.columns)
	column_ids = [i for i in range(len(column_metadata))]
	full_vector_df.columns = column_ids
	
	end_beginning = time.time()
	print("File " + str(input_file) + ": " + str(end_beginning - start_beginning))
	
	#Save vectors to file and return#
	if write_output == True:
		
		from functions_input.get_temp_filename import get_temp_filename
		print("Saving vectors to file.")
		
		if use_centroid == True:
			output_name = get_temp_filename(input_file, ".Centroid.Features")
			
		elif relative_freq == True:
			output_name = get_temp_filename(input_file, ".Relative.Features")
			
		else:
			output_name = get_temp_filename(input_file, ".Raw.Features")
	
		write_candidates(output_name + ".Columns", column_metadata) #Save column names#
		full_vector_df.to_hdf(output_name, "Table", format='fixed', complevel=9, complib="blosc")
		
		return output_name
		
	elif write_output == False:
		
		return full_vector_df, column_metadata
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def extract_vectors(model_file, 
					encoding_type, 
					number_processes, 
					input_folder, 
					output_folder, 
					annotate_pos, 
					input_files,
					use_centroid,
					use_metadata,
					full_scope,
					relative_freq,
					delete_temp,
					settings_dictionary,
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
		
		from extract_vectors import process_extract_vectors

		#Import required script-specific modules#
		from functions_annotate.annotate_files import annotate_files
		from functions_candidate_extraction.read_candidates import read_candidates
		
		if use_centroid == True:
			relative_freq = False

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
		encoding_type = encoding_type
		phrase_constituent_list = write_dictionary['phrase_constituent_list']
		emoji_dictionary = write_dictionary['emoji_dictionary']
		lemma_dictionary = write_dictionary['lemma_dictionary']
		category_dictionary = write_dictionary['category_dictionary']
		pos_dictionary = write_dictionary['pos_dictionary']
		
		if use_centroid == True:
		
			if full_scope == True:
				print("Loading full grammar centroid expectations.")
				centroid_df = write_dictionary['centroid_df_full']
				
			elif full_scope == False:
				print("Loading lexical only centroid expectations.")
				centroid_df = write_dictionary['centroid_df_lexical']
					
		else:
			centroid_df = ""

		#Fix elements which throw off ARFF exporting#
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
		
			updated_input = []
			metadata_dictionary = {}
			
			for input_file in input_files:
			
				input_short = input_file
				output_name = output_folder + "/" + input_short + ".Vectors.hdf5"
				
				#Different return if using meta-data and annotating or not using meta-data#
				if use_metadata == True:
					
					conll_file, metadata_tuples = annotate_files(input_folder, 
											input_file, 
											settings_dictionary, 
											encoding_type, 
											number_processes, 
											emoji_dictionary, 
											docs_per_file = 100000000000,
											use_metadata = use_metadata
											)

					updated_input += conll_file
					metadata_dictionary[str(conll_file[0])] = metadata_tuples
				
				elif use_metadata == False:

					conll_file = annotate_files(input_folder, 
											input_file, 
											settings_dictionary, 
											encoding_type, 
											number_processes, 
											emoji_dictionary, 
											docs_per_file = 100000000000,
											use_metadata = use_metadata
											)
					
					updated_input += conll_file					
				#Done with annotating meta-data check#
				
		elif annotate_pos == False:
		
			if use_metadata == True:
				
				print("Must annotate files as part of feature extraction with meta-data.")
				sys.kill()
				
			else:
			
				updated_input = input_files
				metadata_dictionary = {}
		#--------------------------------------------------------------------------------------------#

		holder_list = []
		#Now, multi-process for input files#
		pool_instance=mp.Pool(processes = number_processes, maxtasksperchild = None)
		pool_instance.map(partial(process_extract_vectors, 
										input_folder=input_folder, 
										output_folder=output_folder, 
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
										centroid_df=centroid_df,
										metadata_dictionary=metadata_dictionary,
										delete_temp=delete_temp,
										use_centroid=use_centroid,
										use_metadata=use_metadata,
										relative_freq=relative_freq,
										full_scope=full_scope
											), updated_input, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for input files#
	
		end_all = time.time()
		print("All files completed in " + str(end_all - start_all) + " seconds.")
		
		return
#------------------------------------------------------------------------------------------------------------------------#

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
	from extract_vectors import extract_vectors
	from extract_vectors import process_extract_vectors

	try:
		pm = importlib.import_module(parameters_file)
	except ImportError:
		print("Error in specified parameters file. Format is 'files_parameters.FILENAME'")
		
	if pm.use_centroid == False:
		data_file = pm.data_file_constructions
		
	else:
		data_file = pm.data_file_usage
	
	extract_vectors(data_file, 
					pm.encoding_type, 
					pm.number_of_cpus_extract, 
					pm.input_folder, 
					pm.output_folder, 
					pm.annotate_pos, 
					pm.input_files,
					pm.use_centroid,
					pm.use_metadata,
					pm.full_scope,
					pm.relative_freq,
					pm.delete_temp,
					pm.settings_dictionary
					)

	#END CODE TO RUN FROM COMMAND LINE#