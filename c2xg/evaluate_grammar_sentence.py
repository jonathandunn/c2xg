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

#Define function for multi-processing# -------------------------------------------------------#
def process_evaluate_grammar(input_file, 
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
								category_dictionary
								):
	
	import time 
	import csv
	import pandas
	from functions_input.pandas_open import pandas_open
	from functions_constituent_reduction.expand_sentences import expand_sentences
	from functions_autonomous_extraction.process_extraction import process_extraction
	from functions_autonomous_extraction.get_coverage_column import get_coverage_column
	from functions_annotate.annotate_files import annotate_files
	
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
									write_output = False
									)

	#---------------------------------------------------------------------------------------------#
	#EXPAND TEST FILE ----------------------------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#

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
	
	full_sentences = input_dataframe.loc[:,'Sent'].tolist()
	number_of_sentences = len(set(full_sentences))
	
	print("Finished expanding sentences.")
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
												full_scope = True
												)
	
	end_extraction = time.time()
	print("Done with feature extraction: " + str(end_extraction - start_extraction))

	#Save vector#
	print("")
	print("Standardizing dtypes and getting coverage column.")
	
	full_vector_df.fillna(value = 0, inplace=True)
	full_vector_df = get_coverage_column(full_vector_df)
	
	return_df = full_vector_df.loc[:,['Coverage', 'Length']]

	#---------------------------------------------------------------------------------------------#
	#PRINT TIME ELAPSED --------------------------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#
	print("")
	end_beginning = time.time()
	print("File " + str(input_file) + ": " + str(end_beginning - start_beginning))
	
	return return_df
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def evaluate_grammar(model_file, 
						encoding_type, 
						number_processes, 
						input_folder, 
						output_folder, 
						annotate_pos, 
						input_files,
						coverage_graph,
						settings_dictionary,
						run_parameter = 0
						):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameters = 1
		#---------------------------------------------------------------------------------------------#
		#IMPORT DEPENDENCIES -------------------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#

		import time
		import sys
		import cytoolz as ct
		import multiprocessing as mp
		import pandas as pd
		import numpy as np
		from functools import partial
		import matplotlib.pyplot as plt
		import matplotlib
		matplotlib.style.use('ggplot')
		
		from evaluate_grammar import process_evaluate_grammar

		#Import required script-specific modules#
		from functions_candidate_extraction.read_candidates import read_candidates
		from functions_candidate_extraction.write_candidates import write_candidates
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
		encoding_type = encoding_type
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
		
			updated_input = []
			
			for input_file in input_files:
			
				output_name = input_file + ".Vectors"
				
				conll_file = annotate_files(input_folder, 
											input_file, 
											settings_dictionary, 
											encoding_type, 
											number_processes, 
											emoji_dictionary, 
											docs_per_file = 100000000000
											)
						
				updated_input += conll_file
				
		elif annotate_pos == False:
		
			updated_input = []
	
			for filename in input_files:
				input_short = filename
				updated_input.append(input_folder + "/" + filename)
			
			input_files = updated_input

		input_files = updated_input
		#----------------------------------------------------------------------------------------------#
	
		holder_list = []
		#Now, multi-process for input files#
		pool_instance=mp.Pool(processes = number_processes, maxtasksperchild = None)
		holder_list.append(pool_instance.map(partial(process_evaluate_grammar, 
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
														category_dictionary=category_dictionary
														), input_files, chunksize = 1))
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for input files#
	
		coverage_list = []
		for i in range(len(holder_list[0])):
			coverage_list.append(holder_list[0][i])
		
		del holder_list
	
		coverage_df = pd.concat([x for x in coverage_list], axis = 0, ignore_index = True)
		coverage_df = coverage_df.query("(Length < 100 and Length > 1)")

		print("Plotting coverage results as file: " + str(coverage_graph))
	
		plot = coverage_df.loc[:,'Coverage'].plot(kind='hist', bins=100, cumulative=-1, range=(1, 100), normed=True, alpha=0.5)
		plot.set_ylim([0,1])
		plot.set_xlabel("Texts With Over X Features")
		start, end = plot.get_xlim()
		plot.xaxis.set_ticks(np.arange(start, end, 5))
		plot.set_ylabel("Percentage of Test Set")
		plot.set_title("Coverage With Lexical Items")
		fig = plot.get_figure()
		fig.savefig(coverage_graph)
	
		end_all = time.time()
		print("All files completed in " + str(end_all - start_all) + " seconds.")
	
	return
#---------------------------------------------------------------------------------------------#

#Prevent pool workers from starting here#
if __name__ == '__main__':
#---------------------------------------#

	#START CODE FOR RUNNING FROM COMMAND LINE#
	import sys

	#Get parameters file to use#
	parameters_file = str(sys.argv[1])

	#Import parameter values and global variables#
	#All parameter and global variables must be prefaced with "pm."  ---#

	import importlib
	from evaluate_grammar import evaluate_grammar
	from evaluate_grammar import process_evaluate_grammar

	try:
		pm = importlib.import_module(parameters_file)
	except ImportError:
		print("Error in specified parameters file. Format is 'files_parameters.FILENAME'")
	
	evaluate_grammar(pm.data_file_constructions, 
						pm.encoding_type, 
						pm.number_of_cpus_extract, 
						pm.input_folder, 
						pm.output_folder, 
						pm.annotate_pos, 
						pm.input_files,
						pm.coverage_graph,
						pm.settings_dictionary
						)
						
	#END CODE FOR RUNNING FROM COMMAND LINE#