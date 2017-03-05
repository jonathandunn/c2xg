#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

#-- Helper function for learning a grammar from provided hypothesis space using association vectors and MDL component

def get_mdl_grammar(Parameters, 
						Grammar,
						training_testing_files,
						testing_files,						
						association_df, 
						max_candidate_length, 
						candidate_list_formatted, 
						expand_check, 
						atomic_unit_types,
						run_parameter = 0
						):
	
	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
		
		print("")
		print("\tStarting C2xG.Get_MDL_Grammar")
		print("")
		
		import datetime
		import time
		import sys
		import multiprocessing as mp
		import pandas as pd
		from functools import partial
		import cytoolz as ct

		#Import required script-specific modules#
		from candidate_extraction.read_candidates import read_candidates
		from candidate_extraction.write_candidates import write_candidates
		from candidate_selection.process_coverage import process_coverage
		from candidate_selection.tabu_search_restarts import tabu_search_restarts
		from candidate_selection.write_results_pruned import write_results_pruned
		from candidate_selection.write_model import write_model
	
		start_beginning = datetime.datetime.now().time()
	
		#1: Get coverage DataFrame for each candidate -----------------------------------------------#
		
		print("")
		print("\t\tGetting coverage information for grammar evaluation.")
		
		#Collect indexes covered per construction as partial pre-calculation of the MDL metric#
		#Multi-process below by search_df ---------------------#
		training_testing_files, testing_files = process_coverage(Parameters, 
																Grammar, 
																training_testing_files, 
																testing_files, 
																max_candidate_length, 
																candidate_list_formatted, 
																association_df, 
																expand_check
																)
	
		#Clean up memory#
		del candidate_list_formatted

		#2: Start MDL learning process ---------------------------------------------------------------#

		print("")
		print("\t\tBegin grammar generation and evaluation for MDL learning process.")
		print("")

		optimum_grammar_df, weighted_mdl = tabu_search_restarts(training_testing_files, testing_files, Parameters, Grammar, max_candidate_length, full_cxg = expand_check)

	return optimum_grammar_df, weighted_mdl
#-----------------------------------------------------------------------------------------------#