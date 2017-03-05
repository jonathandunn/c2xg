#---------------------------------------------------------------------------------------------#
#FUNCTION: merge_output ----------------------------------------------------------------------#
#INPUT: List of output files from Learning.2.Processing.py -----------------------------------#
#OUTPUT: Dictionary with all acceptable candidates, their frequency, other info --------------#
#---------------------------------------------------------------------------------------------#
def merge_output(output_files, frequency_threshold, number_of_cpus, run_parameter = 0):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
		
		print("Starting to load and merge candidate files.")
		
		from association_measures.split_output_files import split_output_files
		from association_measures.process_merge_output import process_merge_output
		from association_measures.get_candidate_count import get_candidate_count
		
		import multiprocessing as mp
		from functools import partial
		import cytoolz as ct
		
		print("Loading files from disk.")

		#If multiple CPUs, first join many equal sized dictionaries and then merge these#
		if number_of_cpus > 1 and len(output_files) > 3:
			
			print("Splitting candidate files to distribute across processes")
			output_files = split_output_files(output_files, number_of_cpus)
		
			#Start multi-processing#
			pool_instance=mp.Pool(processes = number_of_cpus, maxtasksperchild = None)
			merged_list = pool_instance.map(partial(process_merge_output, 
													action = "Load",
													), output_files, chunksize = 1)
			pool_instance.close()
			pool_instance.join()
			#End multi-processing#
			
			print("Now merging pre-merged dictionaries.")
			final_dictionary = process_merge_output(merged_list, action = "Pass")
		
		#If only one CPU, just merge all at once from disk#
		elif number_of_cpus == 1 or len(output_files) < 3:
			
			print("Not distributing candidate_files")
			final_dictionary = process_merge_output(output_files, action = "Load")
		
		#Now, do frequency pruning#
		candidate_dictionary = final_dictionary['candidate_dictionary']
		Grammar = final_dictionary['Grammar']
		
		total = get_candidate_count(candidate_dictionary)
		
		print("")
		print("Total candidates after merging: " + str(total))
		
		above_threshold = lambda x: x > frequency_threshold
		final_candidate_dictionary = {}
	
		for key in candidate_dictionary.keys():
			final_candidate_dictionary[key] = ct.valfilter(above_threshold, candidate_dictionary[key])
		
		del candidate_dictionary
		
		total = get_candidate_count(final_candidate_dictionary)
		print("Total candidates after applying frequency threshold of " + str(frequency_threshold) + ": " + str(total))
				
		final_dictionary['candidate_dictionary'] = final_candidate_dictionary
		
		return final_dictionary, Grammar
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#