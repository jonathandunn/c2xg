#-----------------------------------------------#
def process_create_unit_index(input_files, 
							frequency_threshold_individual, 
							encoding_type, 
							semantic_category_dictionary, 
							illegal_pos,
							number_of_cpus,
							run_parameter = 0
							):
							
	#Protect for multi-processing#
	if run_parameter == 0:
		run_parameter = 1
	#----------------------------#
	
		import multiprocessing as mp
		import cytoolz as ct
		from functools import partial
		
		from functions_input.create_unit_index import create_unit_index
		from functions_candidate_evaluation.split_output_files import split_output_files
		
		#Make a list of lists of files to send each sub-list to a different process#
		input_files = split_output_files(input_files, number_of_cpus)
		
		#Multi-process ingest#
		#Multi-process #
		pool_instance=mp.Pool(processes = number_of_cpus, maxtasksperchild = None)
		result_list = pool_instance.map(partial(create_unit_index,
												encoding_type = encoding_type, 
												semantic_category_dictionary = semantic_category_dictionary, 
												illegal_pos = illegal_pos,
												), input_files, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
				
		#Merge results#
		lemma_list = []
		pos_list = []
		word_list = []
		category_list = []
		
		for i in range(0, len(result_list)):
			lemma_list.append(result_list[i]["lemma"])
			pos_list.append(result_list[i]["pos"])
			word_list.append(result_list[i]["word"])
			category_list.append(result_list[i]["category"])
			
		del result_list
		
		lemma_dictionary = ct.merge([x for x in lemma_list])
		pos_dictionary = ct.merge([x for x in pos_list])
		word_dictionary = ct.merge([x for x in word_list])
		category_dictionary = ct.merge([x for x in category_list])
		
		del lemma_list
		del pos_list
		del word_list
		del category_list
		
		print("")
		print("Removing infrequent labels and creating label indexes")
		
		#Reduce unit inventories by removing infrequent labels#
		above_threshold = lambda x: x > frequency_threshold_individual
		
		lemma_dictionary = ct.valfilter(above_threshold, lemma_dictionary)
		pos_dictionary = ct.valfilter(above_threshold, pos_dictionary)
		word_dictionary = ct.valfilter(above_threshold, word_dictionary)
		category_dictionary = ct.valfilter(above_threshold, category_dictionary)
		
		full_dictionary = {}
		full_dictionary['lemma'] = lemma_dictionary
		full_dictionary['pos'] = pos_dictionary
		full_dictionary['word'] = word_dictionary
		full_dictionary['category'] = category_dictionary	
								
		return full_dictionary
#-----------------------------------------------#