#-----------------------------------------------#
def process_create_unit_index(Parameters, Grammar, input_files = None, run_parameter = 0):
							
	#Protect for multi-processing#
	if run_parameter == 0:
		run_parameter = 1
	#----------------------------#
	
		import multiprocessing as mp
		import cytoolz as ct
		from functools import partial
		
		from process_input.create_unit_index import create_unit_index
		from association_measures.split_output_files import split_output_files
		
		if input_files == None:
			input_files = Parameters.Input_Files
		
		#Make a list of lists of files to send each sub-list to a different process#
		input_files = split_output_files(input_files, Parameters.CPUs_General)
		
		#Multi-process ingest#
		#Multi-process #
		pool_instance=mp.Pool(processes = Parameters.CPUs_General, maxtasksperchild = None)
		result_list = pool_instance.map(partial(create_unit_index,
												Parameters = Parameters,
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
		
		#Save previously found idioms frequency#
		if Grammar != None:
			if Grammar.Idiom_List != []:
				idiom_freq_dict = {}
				
				for idiom in Grammar.Idiom_List:
					idiom_label = idiom[1]
					
					if idiom_label in lemma_dictionary:
						idiom_freq_dict[idiom_label] = lemma_dictionary[idiom_label]
		#Ensure previously found idioms make the cut#
		
		#Reduce unit inventories by removing infrequent labels#
		above_threshold = lambda x: x > Parameters.Freq_Threshold_Individual
		
		lemma_dictionary = ct.valfilter(above_threshold, lemma_dictionary)
		pos_dictionary = ct.valfilter(above_threshold, pos_dictionary)
		word_dictionary = ct.valfilter(above_threshold, word_dictionary)
		category_dictionary = ct.valfilter(above_threshold, category_dictionary)
		
		#Ensure previously found idioms make the cut#
		if Grammar != None:
			if Grammar.Idiom_List != []:
				for idiom in Grammar.Idiom_List:
					idiom_label = idiom[1]
					
					if idiom_label not in lemma_dictionary and idiom_label in idiom_freq_dict:
						lemma_dictionary[idiom_label] = idiom_freq_dict[idiom_label]
		#Ensure previously found idioms make the cut#
		
		full_dictionary = {}
		full_dictionary['lemma'] = lemma_dictionary
		full_dictionary['pos'] = pos_dictionary
		full_dictionary['word'] = word_dictionary
		full_dictionary['category'] = category_dictionary	
								
		return full_dictionary
#-----------------------------------------------#