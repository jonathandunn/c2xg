#---------------------------------------------------------------------------------------------#
#FUNCTION: annotate_files --------------------------------------------------------------------#
#INPUT: Raw text files to annotate and parameters; may or may not contain meta-data ----------#
#---Meta-data format: Field:Value,Field:Value\tText ------------------------------------------#
#OUTPUT: Write tabbed CoNLL output for main script, return list of files and maybe metadata --#
#---------------------------------------------------------------------------------------------#
def annotate_files(input_folder, 
					input_file, 
					settings_dictionary, 
					encoding_type, 
					number_processes, 
					emoji_dictionary, 
					docs_per_file,
					use_metadata = False,
					run_parameter = 0
					):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		#Run parameter keeps pool workers out for this imported module#
		run_parameter = 1
	
		from functions_annotate.load_utf8 import load_utf8
		from functions_annotate.stanford_start import stanford_start
		from functions_annotate.stanford_run import stanford_run
		from functions_annotate.stanford_stop import stanford_stop
		from functions_annotate.write_conll import write_conll
		from functions_annotate.process_lines import process_lines
		from functions_annotate.rdrpos_run import rdrpos_run
		from functions_annotate.get_write_list import get_write_list
		import time
		
		import multiprocessing as mp
		from functools import partial
		
		input_file = input_folder + "/" + input_file

		nlp_system = settings_dictionary['nlp_system']
		memory_limit = settings_dictionary['memory_limit']
		working_directory = settings_dictionary['working_directory']
		pos_model = settings_dictionary['pos_model']
		language = settings_dictionary['language']
		
		#If meta-data flag is on, create (ID, META-DATA) tuples for re-populating in vector representation#
		if use_metadata == True:
			from functions_annotate.get_metadata_tuples import get_metadata_tuples
			metadata_tuples = get_metadata_tuples(input_file, encoding_type)
	
		time_start = time.time()
		print("")
		print("Loading input file, detecting encoding, and converting to " + str(encoding_type))
		line_list = load_utf8(input_file, encoding_type, language, memory_limit, working_directory, use_metadata)
		time_end = time.time()
		print("Time to standardize encoding: " + str(time_end - time_start))
		print("")
		
		time_start = time.time()
		print("Tokenizing lines and dealing with emojis, etc.")
		#Multi-process lines to tokenizing and emoji detection#
		pool_instance=mp.Pool(processes = number_processes, maxtasksperchild = None)
		tokenized_line_list = pool_instance.map(partial(process_lines, emoji_dictionary=emoji_dictionary), line_list, chunksize = 1000)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for tokenizing and emoji detection#
		time_end = time.time()
		print("Time to tokenize and find emojis: " + str(time_end - time_start))
		print("")
		
		del line_list
	
		time_start = time.time()
		if nlp_system == "stanford":
	
			print("Annotating files using Stanford CoreNLP: " + str(input_file))
			process_id = stanford_start(memory_limit, working_directory)
		
			#Multi-process lines to Stanford Core NLP#
			pool_instance=mp.Pool(processes = number_processes, maxtasksperchild = None)
			text_dictionary = pool_instance.map(partial(stanford_run, pos_model=pos_model), tokenized_line_list, chunksize = 1000)
			pool_instance.close()
			pool_instance.join()
			#End multi-processing for tagging with Stanford CoreNLP#
		
			stanford_stop(process_id)
		
		elif nlp_system == "rdrpos":
		
			print("Annotating files using RDR POS-Tagger:" + str(input_file))
			from functions_annotate.rdrpos_tagger.Utility.Utils import readDictionary
				
			import os
			import codecs
			import platform
	
			if language == "English":
				from functions_annotate.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger4En import RDRPOSTagger4En
				from functions_annotate.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger4En import unwrap_self_RDRPOSTagger4En
				from functions_annotate.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger4En import printHelp
				r = RDRPOSTagger4En()
				
			elif language == "Vietnamese":
				from functions_annotate.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger4Vn import RDRPOSTagger4Vn
				from functions_annotate.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger4Vn import unwrap_self_RDRPOSTagger4Vn
				from functions_annotate.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger4Vn import printHelp
				r = RDRPOSTagger4Vn()
				
			else:
				from functions_annotate.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
				from functions_annotate.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import unwrap_self_RDRPOSTagger
				from functions_annotate.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import printHelp
				r = RDRPOSTagger()
			
			#Check and Change directory if necessary; only once if multi-processing#
			current_dir = os.getcwd()

			if platform.system() == "Windows":
				slash_index = current_dir.rfind("\\")
				
			else:
				slash_index = current_dir.rfind("/")
				
			current_dir = current_dir[slash_index+1:]
			
			if current_dir == "Utility":
				os.chdir("../../../")
			#End directory check#
			
			model_string = "./files_data/pos_rdr/" + language + ".RDR"
			dict_string = "./files_data/pos_rdr/"  + language + ".DICT"
		
			r.constructSCRDRtreeFromRDRfile(model_string)
			DICT = readDictionary(dict_string)
			
			#Multi-process lines to RDR Pos-Tagger#
			pool_instance=mp.Pool(processes = number_processes, maxtasksperchild = None)
			text_dictionary = pool_instance.map(partial(rdrpos_run, encoding_type=encoding_type, r=r, DICT=DICT, language=language), tokenized_line_list, chunksize = 1000)
			pool_instance.close()
			pool_instance.join()
			#End multi-processing for tagging#
			
		time_end = time.time()
		print("Time to pos-tag: " + str(time_end - time_start))
		print("")
		
		del tokenized_line_list
	
		#Make a list of tuples for multi-processing writing CONLL files:#
		#--- (Doc Number, Start Index, End Index) ----------------------#
		text_dictionary = get_write_list(text_dictionary, docs_per_file)
	
		time_start = time.time()
		print("Now reformatting files for ingest.")
		
		#Multi-process write results#
		pool_instance=mp.Pool(processes = number_processes, maxtasksperchild = None)
		conll_files = pool_instance.map(partial(write_conll, 
													input_file = input_file, 
													encoding_type = encoding_type
													), text_dictionary, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing #
		
		time_end = time.time()
		print("Time to write CoNLL format files: " + str(time_end - time_start))
		print("")
		
		if use_metadata == True:
			return (conll_files, metadata_tuples)
		
		else:
			return conll_files
#---------------------------------------------------------------------------------------------#