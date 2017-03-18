#---------------------------------------------------------------------------------------------#
#FUNCTION: annotate_files --------------------------------------------------------------------#
#INPUT: Raw text files to annotate and parameters; may or may not contain meta-data ----------#
#---Meta-data format: Field:Value,Field:Value\tText ------------------------------------------#
#OUTPUT: Write tabbed CoNLL output for main script, return list of files and maybe metadata --#
#---------------------------------------------------------------------------------------------#
def annotate_files(input_file, Parameters, Grammar, metadata = False, same_size = False, run_parameter = 0):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		#Run parameter keeps pool workers out for this imported module#
		run_parameter = 1
	
		import time
		import multiprocessing as mp
		from functools import partial
		from process_input.load_utf8 import load_utf8
		from process_input.write_conll import write_conll
		from process_input.process_lines import process_lines
		from process_input.rdrpos_run import rdrpos_run
		from process_input.get_write_list import get_write_list
		
		input_file = Parameters.Input_Folder + "/" + input_file

		#If meta-data flag is on, create (ID, META-DATA) tuples for re-populating in vector representation#
		if metadata == True:
			from process_input.get_metadata_tuples import get_metadata_tuples
			metadata_tuples = get_metadata_tuples(input_file, Parameters)
	
		print("")
		print("Loading input file, detecting encoding, and converting to " + str(Parameters.Encoding_Type))
		
		time_start = time.time()
		line_list = load_utf8(input_file, Parameters)
		time_end = time.time()
		
		print("Time to standardize encoding: " + str(time_end - time_start))
		print("")
		
		
		print("Tokenizing lines and dealing with emojis, etc.")
		time_start = time.time()
		
		#Multi-process lines to tokenizing and emoji detection#
		pool_instance=mp.Pool(processes = Parameters.CPUs_General, maxtasksperchild = None)
		tokenized_line_list = pool_instance.map(partial(process_lines, Parameters = Parameters, Grammar = Grammar), line_list, chunksize = 1000)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for tokenizing and emoji detection#
		
		time_end = time.time()
		print("Time to tokenize and find emojis: " + str(time_end - time_start))
		print("")
		
		del line_list
	
		time_start = time.time()
		
		print("Annotating files using RDR POS-Tagger:" + str(input_file))
		from process_input.rdrpos_tagger.Utility.Utils import readDictionary
				
		import os
		import codecs
		import platform
	
		from process_input.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
		from process_input.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import unwrap_self_RDRPOSTagger
		from process_input.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import printHelp
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
			
		model_string = "./files_data/pos_rdr/" + Parameters.Language + ".RDR"
		dict_string = "./files_data/pos_rdr/"  + Parameters.Language + ".DICT"
		
		r.constructSCRDRtreeFromRDRfile(model_string)
		DICT = readDictionary(dict_string)
		
		if Grammar.Idiom_List != []:
			for idiom in Grammar.Idiom_List:
				DICT[idiom[1]] = idiom[2]
				
		#Multi-process lines to RDR Pos-Tagger#
		pool_instance=mp.Pool(processes = Parameters.CPUs_General, maxtasksperchild = None)
		text_dictionary = pool_instance.map(partial(rdrpos_run, 
														r = r, 
														DICT = DICT, 
														Parameters = Parameters, 
														Idiom_List = Grammar.Idiom_List
														), tokenized_line_list, chunksize = 1000)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for tagging#

		time_end = time.time()
		print("Time to pos-tag: " + str(time_end - time_start))
		print("")
		
		del tokenized_line_list
	
		#Make a list of tuples for multi-processing writing CONLL files:#
		#--- (Doc Number, Start Index, End Index) ----------------------#
		if same_size == True:
			lines_per_file = 99999999999999999
		else:
			lines_per_file = Parameters.Lines_Per_File
			
		text_dictionary = get_write_list(text_dictionary, lines_per_file)
	
		time_start = time.time()
		print("Now reformatting files for ingest.")
		
		#Multi-process write results#
		pool_instance=mp.Pool(processes = Parameters.CPUs_General, maxtasksperchild = None)
		conll_files = pool_instance.map(partial(write_conll, 
													input_file = input_file, 
													encoding_type = Parameters.Encoding_Type
													), text_dictionary, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing #
		
		time_end = time.time()
		print("Time to write CoNLL format files: " + str(time_end - time_start))
		print("")
		
		if metadata == True:
			return (conll_files, metadata_tuples)
		
		else:
			return conll_files
#---------------------------------------------------------------------------------------------#