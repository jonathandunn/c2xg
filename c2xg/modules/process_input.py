#General imports
import codecs
import os
import os.path
import csv
import time
import random
import pandas as pd
import cytoolz as ct
from functools import partial
import multiprocessing as mp
		
#INPUT: List of unicode sequences with emoji descriptions ------------------------------------#
#OUTPUT: Dictionary with UTF-8 string as keys and descriptions as values ---------------------#

def create_emoji_dictionary(emoji_file):

	emoji_dictionary = {}
	
	with codecs.open(emoji_file, "r", encoding = "utf-8") as fo:
	
		for line in fo:
		
			line = line.replace("\r","").replace("\n","")
			line_list = line.split("\t")
			
			current_symbol = line_list[0]
			current_label = " EMOJI" + str(line_list[1].upper()) + "EMOJI "
			
			emoji_dictionary[current_symbol] = current_label
	
	emoji_list = list(emoji_dictionary.keys())
	emoji_list.sort(key = len, reverse = True)
	
	emoji_dictionary['emoji_list'] = emoji_list

	return emoji_dictionary
#---------------------------------------------------------------------------------------------#
#INPUT: Semantic category dictionary ---------------------------------------------------------#
#OUTPUT: Index of semantic categories --------------------------------------------------------#
#Take file and return full semantic category dictionary --------------------------------------#

def create_category_index(category_dictionary):

	category_index = []

	for word in category_dictionary.keys():
		if category_dictionary[word] not in category_index:
			category_index.append(category_dictionary[word])
			
	category_index = sorted(category_index)	

	category_index.insert(0, "n/a")
		
	return category_index
#---------------------------------------------------------------------------------------------#
#INPUT: Filename of semantic category dictionary ---------------------------------------------#
#Take file and return full semantic category dictionary --------------------------------------#

def create_category_dictionary(filename, encoding_type):

	semantic_category_dictionary = {}
	
	fo = open(filename, "r", encoding=encoding_type)
	
	for line in fo:
	
		line = line.replace("\n","")
		line_list = line.split(",")
		semantic_category_dictionary[line_list[0]] = line_list[1]
		
	fo.close()	
		
	return semantic_category_dictionary
#---------------------------------------------------------------------------------------------#
#OUTPUT: None --------------------------------------------------------------------------------#
#For each data file, make sure does not exist before writing current data files --------------#

def check_folders(input_folder, 
					temp_folder, 
					candidate_folder,
					debug_folder, 
					output_folder,
					dict_folder,
					pos_training_folder,
					pos_testing_folder,
					parameters_folder
					):

	if os.path.isdir(input_folder) == False:
		os.makedirs(input_folder)
		print("Creating input folder")
			
	if os.path.isdir(output_folder) == False:
		os.makedirs(output_folder)
		print("Creating output folder")
		
	if os.path.isdir(parameters_folder) == False:
		os.makedirs(parameters_folder)
		print("Creating parameters folder")
		
	if os.path.isdir(temp_folder) == False:
		os.makedirs(temp_folder)
		print("Creating temp folder")
		
	if os.path.isdir(candidate_folder) == False:
		os.makedirs(candidate_folder)
		print("Creating candidate folder")
		
	if os.path.isdir(debug_folder) == False:
		os.makedirs(debug_folder)
		print("Creating debug folder")
		
	if os.path.isdir(dict_folder) == False:
		os.makedirs(dict_folder)
		print("Creating dict folder")
		
	if os.path.isdir(pos_training_folder) == False:
		os.makedirs(pos_training_folder)
		print("Creating POS training folder")
		
	if os.path.isdir(pos_testing_folder) == False:
		os.makedirs(pos_testing_folder)
		print("Creating POS testing folder")
			
	return
#---------------------------------------------------------------------------------------------#
#INPUT: List of data files -------------------------------------------------------------------#
#OUTPUT: None --------------------------------------------------------------------------------#
#For each data file, make sure does not exist before writing current data files --------------#

def check_data_files(data_file):
	
	if os.path.isfile(data_file):
		os.remove(data_file)
			
	return
#---------------------------------------------------------------------------------------------#
#INPUT: Raw text files to annotate and parameters; may or may not contain meta-data ----------#
#---Meta-data format: Field:Value,Field:Value\tText ------------------------------------------#
#OUTPUT: Write tabbed CoNLL output for main script, return list of files and maybe metadata --#

def annotate_files(input_file, 
					Parameters, 
					Grammar = "", 
					metadata = False, 
					same_size = False, 
					run_parameter = 0
					):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		#Run parameter keeps pool workers out for this imported module#
		run_parameter = 1
	
		import os
		input_file = os.path.join(Parameters.Input_Folder, input_file)

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
				
		import os
		import codecs
		import platform
		from process_input.rdrpos_tagger.Utility.Utils import readDictionary
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
			os.chdir(os.path.join("..", "..", ".."))
		#End directory check#
			
		model_string = os.path.join(".", "files_data", "pos_rdr", Parameters.Language + ".RDR")
		dict_string = os.path.join(".", "files_data", "pos_rdr", Parameters.Language + ".DICT")
		
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
#---------------------------------------------------------------------------------------------#
def write_debug(category_frequency, 
				lemma_frequency, 
				word_frequency, 
				pos_frequency, 
				debug_file, 
				encoding_type
				):

	with open(debug_file + "Category", "w", encoding=encoding_type) as fdebug:
		for key in category_frequency.keys():
			fdebug.write(str(key))
			fdebug.write(": ")
			fdebug.write(str(category_frequency[key]))
			fdebug.write("\n")
	
	with open(debug_file + "Lemma", "w", encoding=encoding_type) as fdebug:
		for key in lemma_frequency.keys():
			fdebug.write(str(key))
			fdebug.write(": ")
			fdebug.write(str(lemma_frequency[key]))
			fdebug.write("\n")
	
	with open(debug_file + "Word", "w", encoding=encoding_type) as fdebug:
		for key in word_frequency.keys():
			fdebug.write(str(key))
			fdebug.write(": ")
			fdebug.write(str(word_frequency[key]))
			fdebug.write("\n")
	
	with open(debug_file + "POS", "w", encoding=encoding_type) as fdebug:
		for key in pos_frequency.keys():
			fdebug.write(str(key))
			fdebug.write(": ")
			fdebug.write(str(pos_frequency[key]))
			fdebug.write("\n")
	
	return
#---------------------------------------------------------------------------------------------#
#INPUT: List of line dictionaries, input filename (str), encoding_type and docs_per_file------#
#OUTPUT: Write CoNLL file to disk and return list of written filenames -----------------------#

def write_conll_raw(input_file, Parameters):

	print("\tConverting raw text to untagged CoNLL format for " + str(input_file))
	output_file = get_temp_filename(input_file, ".conll_raw")
	
	#Define punctuation#
	punc_list = [
	'"', ']', '[', ',', ')', '(', '>', '#', '_', '•',
	'<', '&', '. ', "' ", '/ ', '“', "'", '-', '...',
	'’ ', '^', '\0', '*', ': ', '@ ', '­', '..',
	"# ", '+', '=', '~', '?', '!', '` ', '%', ':',
	'”', '\n', '` ', '…', '·', ';', '.', '\\'
	]
	
	counter = 0
	doc_counter = 1
	
	output_files = []
	
	with codecs.open(input_file, "rb") as fo:
		fw = codecs.open(output_file + "." + str(doc_counter), "w", encoding = Parameters.Encoding_Type)
		
		for line in fo:
		
			line = line.decode(encoding = Parameters.Encoding_Type, errors = "replace")
			line = line.replace("\n","").replace("\r","")
			
			#First, check if new file needs to be opened#
			if counter >= Parameters.Lines_Per_File:
				
				output_files += [output_file + "." + str(doc_counter)]
				doc_counter += 1
				counter = 0
				fw.close()
				fw = codecs.open(output_file + "." + str(doc_counter), "w", encoding = Parameters.Encoding_Type) 
			#Done checking file size#
			
			if line != None:
				
				line = replace_emojis(line, Parameters)
				line = tokenize_line(line)
				
				if line != None:
				
					line_list = line.split(" ")
					
					if len(line_list) > 1:
				
						counter += 1
						id_string = "<s:" + str(counter) + ">"
						fw.write(id_string + "\t" + id_string + "\t" + id_string + "\t" + id_string + "\t" + id_string + "\n")
							
						for unit in line_list:
						
							if unit != "" and unit != None:

								if unit[0:5] == "EMOJI" and unit[len(unit) - 5:] == "EMOJI":
									current_word = unit[5:len(unit)-5]
									current_word = "{" + current_word + "}"
								
								elif "http" in unit:
									current_word = ""
									
								elif unit in punc_list:
									current_word = ""
								
								else:
									current_word = unit
						
							if current_word != "" and current_word != None:
								
								fw.write(str(current_word.replace("\0", "NULL")))
								fw.write("\t")
								
								fw.write(str(current_word).replace("\0", "NULL").lower())
								fw.write("\t")
								
								fw.write("A\t")
								fw.write("B\t")
								fw.write("C\n")

	fw.close()
	
	#Delete last, unfinished file#
	check_data_files(output_file + "." + str(doc_counter))
	
	return output_files
#---------------------------------------------------------------------------------------------#
#INPUT: List of line dictionaries, input filename (str), encoding_type and docs_per_file------#
#OUTPUT: Write CoNLL file to disk and return list of written filenames -----------------------#

def write_conll(text_dictionary, 
					input_file, 
					encoding_type
					):

	doc_counter = text_dictionary[0]
	text_dictionary = text_dictionary[1]
	
	base_filename = get_temp_filename(input_file, "")
	actual_filename = base_filename + "." + str(doc_counter) + ".conll"
	
	fw = codecs.open(actual_filename, "w", encoding = encoding_type)
	
	for line_tuple in text_dictionary:
	
		try:
			
			id = line_tuple[0]
			line = line_tuple[1]
		
			for unit in line:
					
				if unit == "<s>":
					fw.write("<s:")
					fw.write(str(id))
					fw.write(">\n")
				
				else:
				
					skip_flag = 0
					
					current_index = unit['index']
					current_word = unit['word']
					current_lemma = current_word
					current_pos = unit['pos']
					
					if current_word != "":
				
						if current_word[0] == "#":
							skip_flag = 1

						if current_word[0] == "@":
							skip_flag = 1
					
						if current_word[0:5] == "EMOJI" and current_word[len(current_word) - 5:] == "EMOJI":
							skip_flag = 1
					
						if "http" in current_word:
							skip_flag = 1
							
						if ":\\" in current_word:
							skip_flag = 1
						
						if skip_flag == 0:
							fw.write(str(current_word) + "\t")
							fw.write(str(current_lemma.lower()) + "\t")
							fw.write(str(current_pos) + "\t")
							fw.write(str(current_index) + "\t")
							fw.write(str("\n"))	
		
		except:
			null_counter = 0
			
	fw.close()
	
	return actual_filename
#---------------------------------------------------------------------------------------------#
#Take string, return a tokenized version -----------------------------------------------------#

def tokenize_line(line):
	
	line = line.replace('"',' " ').replace(']',' ] ').replace('[',' [ ').replace(',',' , ').replace('--',' -- ')
	line = line.replace(')',' ) ').replace('(',' ( ').replace('>',' > ').replace('<',' < ')
	line = line.replace('&',' & ').replace('. ',' . ').replace("' "," ' ").replace('/',' / ')
	line = line.replace('“',' “ ').replace('’ ',' ’ ').replace('^',' ^ ').replace('\0','')
	line = line.replace('*',' * ').replace(': ',' : ').replace('@ ', '@').replace("# ","#")
	line = line.replace('+',' + ').replace('=',' = ').replace('~',' ~ ').replace('?',' ? ')
	line = line.replace('!',' ! ').replace('` ',' ` ').replace('”',' ” ').replace('\n','')
	line = line.replace('` ',' ` ').replace('…',' … ').replace('·',' · ').replace(';',' ; ')
	line = line.replace('\r', '').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace("... ", " ... ").replace("...", " ... ")
	
	try:
		if line[0] == " ":
			line = line[1:]
	
		return line
	
	except:
		return
#---------------------------------------------------------------------------------------------#
#INPUT: Input file, encoding type ------------------------------------------------------------#
#OUTPUT: (ID, Meta-Data) tuples --------------------------------------------------------------#

def strip_metadata(line):

	line_list = line.split("\t")
		
	#Check to make sure the text is not split into multiple parts, recombine#
	if len(line_list) > 2:
		for i in range(2, len(line_list)):
			line_list[2] += (str(" ") + str(line_list[i]))
				
	current_text = line_list[1]

	return current_text
#---------------------------------------------------------------------------------------------#
#Take string, return with emojis labelled ----------------------------------------------------#

def replace_emojis(line, Parameters):
	
	emoji_list = Parameters.Emoji_Dictionary['emoji_list']
	
	for emoji in emoji_list:
		
		if emoji in line:
			line = line.replace(emoji, " " + Parameters.Emoji_Dictionary[emoji] + " ")
	
	return line
#---------------------------------------------------------------------------------------------#
#INPUT: Line tuple (ID, Str), encoding, rdr object, dict file, language name -----------------#
#OUTPUT: Annotated data for writing to CoNLL file --------------------------------------------#

def rdrpos_run(line_tuple, r, DICT, Parameters, Idiom_List):
		
	counter = 0
	
	id = line_tuple[0]
	line = line_tuple[1]
	
	try:
		if len(line) > 1:

			if Idiom_List != []:
				for idiom in Idiom_List:
					line = line.replace(" " + idiom[0] + " ", " " + idiom[1] + " ")
			
			line_annotated = r.tagRawSentence(DICT, line)
			line_annotated = line_annotated.split()
	
			#Now prepare line for adding to line list#
			current_line = ["<s>"]
			
			for unit in line_annotated:
				
				counter += 1
				
				divider_index = unit.rfind("/")
				temp_word = unit[:divider_index]
				temp_pos = unit[divider_index+1:].lower()
				
				current_unit = {}
				
				if len(temp_word) > 0:
					current_unit['word'] = temp_word
					current_unit['pos'] = temp_pos
					current_unit['index'] = counter
					current_line.append(current_unit)	

			return (id, current_line)
		
		else:
			return
			
	except:
		return
#---------------------------------------------------------------------------------------------#
#INPUT: Line tuple (ID, Str) and Emoji dictionary --------------------------------------------#
#OUTPUT: Tokenized line tuple with emojis identified and labelled ----------------------------#

def process_lines(line_tuple, Parameters, Grammar):
	
	id = line_tuple[0]
	line = line_tuple[1]
	
	if line and line != "" and line != None:
	
		line = line.replace("\r", "").replace("\n","")
	
		current_line = replace_emojis(line, Parameters)
		current_line = tokenize_line(current_line)
		
		# #Only find MWEs if the MWE Grammar has been passed#
		# if Grammar != None:
			# print(Grammar.MWE_List)
			# sys.kill()
			# if Grammar.Type == "MWE":
				# current_line = find_mwes(current_line, Grammar)

		return (id, current_line)
		
	else:
		
		return
#---------------------------------------------------------------------------------------------#
#INPUT: Line of text from Malt-Parser formatted corpus and semantic category dictionary ------#
#OUTPUT: Dictionary of representations for current word --------------------------------------#
#Take line, containing multiple representations of a single word, return dictionary ----------#

def process_line_ingest(line, semantic_category_dictionary):

	semantic_category = ct.get(line[0].lower(), semantic_category_dictionary, default="n/a")
	line.insert(4, semantic_category.lower())
		
	return line
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def process_create_unit_index(Parameters, 
								Grammar, 
								input_files = None, 
								run_parameter = 0
								):
							
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
#---------------------------------------------------------------------------------------------#
#INPUT: List of files containing formatted corpus, file name to write, max sentences ---------#
#OUTPUT: List of sentences contained in input files ------------------------------------------#
#Open files, loop through lines, send lines to other functions, write list of sentences ------#

def pandas_open(file, 
				Parameters, 
				Grammar, 
				save_words = False, 
				write_output = True, 
				delete_temp = False
				):

	begin = time.time()
	
	temp_dataframe = pd.read_csv(file, 
									sep="\t", 
									engine="c", 
									header=None, 
									names=['Word', "Lex", 'Pos', 'Ind', 'Hea', 'Rol'], 
									encoding=Parameters.Encoding_Type,
									quotechar="\t",							
									error_bad_lines=False, 
									skip_blank_lines=True
									)
									
	temp_dataframe = temp_dataframe.loc[:,['Ind', 'Word', "Lex", 'Pos']]	

	#print("Adding semantic category labels")
	apply_get = partial(ct.get, seq=Grammar.Semantic_Category_Dictionary, default="n/a")
	temp_dataframe.loc[:,'Cat'] = temp_dataframe.loc[:,'Word'].str.lower().apply(apply_get)
	
	#print("Indexing Lemma")
	apply_get = partial(ct.get, seq=Grammar.Lemma_Dictionary, default=0)
	temp_dataframe.loc[:,"Lex"] = temp_dataframe.loc[:,"Lex"].str.lower().apply(apply_get)
	
	#print("Indexing POS")
	apply_get = partial(ct.get, seq=Grammar.POS_Dictionary, default=0)
	temp_dataframe.loc[:,'Pos'] = temp_dataframe.loc[:,'Pos'].str.lower().apply(apply_get)
	
	#print("Indexing Category")
	apply_get = partial(ct.get, seq=Grammar.Category_Dictionary, default=0)
	temp_dataframe.loc[:,'Cat'] = temp_dataframe.loc[:,'Cat'].str.lower().apply(apply_get)
	
	#print("Adding Sentence IDs and removing Sentence markers")
	sentence_list = []
	
	for row in temp_dataframe.itertuples(index=False):

		if pd.notnull(row[1]):
		
			temp_id = row[1]
			
			if "<s:" in temp_id:
				current_id = temp_id.replace("<s:", "").replace(">", "")
				current_id = int(current_id)

		sentence_list.append(current_id)		
	
	temp_dataframe.loc[:,'Sent'] = pd.Series(sentence_list, index=None)
	
	temp_dataframe['Word'] = temp_dataframe['Word'].astype(str)
	temp_dataframe = temp_dataframe[~temp_dataframe['Word'].str.contains("<")]
	
	#print("Removing Word-Forms and Adding Master Index")
	
	sLength = len(temp_dataframe['Ind'])
	temp_dataframe.loc[:,'Mas'] = pd.Series(range(0, sLength), index=temp_dataframe.index)
	
	if save_words == True:
		temp_dataframe = temp_dataframe.loc[:,['Sent', 'Ind', 'Word', "Lex", 'Pos', 'Cat', 'Mas']]
		
	else:
		temp_dataframe = temp_dataframe.loc[:,['Sent', 'Ind', "Lex", 'Pos', 'Cat', 'Mas']]
	
	end = time.time()
	#print("\t\t\t\tIngest Time: " + str(end - begin))
	
	if delete_temp == True:
		import os
		os.remove(file)
	
	if write_output == True:
		
		from process_input.check_data_files import check_data_files
		from process_input.get_temp_filename import get_temp_filename
		
		output_file = get_temp_filename(file, ".Pandas")
		check_data_files(output_file)

		temp_dataframe.to_hdf(output_file, "Table", format='table', complevel=9, complib="blosc")

		return output_file
	
	else:

		return temp_dataframe
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def split_list(seq, num):
  
	avg = len(seq) / float(num)
	out = []
	last = 0.0

	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg

	return out
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def merge_conll_names(training_testing_files, testing_files, Parameters):
	
	tuple_list = []
	training_list = []
	testing_list = []
	counter = 0

	for file_list in training_testing_files:
		
		counter += 1
		current_filename = os.path.join(Parameters.Temp_Folder, Parameters.Nickname + ".Restart." + str(counter) + ".conll")
		tuple_list.append((file_list, current_filename))
		training_list.append(current_filename)
				
	current_filename = os.path.join(Parameters.Temp_Folder, Parameters.Nickname + ".Testing.conll")
	tuple_list.append((testing_files, current_filename))
	testing_list.append(current_filename)

	return tuple_list, training_list, testing_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def merge_conll(file_tuple, encoding_type):

	counter = 0
	
	file_list = file_tuple[0]
	output_name = file_tuple[1]
	
	with codecs.open(output_name, "w", encoding = encoding_type) as fw:
	
		for file in file_list:
		
			with codecs.open(file, "r", encoding = encoding_type) as fo:
			
				for line in fo:
				
					if line[0] == "<":
						counter += 1
						fw.write("<s:" + str(counter) + ">\n")
						
					else:
						fw.write(line)
			
	return
#---------------------------------------------------------------------------------------------#
#INPUT: Input file and parameters ------------------------------------------------------------#
#OUTPUT: List of lines -----------------------------------------------------------------------#

def load_utf8(input_file, Parameters):
	line_list = []
	
	with open(input_file, "rb") as fo:
		counter = 0
		
		for line in fo:
			counter += 1
			line = line.decode(Parameters.Encoding_Type, errors="replace")
				
			if Parameters.Use_Metadata == True:
				line = strip_metadata(line)
					
			line_list.append((counter, line))
			
	return line_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def get_write_list(text_dictionary, docs_per_file):

	number_of_files = int(len(text_dictionary) / docs_per_file) + 1
		
	write_list = []
	start_index = 0
	end_index = docs_per_file 
		
	if end_index > len(text_dictionary):
		end_index = len(text_dictionary) 
			
	for i in range(1,number_of_files+1):
		temp_tuple = (i, start_index, end_index)
		write_list.append(temp_tuple)
		start_index += docs_per_file
		end_index += docs_per_file
		
		if end_index > len(text_dictionary):
			end_index = len(text_dictionary) 
			
	temp_list = []
	
	for segment in write_list:
	
		doc_id = segment[0]
		start_index = segment[1]
		end_index = segment[2]
		
		temp_dictionary = text_dictionary[start_index:end_index]
		temp_tuple = (doc_id, temp_dictionary)
		temp_list.append(temp_tuple)
			
	return temp_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def get_temp_filename(input_file, 
						suffix,
						candidate_flag = False
						):

	input_list = input_file.split("/")

	if len(input_list) == 1:
		input_list = input_file.split("\\")


	base_input_file = input_list[-1]
	input_file_path = input_list[0:len(input_list) - 1]
	input_file_path = os.path.join(*input_file_path)

	if candidate_flag == False:
		if "Temp" in input_list:
			output_file = input_file + suffix
		
		else:
			output_file = str(os.path.join(input_file_path, "Temp", base_input_file) + suffix)
			
	elif candidate_flag == True:
		if "Candidates" in input_list:
			output_file = os.path.join(input_file, suffix)
		
		else:
			output_file = str(os.path.join(input_file_path, "Candidates", base_input_file) + suffix)
		
	return output_file
#---------------------------------------------------------------------------------------------#
#INPUT: Input file, encoding type ------------------------------------------------------------#
#OUTPUT: (ID, Meta-Data) tuples --------------------------------------------------------------#

def get_metadata_tuples(input_file, Parameters):
	
	with open(input_file, "rb") as fo:
	
		counter = 0
		metadata_list = []
		
		for line in fo:
			
			counter += 1
			line = line.decode(Parameters.Encoding_Type, errors="replace")
			line_list = line.split("\t")
			
			#Check to make sure the text is not split into multiple parts, recombine#
			if len(line_list) > 2:
				for i in range(2, len(line_list)):
					line_list[2] += (str(" ") + str(line_list[i]))
					
			current_metadata = line_list[0]
			current_text = line_list[1]
			current_id = counter
		
			#Meta-Data format: Field:Value,Field:Value,Field:Value #
			current_metadata_list = current_metadata.split(",")
			current_meta_dictionary = {}
			
			for field in current_metadata_list:
			
				current_temp = field.split(":")
				current_field = current_temp[0]
				current_value = current_temp[1]
				
				current_meta_dictionary[current_field] = current_value
				
			current_tuple = (current_id, current_meta_dictionary)
			metadata_list.append(current_tuple)			
			
	return metadata_list
#---------------------------------------------------------------------------------------------#
#INPUT: List of dictionaries with unit frequencies--------------------------------------------#
#OUTPUT: List of lists of allowed units-------------------------------------------------------#
#Take dictionary of elements and return index lists ------------------------------------------#

def get_index_lists(full_dictionary):

	new_dictionary = {}
	
	#First, separate label dictionaries with frequency info#
	lemma_dictionary = full_dictionary['lemma']
	pos_dictionary = full_dictionary['pos']
	word_dictionary = full_dictionary['word']
	category_dictionary = full_dictionary['category']	
		
	#Second, create lists of label indexes#
	word_list = sorted(word_dictionary.keys())
	lemma_list = sorted(lemma_dictionary.keys())
	pos_list = sorted(pos_dictionary.keys())
	category_list = sorted(category_dictionary.keys())	
	
	word_list.insert(0, "n/a")	
	lemma_list.insert(0, "n/a")
	pos_list.insert(0, "n/a")
	
	try:
		temp_index = category_list.index("n/a")
		del category_list[temp_index]
		
	except:
		null_counter = 0
		
	category_list.insert(0, "n/a")
	
	word_frequency = word_dictionary
	lemma_frequency = lemma_dictionary
	pos_frequency = pos_dictionary
	category_frequency = category_dictionary
		
	#Fifth, append items onto a single list for returning and writing#	
	new_dictionary['lemma_list'] = lemma_list
	new_dictionary['pos_list'] = pos_list
	new_dictionary['word_list'] = word_list
	new_dictionary['category_list'] = category_list
	
	new_dictionary['lemma_frequency'] = lemma_frequency
	new_dictionary['pos_frequency'] = pos_frequency
	new_dictionary['word_frequency'] = word_frequency
	new_dictionary['category_frequency'] = category_frequency
	
	lemma_dictionary = {}
	pos_dictionary = {}
	category_dictionary = {}
	
	for i in range(len(lemma_list)):
		lemma_dictionary[lemma_list[i]] = i
		
	for i in range(len(pos_list)):
		pos_dictionary[pos_list[i]] = i
		
	for i in range(len(category_list)):
		category_dictionary[category_list[i]] = i
		
	new_dictionary['lemma_dictionary'] = lemma_dictionary
	new_dictionary['pos_dictionary'] = pos_dictionary
	new_dictionary['category_dictionary'] = category_dictionary
			
	return new_dictionary
#---------------------------------------------------------------------------------------------#
#INPUT: DataFrame, index lists ---------------------------------------------------------------#
#OUTPUT: Dictionary of frequency dictionaries ------------------------------------------------#
#Count individual items in unexpanded DataFrame ----------------------------------------------#

def get_frequencies(current_df, Grammar):

	start = time.time()
	
	lemma_frequency = {}
	pos_frequency = {}
	category_frequency = {}
	
	count = current_df.loc[:,"Lex"].value_counts(sort=True, ascending=True, dropna=True)
	for row in count.iteritems():
		unit = row[0]
		label = Grammar.Lemma_List[unit]
		count = row[1]
		lemma_frequency[label] = count
		
	count = current_df.loc[:,'Pos'].value_counts(sort=True, ascending=True, dropna=True)
	for row in count.iteritems():
		unit = row[0]
		label = Grammar.POS_List[unit]
		count = row[1]
		pos_frequency[label] = count
		
	count = current_df.loc[:,'Cat'].value_counts(sort=True, ascending=True, dropna=True)
	for row in count.iteritems():
		unit = row[0]
		label = Grammar.Category_List[unit]
		count = row[1]
		category_frequency[label] = count
		
	end = time.time()
	#print("Create frequency dictionaries: " + str(end-start))
	
	current_df = current_df.query("(Pos != 0)")
	number_of_words = len(current_df)
			
	return lemma_frequency, pos_frequency, category_frequency, number_of_words
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def fold_split(Parameters, input_files):

	fold_file_dict = {}
	
	training_candidates = int(Parameters.Training_Candidates / Parameters.Lines_Per_File)
	
	if training_candidates < 1: 
		training_candidates = 1
		
	training_search = int(Parameters.Training_Search / Parameters.Lines_Per_File)
	
	if training_search < 1:
		training_search = 1
		
	testing = int(Parameters.Testing / Parameters.Lines_Per_File)
	
	if testing < 1:
		testing = 1
	
	#Distribute files for each fold#
	for fold in range(1,Parameters.CV +1):
	
		fold_file_dict[fold] = {}
	
		#First, randomly select training candidate files#
		if len(input_files) > training_candidates:
			current_training_candidates_files = random.sample(input_files, training_candidates)
			input_files = [x for x in input_files if x not in current_training_candidates_files]
			
		else:
			print("Not enough data to fill desired data distribution.")
			sys.kill()
		
		#Second, randomly select testing files#
		if len(input_files) > testing:
			current_testing_files = random.sample(input_files, testing)
			input_files = [x for x in input_files if x not in current_testing_files]
	
		else:
			print("Not enough data to fill desired data distribution.")
			sys.kill()
			
		#Third, randomly select training search files, looping through#
		current_training_search_files = []
		
		for i in range(1, Parameters.Restarts+1):
			
			if len(input_files) > training_search:
				temp_training_search_files = random.sample(input_files, training_search)
				input_files = [x for x in input_files if x not in temp_training_search_files]
				current_training_search_files.append(temp_training_search_files)
			
			else:
				print("Not enough data to fill desired data distribution.")
				sys.kill()
		
		#Save current fold#
		fold_file_dict[fold]["Training_Candidates"] = current_training_candidates_files
		fold_file_dict[fold]["Training_Search"] = current_training_search_files
		fold_file_dict[fold]["Testing"] = current_testing_files
		
	print("\tInstances remaining after distributing unique sets across folds and restarts: " + str(len(input_files) * Parameters.Lines_Per_File))
	
	return fold_file_dict
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def find_mwes(current_line, Grammar):

	current_line = current_line.lower()
	print(Grammar.MWE_List)

	for mwe in Grammar.MWE_List:
	
		try:
			current_line.replace(mwe, mwe.replace(" ", "_", 1))
			print(current_line)
		except:
			counter = None

	return current_line
#---------------------------------------------------------------------------------------------#
#INPUT: List of files containing formatted corpus, max_sentences, unit frequency threshold ---#
#OUTPUT: Lists of frequency reduced unit labels ----------------------------------------------#
#Open files, loop through lines, send lines to other functions, write list of sentences ------#

def create_unit_index(input_files, Parameters):
	
	sentence_list = []
	sentence_counter = 0
	word_counter = 0
	
	pos_dictionary = {}
	lemma_dictionary = {}
	word_dictionary = {}
	category_dictionary = {}

	#Loop through input files#
	for file in input_files:

		print("\tOpening ", end="")
		print(file)
		
		fo = open(file, "r", newline="", encoding = Parameters.Encoding_Type)
		ingest_file = csv.reader(fo, delimiter="\t", quoting=csv.QUOTE_NONE)
		
		# #Loop through lines in current file#
		for line in ingest_file:

			if len(line) < 5:
				if len(line) == 1:
					if line[0] == "<s>":
				
						sentence_counter += 1
						if sentence_counter % 10000 == 0:
					
							print("Processing sentence number ", end="")
							print(str(sentence_counter))
					
			else:

				#Get line information as a list#
				word_counter += 1
				line = process_line_ingest(line, Parameters.Semantic_Category_Dictionary)
				#0:Word, 1:Lemma, 2:POS, 3: Index#, 4: Category#

				#Add word information#
				if line[2] not in Parameters.Illegal_POS and line[0] != "|":
					try:
						word_dictionary[line[0].lower()] += 1
					except: 
						word_dictionary[line[0].lower()] = 1
				
				#Add lemma information#
				if line[2] not in Parameters.Illegal_POS and line[1] != "|":
					try:
						lemma_dictionary[line[1].lower()] += 1
					except: 
						lemma_dictionary[line[1].lower()] = 1
				
				#Add pos information#
				if line[2] not in Parameters.Illegal_POS:
					try:
						pos_dictionary[line[2].lower()] += 1
					except:
						pos_dictionary[line[2].lower()] = 1
				
				#Add category information#
				if line[2] not in Parameters.Illegal_POS:
					try:
						category_dictionary[line[4].lower()] += 1
					except:
						category_dictionary[line[4].lower()] = 1
		
		#End loop through lines in current file#
		fo.close()
		
	#End loop through input files#
	
	full_dictionary = {}
	full_dictionary['lemma'] = lemma_dictionary
	full_dictionary['pos'] = pos_dictionary
	full_dictionary['word'] = word_dictionary
	full_dictionary['category'] = category_dictionary
	
	return full_dictionary
#---------------------------------------------------------------------------------------------#