#General modules
import time
import pickle
import os
import os.path
import pandas as pd
import cytoolz as ct
import itertools

#INPUT: Name for candidate file  and list of candidates---------------------------------------#
#OUTPUT: None: Write file with candidates ----------------------------------------------------#

def write_candidates(file, candidate_list):
    
	if os.path.isfile(file):
		os.remove(file)
	
	with open(file,'wb') as f:
		pickle.dump(candidate_list,f)
	
	return
#---------------------------------------------------------------------------------------------#
#OUTPUT: Candidates from all files for current template that are above frequency threshold ---#
#Process single template and return first-cut candidates--------------------------------------#

def templates_to_candidates(current_df, 
								filename, 
								sequence_list, 
								annotation_types, 
								max_construction_length, 
								frequency_threshold_constructions_perfile
								):

	#print("Beginning candidate extraction.")
	start_all = time.time()

	candidate_dictionary = {}
	match_counter = 0
	
	#print("Opened: " + filename + ", Length: " + str(len(current_df)))
	
	#Begin loop through templates#
	for template in sequence_list:
	
		start_file = time.time()
		
		copy_df = current_df.copy("Deep")
		
		template = list(template)
		template_name = get_template_name(template)
		
		candidate_dictionary[str(template_name)] = find_template_matches(copy_df, template_name, frequency_threshold_constructions_perfile)
		
		match_counter += len(candidate_dictionary[str(template_name)])
		print("Total: " + str(match_counter) + filename + ": " + str(template_name) + ": "  + str(time.time() - start_file))
			
	#End loop through templates#
	
	end_all = time.time()
	#print("File Time: " + filename + ": " + str(end_all - start_all))
	#print("")
	
	#print("Total candidates: " + str(match_counter))

	return candidate_dictionary
#---------------------------------------------------------------------------------------------#
#INPUT: Current template and data_file_candidate_constructions -------------------------------#
#OUTPUT: Filename for storing candidates from current template -------------------------------#

def remove_infrequent_candidates(candidate_list, data_file_candidate_constructions):
    
	for i in range(len(candidate_list)):
		if candidate_dictionary[i] > frequency_threshold_constructions:
			
			temp_list = candidate_list[i]
			final_candidate_list.append(temp_list)
			
	print("")
	print("For template: " + str(template) + ", All candidates: " + str(len(candidate_list)))
	print("For template: " + str(template) + ", Reduced candidates: " + str(len(final_candidate_list)))
		
	return pickled_list_file
#---------------------------------------------------------------------------------------------#
#INPUT: Name for candidate H5 file -----------------------------------------------------------#
#OUTPUT: List of candidates ------------------------------------------------------------------#

def read_candidates(file):
    
	candidate_list = []
	
	with open(file,'rb') as f:
		candidate_list = pickle.load(f)
		
	return candidate_list
#---------------------------------------------------------------------------------------------#
#INPUT: Full candidate list ------------------------------------------------------------------#
#OUTPUT: None: just print how many candidates total ------------------------------------------#

def print_full_candidate_info(full_candidate_list):

	total_count = 0
	
	for item in full_candidate_list:
	
		template = item[0]
		dictionary = item[1]
		
		current_count = len(dictionary.keys())
		total_count += current_count
		
	print("Total number of frequency reduced candidates: " + str(total_count))
	
	return
#---------------------------------------------------------------------------------------------#
#INPUT: Template with numbers for column labels ----------------------------------------------#
#OUTPUT: List of template units with strings as column labels --------------------------------#

def merge_file_results(data_file_candidate_constructions, 
						data_files_expanded, sequence_list, 
						frequency_threshold_constructions
						):

	full_candidate_list = []
	
	for i in range(len(sequence_list)):
		
		start_file = time.time()
		full_candidate_dictionary = {}
		
		for file in data_files_expanded:
		
			pickled_list_file = get_candidate_name(data_files_expanded, data_file_candidate_constructions)
			pickled_list_file = pickled_list_file.replace("']","")
			current_file_list = read_candidates(pickled_list_file)
			
			current_template = current_file_list[i][0]
			current_file_candidate_dictionary = current_file_list[i][1]
		
			#Merge current candidates with candidates from previous files, summing individual counts#
			full_candidate_dictionary = ct.merge_with(sum, full_candidate_dictionary, current_file_candidate_dictionary)
			
		#Done with all files for this template, now remove infrequent candidates#
		print("Finished candidate merging. Now removing infrequent candidates.")
		above_threshold = lambda x: x > frequency_threshold_constructions
		full_candidate_dictionary = ct.valfilter(above_threshold, full_candidate_dictionary)
		
		#Now add frequency reduced template candidates to full list#
		current_addition = [current_template, full_candidate_dictionary]
		full_candidate_list.append(current_addition)
				
		end_file = time.time()
		print(str(current_template) + ": "  + str(end_file - start_file))
		print("")

	return full_candidate_list
#---------------------------------------------------------------------------------------------#
#INPUT: Template with numbers for column labels ----------------------------------------------#
#OUTPUT: List of template units with strings as column labels --------------------------------#

def merge_candidates(data_file_candidate_constructions, sequence_list):
		
	#"Lemma" = 3, "Pos" = 4, "Category" = 5#
	
	full_candidate_list = []
	
	for original_template in sequence_list:
	
		template_name = []
		
		template = list(original_template)
		template = [x+3 for x in template]
	
		for i in range(len(template)):
			current_unit = template[i]
		
			template_name.append(current_unit)
					
		pickled_list_file = get_candidate_name(template_name, data_file_candidate_constructions)
		
		current_candidate_list = read_candidates(pickled_list_file)
		
		full_candidate_list.append(current_candidate_list)
	
	return full_candidate_list
#---------------------------------------------------------------------------------------------#
#INPUT: Template with numbers for column labels ----------------------------------------------#
#OUTPUT: List of template units with strings as column labels --------------------------------#

def get_template_name(template):
	
	#Template names come from annotaton types variable in parameters and are column names#
	
	template_name = []
	
	for i in range(len(template)):
		current_unit = template[i]
		
		template_name.append(current_unit)
	
	return template_name
#---------------------------------------------------------------------------------------------#
#INPUT: List of columns to check for equivalence ---------------------------------------------#
#OUTPUT: String of query ---------------------------------------------------------------------#

def get_query_zero(column_names):
		
	query = ""
	
	for i in range(len(column_names)):
		
		if i == 0:
			query = "(c" + str(i+1) + " != 0 "
			
		elif i  > 0:
			query += " and " + "c" + str(i+1) + " != 0"
	
	query += ")"
	
	return query
#---------------------------------------------------------------------------------------------#
#INPUT: List of columns to check for equivalence ---------------------------------------------#
#OUTPUT: String of query ---------------------------------------------------------------------#

def get_query(column_names):
		
	query = ""
	
	for i in range(len(column_names)):
		
		if i == 0:
			query = "(" + str(column_names[i]) + " == "
			
		elif i != len(column_names) - 1:
			query += str(column_names[i]) + " == "
		
		elif i == len(column_names) - 1:
			query += str(column_names[i]) + ")"
		
	return query
#---------------------------------------------------------------------------------------------#
#INPUT: template -----------------------------------------------------------------------------#
#OUTPUT: Take template, return list of lists of columns to be shifted ------------------------#

def get_column_shift(template):
	
	column_shifts = []
	
	for i in range(len(template) + 2):
	
		index1 = i
		
		if index1 > 2:
			column_shifts.append(["c" + str(index1)])

	return column_shifts
#---------------------------------------------------------------------------------------------#
#INPUT: template -----------------------------------------------------------------------------#
#OUTPUT: Take template, return list of columns to include in candidate search DataFrame ---------#

def get_column_list(template):
	
	column_list = [0, template[0]]
	column_names = ["c1", "c2"]
	column_counter = 2
	
	for i in range(len(template) - 1):
		column_list.append(template[i + 1])
		
		column_names.append("c" + str(column_counter + 1))
		
		column_counter += 1

	return [column_list, column_names]
#---------------------------------------------------------------------------------------------#
#INPUT: Current template and data_file_candidate_constructions -------------------------------#
#OUTPUT: Filename for storing candidates from current template -------------------------------#

def get_candidate_name(file, data_file_candidate_constructions):
    
	file_name = str(file)
	
	begin_name = file_name.rfind("/")
	file_name = file_name[begin_name+1:]
		
	temp_pickled_name = file_name + ".p"
	pickled_list_file = data_file_candidate_constructions + temp_pickled_name
		
	return pickled_list_file
#---------------------------------------------------------------------------------------------#
#INPUT: Reduced and sorted DataFrame, per file candidate frequency threshold -----------------#
#OUTPUT: Take template, return list of rows to include in candidate search DataFrame ---------#

def get_candidate_count(current_df, frequency_threshold_constructions_perfile):
	
	import pandas as pd
	import cytoolz as ct

	tuple_list = [tuple(x) for x in current_df.values]
	pair_frequency = ct.frequencies(tuple_list)
	
	above_threshold = lambda x: x > frequency_threshold_constructions_perfile
	pair_frequency = ct.valfilter(above_threshold, pair_frequency)
		
	return pair_frequency	
#---------------------------------------------------------------------------------------------#
#INPUT: Current template and DataFrame -------------------------------------------------------#
#OUTPUT: DataFrame with matches to template --------------------------------------------------#

def find_template_matches(current_df, 
							template, 
							frequency_threshold_constructions_perfile
							):

	start_all = time.time()
	
	#Create shifted alt-only dataframe for length of template#
	alt_columns = []
	alt_columns_names = []
	for i in range(len(template)):
		alt_columns.append(1)
		alt_columns_names.append("c" + str(i))
	
	alt_dataframe = create_shifted_df(current_df, 1, alt_columns)
	alt_dataframe.columns = alt_columns_names
	query_string = get_query(alt_columns_names)
	row_mask_alt = alt_dataframe.eval(query_string)
	del alt_dataframe

	#Create shifted sent-only dataframe for length of template#
	sent_columns = []
	sent_columns_names = []
	for i in range(len(template)):
		sent_columns.append(0)
		sent_columns_names.append("c" + str(i))
	
	sent_dataframe = create_shifted_df(current_df, 0, sent_columns)
	sent_dataframe.columns = sent_columns_names
	query_string = get_query(sent_columns_names)
	row_mask_sent = sent_dataframe.eval(query_string)
	del sent_dataframe
	
	#Create and shift template-specific dataframe#
	temp_list = get_column_list(template)
	column_list = temp_list[0]
	column_names = temp_list[1]
	current_df = create_shifted_template_df(current_df, column_list)
	current_df.columns = column_names
	
	current_df = current_df.loc[row_mask_sent & row_mask_alt,]
	del row_mask_sent
	del row_mask_alt
	
	#Remove NaNS and change dtypes#
	current_df.fillna(value=0, inplace=True)
	column_list = current_df.columns.values.tolist()
	current_df = current_df[column_list].astype(int)

	#Remove zero valued indexes#
	column_list = current_df.columns.values.tolist()
	query_string = get_query_zero(column_list)
	current_df = current_df.query(query_string, parser='pandas', engine='numexpr')
	
	#Find duplicated rows within same sentence and remove those which are duplicated#
	column_list = current_df.columns.values.tolist()
	row_mask2 = current_df.duplicated(subset=column_list, keep="first")
	current_df = current_df.loc[~row_mask2,]
	del row_mask2
	
	#Find unique rows and remove them#
	current_df = current_df.drop('c1', 1)
	column_list = current_df.columns.values.tolist()
	row_mask1 = current_df.duplicated(subset=column_list, keep=False)
	
	current_df = current_df.loc[row_mask1,]
	del row_mask1
	
	#Sort DataFrame to get similar candidates from same sentences together#
	column_list = current_df.columns.values.tolist()
	current_df = current_df[column_list].astype(int)
	current_df = current_df.sort_values(by=column_list, axis=0, ascending=True, inplace=False, kind="mergesort")
	
	#Count remaining candidates and remove those below frequency threshold#
	candidate_count_dictionary = get_candidate_count(current_df, frequency_threshold_constructions_perfile)
	del current_df
	
	print("\t\t\tNumber found: " + str(len(candidate_count_dictionary)) + ": ", end="")
	
	end_all = time.time()

	return candidate_count_dictionary
#---------------------------------------------------------------------------------------------#
#INPUT: Max construction length, annotation types, and lists of allowable elements for each --#
#OUTPUT: List of templates for possible constructions ----------------------------------------#

def create_templates(annotation_types, max_construction_length):

	sentence_list_candidates = {}
	current_candidates = {}
	sequence_list = []
	counter = 0
	progress_counter = 0
	
	#First, generate all possible templates (e.g., Word-Form, POS, Role, Word-Form)#
	for ngram in range(2,max_construction_length + 1):
		for p in itertools.product(annotation_types, repeat=ngram):
			sequence_list.append(p)
			
	#print("Number of templates: " + str(len(sequence_list)))
			
	return sequence_list
#---------------------------------------------------------------------------------------------#
#OUTPUT: Take a dataframe, the column to repeat, and a listof times to repeat ----------------#
#Specific to creating alt / sent dataframes b/c more efficient than a generalized version ----#

def create_shifted_template_df(original_df, ordered_columns):
	
	ordered_columns[0] = 'Sent'
	column_list = []
	
	for i in range(len(ordered_columns)):
		holder_df = original_df.loc[:,ordered_columns[i]]
		column_list.append(holder_df.shift(-i))
		del holder_df
	
	original_df = pd.concat(column_list, axis=1)
	del column_list
	
	return original_df
#---------------------------------------------------------------------------------------------#
#INPUT: template -----------------------------------------------------------------------------#
#OUTPUT: Take template, return list of rows to include in candidate search DataFrame ---------#

def create_shifted_df(original_df, desired_column, ordered_columns):
	
	holder_df = original_df.iloc[:,desired_column]
	column_dict = {}
	
	for i in range(len(ordered_columns)):	
		column_dict[i] = holder_df.shift(-i)

	original_df = pd.DataFrame(column_dict)	
	del column_dict
	del holder_df

	return original_df
#---------------------------------------------------------------------------------------------#
#INPUT: list of candidates (frequency reduced), list of lemma and pos and category indexes ---#
#--------and file name for debug info --------------------------------------------------------#
#OUTPUT: Write a file with human readable construction candidates for debugging --------------#

def candidate_debug(candidate_list, 
						lemma_list, 
						pos_list, 
						category_list, 
						candidate_debug_file, 
						encoding_type
						):
	
	fw = open(candidate_debug_file, "w", encoding=encoding_type)
		
	for candidate in candidate_list:
		
		for pair in candidate:
			column = pair[0]
			index = pair[1]
				
			if column == "Lex":
				value = lemma_list[index]
					
			elif column == 'Pos':
				value = pos_list[index]
					
			elif column == 'Cat':
				value = category_list[index]
					
			fw.write(str(value))
			fw.write(" ")
		
		#End current construction candidate, add newline#
		fw.write("\n")
	
	return
#---------------------------------------------------------------------------------------------#