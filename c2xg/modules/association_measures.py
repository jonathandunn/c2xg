#General imports
import pandas as pd
import cytoolz as ct
import multiprocessing as mp
from functools import partial

#C2xG imports
from modules.process_input import create_category_dictionary
from modules.candidate_extraction import read_candidates

#-------------------------------------------------------------------------------------------------------------------#
#INPUT: List of candidates and abcd's ------------------------------------------------------------------------------#
#OUTPUT: Dictionary with co-occurrence data (a, b, c) for each key -------------------------------------------------#

def get_dictionary(pairwise_list):

	pairwise_dictionary = {}
	
	for i in range(len(pairwise_list)):
		
		temp_id = pairwise_list[i][0]
		temp_a = pairwise_list[i][1]
		temp_b = pairwise_list[i][2]
		temp_c = pairwise_list[i][3]
		temp_d = pairwise_list[i][4]
		
		pairwise_dictionary[temp_id] = [temp_a, temp_b, temp_c, temp_d]
	
	return pairwise_dictionary
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def get_df_unitwise(vector_list, condition):

	import pandas as pd
	
	vector_df = pd.DataFrame(vector_list, columns=['Candidate', 
													'Beginning_Divided_LR_' + condition, 
													'Beginning_Divided_RL_' + condition, 
													'End_Divided_LR_' + condition, 
													'End_Divided_RL_' + condition
													])
	return vector_df
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def get_df_pairwise(vector_list, condition):

	import pandas as pd

	vector_df = pd.DataFrame(vector_list, columns=['Candidate', 
													'Frequency', 
													'Summed_LR_' + condition,
													'Smallest_LR_' + condition,
													'Summed_RL_' + condition, 
													'Smallest_RL_' + condition,
													'Mean_LR_' + condition, 
													'Mean_RL_' + condition, 
													'Beginning_Reduced_LR_' + condition,
													'Beginning_Reduced_RL_' + condition,
													'End_Reduced_LR_' + condition,
													'End_Reduced_RL_' + condition,
													'Directional_Scalar_' + condition,
													'Directional_Categorical_' + condition,
													'Endpoint_LR_' + condition,
													'Endpoint_RL_' + condition
													])
	
	return vector_df
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def get_candidate_count(candidate_dict):

	total = 0
	
	for key in candidate_dict.keys():
		total += len(list(candidate_dict[key].keys()))
		
	return total
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: Single candidate, expanded data files, number of total units -----------------------------------------------#
#OUTPUT: List of features for current candidate --------------------------------------------------------------------#

def create_pairwise_single(candidate_id, 
							candidate_frequency, 
							pairwise_dictionary,
							freq_weighted
							):

	# Features for Vector:#
		#PAIRWISE#
		# 1. Simple Frequency (Relative in the sense that all candidates are in same corpus)#
		# 2. Summed ΔP, Left-to-Right#
		# 3. Smallest Pairwise LR
		# 4. Summed ΔP, Right-to-Left#
		# 5. Smallest Pairwise RL
		# 6. Normalized (Summed ΔP, Left-to-Right)#
		# 7. Normalized (Summed ΔP, Right-to-Left)#
		# 8. Beginning-Reduced ΔP, Left-to-Right#
		# 9. Beginning-Reduced ΔP, Right-to-Left#
		# 10. End-Reduced ΔP, Left-to-Right#
		# 11. End-Reduced ΔP, Right-to-Left#
		# 12. Directional ΔP#
	
	#First, get A, B, C, D for pair from pairwise_df#
	candidate_str = str(candidate_id)
	candidate_vector = [candidate_str, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	
	try:
		current_pair = ct.get(candidate_str, pairwise_dictionary)
	except:
		current_pair = []
	
		
	if current_pair !=[]:
	
		a = current_pair[0]
		b = current_pair[1]
		c = current_pair[2]
		d = current_pair[3]
			
		co_occurrence_list = [[a, b, c, d]]

		#Second, calculate Delta P's for pairwise measures#
		lr_tuple = calculate_summed_lr(co_occurrence_list, freq_weighted)
		summed_lr = lr_tuple[0]
		smallest_lr = lr_tuple[1]
		
		rl_tuple = calculate_summed_rl(co_occurrence_list, freq_weighted)
		summed_rl = rl_tuple[0]
		smallest_rl = rl_tuple[1]
	
		normalized_summed_lr = calculate_normalized_summed_lr(co_occurrence_list, summed_lr, freq_weighted)
		normalized_summed_rl = calculate_normalized_summed_rl(co_occurrence_list, summed_rl, freq_weighted)
	
		end_reduced_lr = 0
		end_reduced_rl = 0
	
		beginning_reduced_lr = 0
		beginning_reduced_rl = 0
	
		directional_scalar = 0
		directional_categorical = 0
		
		endpoint_lr = summed_lr
		endpoint_rl = summed_rl
	
		#Third, create list of feature values for current candidate, including candidate id#
	
		candidate_vector = [candidate_str, 
						candidate_frequency, 
						summed_lr, 
						smallest_lr,
						summed_rl, 
						smallest_rl,
						normalized_summed_lr, 
						normalized_summed_rl, 
						beginning_reduced_lr,
						beginning_reduced_rl,
						end_reduced_lr,
						end_reduced_rl,
						directional_scalar,
						directional_categorical,
						endpoint_lr,
						endpoint_rl
						]
	
		return candidate_vector
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: Single candidate, expanded data files, number of total units -----------------------------------------------#
#OUTPUT: List of features for current candidate --------------------------------------------------------------------#

def create_pairwise_multiple(candidate_id, 
								candidate_frequency, 
								pairwise_dictionary,
								freq_weighted
								):
	
	full_candidate_str = str(candidate_id)
	candidate_vector = [full_candidate_str, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	co_occurrence_list = []
	
	# Features for Vector:#
		#PAIRWISE#
		# 1. Simple Frequency (Relative in the sense that all candidates are in same corpus)#
		# 2. Summed ΔP, Left-to-Right#
		# 3. Smallest Pairwise LR
		# 4. Summed ΔP, Right-to-Left#
		# 5. Smallest Pairwise RL
		# 6. Normalized (Summed ΔP, Left-to-Right)#
		# 7. Normalized (Summed ΔP, Right-to-Left)#
		# 8. Beginning-Reduced ΔP, Left-to-Right#
		# 9. Beginning-Reduced ΔP, Right-to-Left#
		# 10. End-Reduced ΔP, Left-to-Right#
		# 11. End-Reduced ΔP, Right-to-Left#
		# 12. Directional ΔP#
	
	#First, get A, B, C, D for pair from pairwise_df#
	for i in range(len(candidate_id) - 1):
		
		unit1 = str(candidate_id[i])
		unit2 = str(candidate_id[i+1])
	
		candidate_str = "[" + unit1 + ", " + unit2 + "]"
				
		try:
			current_pair = ct.get(candidate_str, pairwise_dictionary)
		except:
			current_pair = []
			
		if current_pair !=[]:
		
			a = current_pair[0]
			b = current_pair[1]
			c = current_pair[2]
			d = current_pair[3]
			
			co_occurrence_list.append([a, b, c, d])
	
	if len(co_occurrence_list) > 1:

		#Second, calculate Delta P's for pairwise measures#
		lr_tuple = calculate_summed_lr(co_occurrence_list, freq_weighted)
		summed_lr = lr_tuple[0]
		smallest_lr = lr_tuple[1]
		
		rl_tuple = calculate_summed_rl(co_occurrence_list, freq_weighted)
		summed_rl = rl_tuple[0]
		smallest_rl = rl_tuple[1]
	
		normalized_summed_lr = calculate_normalized_summed_lr(co_occurrence_list, summed_lr, freq_weighted)
		normalized_summed_rl = calculate_normalized_summed_rl(co_occurrence_list, summed_rl, freq_weighted)
	
		end_reduced_lr = calculate_reduced_end_lr(co_occurrence_list, freq_weighted)
		end_reduced_rl = calculate_reduced_end_rl(co_occurrence_list, freq_weighted)
	
		beginning_reduced_lr = calculate_reduced_beginning_lr(co_occurrence_list, freq_weighted)
		beginning_reduced_rl = calculate_reduced_beginning_rl(co_occurrence_list, freq_weighted)
	
		directional_scalar = calculate_directional_scalar(co_occurrence_list, freq_weighted)
		directional_categorical = calculate_directional_categorical(co_occurrence_list, freq_weighted)
		
		endpoint_lr, endpoint_rl = calculate_endpoint(co_occurrence_list, pairwise_dictionary, candidate_id, freq_weighted)
			
		#Third, create list of feature values for current candidate, including candidate id#
	
		candidate_vector = [full_candidate_str, 
							candidate_frequency, 
							summed_lr, 
							smallest_lr,
							summed_rl, 
							smallest_rl,
							normalized_summed_lr, 
							normalized_summed_rl, 
							beginning_reduced_lr,
							beginning_reduced_rl,
							end_reduced_lr,
							end_reduced_rl,
							directional_scalar,
							directional_categorical,
							endpoint_lr,
							endpoint_rl
						]
	
	return candidate_vector
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ----------------------------------------------------------#
#OUTPUT: Given Delta-P measure -------------------------------------------------------------------------------------#

def calculate_summed_rl(co_occurrence_list, freq_weighted):
	
	summed_rl = 0.0
	lowest_pairwise = ''

	for pair in co_occurrence_list:
		
		a = pair[0]
		b = pair[1]
		c = pair[2]
		d = pair[3]
		
		pair_rl = float(a / (a + b)) - float(c / (c + d))
		
		if freq_weighted == True:
			pair_rl = pair_rl * a
		
		#If threshold, then add or flag accordingly#
		if lowest_pairwise == '':
			lowest_pairwise = pair_rl
			
		elif pair_rl < lowest_pairwise:
			lowest_pairwise = pair_rl
				
		summed_rl += pair_rl
			
	return (summed_rl, lowest_pairwise)
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ----------------------------------------------------------#
#OUTPUT: Given Delta-P measure -------------------------------------------------------------------------------------#

def calculate_summed_lr(co_occurrence_list, freq_weighted):
	
	summed_lr = 0.0
	lowest_pairwise = ''

	for i in range(len(co_occurrence_list)):

		a = float(co_occurrence_list[i][0])
		b = float(co_occurrence_list[i][1])
		c = float(co_occurrence_list[i][2])
		d = float(co_occurrence_list[i][3])
		
		pair_lr = float(a / (a + c)) - float(b / (b + d))
		
		if freq_weighted == True:
			pair_lr = pair_lr * a
		
		#If threshold, then add or flag accordingly#
		if lowest_pairwise == '':
			lowest_pairwise = pair_lr
			
		elif pair_lr < lowest_pairwise:
			lowest_pairwise = pair_lr
				
		summed_lr += pair_lr
			
	return (summed_lr, lowest_pairwise)
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ----------------------------------------------------------#
#OUTPUT: Given Delta-P measure -------------------------------------------------------------------------------------#

def calculate_reduced_end_rl(co_occurrence_list, freq_weighted):
	
	length = len(co_occurrence_list) - 1
	
	temp_summed = calculate_summed_rl(co_occurrence_list, freq_weighted)
	main_summed = float(temp_summed[0])
	
	temp_end = calculate_summed_rl(co_occurrence_list[:length], freq_weighted)
	end_reduced_summed = float(temp_end[0])
	
	end_reduced_rl = main_summed - end_reduced_summed
	
	return end_reduced_rl
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ----------------------------------------------------------#
#OUTPUT: Given Delta-P measure -------------------------------------------------------------------------------------#

def calculate_reduced_end_lr(co_occurrence_list, freq_weighted):
	
	length = len(co_occurrence_list) - 1
	
	temp_summed = calculate_summed_lr(co_occurrence_list, freq_weighted)
	main_summed = float(temp_summed[0])
	
	temp_end = calculate_summed_lr(co_occurrence_list[:length], freq_weighted)
	end_reduced_summed = float(temp_end[0])
	
	end_reduced_lr = main_summed - end_reduced_summed
	
	return end_reduced_lr
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ----------------------------------------------------------#
#OUTPUT: Given Delta-P measure -------------------------------------------------------------------------------------#

def calculate_reduced_beginning_rl(co_occurrence_list, freq_weighted):
		
	temp_summed = calculate_summed_rl(co_occurrence_list, freq_weighted)
	main_summed = float(temp_summed[0])
	
	temp_reduced = calculate_summed_rl(co_occurrence_list[1:], freq_weighted)
	beginning_reduced_summed = float(temp_reduced[0])
	
	beginning_reduced_rl = main_summed - beginning_reduced_summed
	
	return beginning_reduced_rl
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ----------------------------------------------------------#
#OUTPUT: Given Delta-P measure -------------------------------------------------------------------------------------#

def calculate_reduced_beginning_lr(co_occurrence_list, freq_weighted):
		
	temp_summed = calculate_summed_lr(co_occurrence_list, freq_weighted)
	main_summed = float(temp_summed[0])
	
	temp_beginning = calculate_summed_lr(co_occurrence_list[1:], freq_weighted)
	beginning_reduced_summed = float(temp_beginning[0])
	
	beginning_reduced_lr = main_summed - beginning_reduced_summed
	
	return beginning_reduced_lr
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ----------------------------------------------------------#
#OUTPUT: Given Delta-P measure -------------------------------------------------------------------------------------#

def calculate_normalized_summed_rl(co_occurrence_list, summed_rl, freq_weighted):
	
	length = len(co_occurrence_list)
	normalized_summed_rl = summed_rl / length
	
	return normalized_summed_rl
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ----------------------------------------------------------#
#OUTPUT: Given Delta-P measure -------------------------------------------------------------------------------------#

def calculate_normalized_summed_lr(co_occurrence_list, summed_lr, freq_weighted):
	
	length = len(co_occurrence_list)
	normalized_summed_lr = summed_lr / length
	
	return normalized_summed_lr
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ----------------------------------------------------------#
#OUTPUT: Given Delta-P measure -------------------------------------------------------------------------------------#

def calculate_directional_scalar(co_occurrence_list, freq_weighted):
	
	directional_scalar = 0
	
	for i in range(len(co_occurrence_list) - 1):
	
		pair = co_occurrence_list[i:i+1]
		
		temp_lr = calculate_summed_lr(pair, freq_weighted)
		summed_lr = float(temp_lr[0])
		
		temp_rl = calculate_summed_rl(pair, freq_weighted)
		summed_rl = float(temp_rl[0])
		
		current_difference = summed_lr - summed_rl
		
		directional_scalar += current_difference
		
	directional_scalar = abs(directional_scalar)
	
	return directional_scalar
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ----------------------------------------------------------#
#OUTPUT: Given Delta-P measure -------------------------------------------------------------------------------------#

def calculate_directional_categorical(co_occurrence_list, freq_weighted):
	
	lr_dominate = 0
	rl_dominate = 0
	
	for i in range(len(co_occurrence_list) - 1):
	
		pair = co_occurrence_list[i:i+1]
	
		temp_lr = calculate_summed_lr(pair, freq_weighted)
		summed_lr = float(temp_lr[0])
		
		temp_rl = calculate_summed_rl(pair, freq_weighted)
		summed_rl = float(temp_rl[0])
		
		current_difference = summed_lr - summed_rl
		
		if current_difference > 0:
			lr_dominate += 1
			
		else:
			rl_dominate += 1
			
	#Combine pairwise dominance#
	directional_categorical = min(lr_dominate, rl_dominate)
	
	return directional_categorical
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: Full vector DataFrame, index lists, and file name ----------------------------------------------------------#
#OUTPUT: File with readable candidate vectors ----------------------------------------------------------------------#

def write_results(full_vector_df, 
					lemma_list, 
					pos_list, 
					category_list, 
					output_file_name, 
					encoding_type
					):

	fresults = open(output_file_name, "w", encoding=encoding_type)
	fresults.write('Name,Length,Template,Frequency,Summed_LR,Smallest_LR,Summed_RL,Smallest_RL,Normalized_Summed_LR,Normalized_Summed_RL,Beginning_Reduced_LR,Beginning_Reduced_RL,End_Reduced_LR,End_Reduced_RL,Directional_Scalar,Directional_Categorical,Endpoint_LR,Endpoint_RL,Beginning_Divided_LR,Beginning_Divided_RL,End_Divided_LR,End_Divided_RL\n')
	
	#Start loop through rows#
	for row in full_vector_df.itertuples():
		
		#First, produce readable construction representation#
		candidate_id = row[1]
		candidate_id = eval(candidate_id)
		candidate_str = ""
		template_str = ""
		item_counter = 0

		for item in candidate_id:
			item_counter += 1

			type = item[0]
			index = item[1]
			
			template_str += " " + str(type)
			
			if type == "Lex":
				readable_item = lemma_list[index]
				readable_item = "<" + readable_item + ">"
				
			elif type == "Pos":
				readable_item = pos_list[index]
				readable_item = readable_item.upper()
				
			elif type == "Cat":
				readable_item = category_list[index]
				readable_item = readable_item.upper()
								
			if item_counter == 1:
				candidate_str += str(readable_item)
			elif item_counter > 1:
				candidate_str += " + " + str(readable_item)
				
		fresults.write('"' + candidate_str + '",')
		#Done loop to create readable construction candidate#
		
		#Second, write features values#
		fresults.write(str(item_counter) + ',')
		fresults.write(str(template_str) + ',')
		fresults.write(str(row[2]) + ',')
		fresults.write(str(row[3]) + ',')
		fresults.write(str(row[4]) + ',')
		fresults.write(str(row[5]) + ',')
		fresults.write(str(row[6]) + ',')
		fresults.write(str(row[7]) + ',')
		fresults.write(str(row[8]) + ',')
		fresults.write(str(row[9]) + ',')
		fresults.write(str(row[10]) + ',')
		fresults.write(str(row[11]) + ',')
		fresults.write(str(row[12]) + ',')
		fresults.write(str(row[13]) + ',')
		fresults.write(str(row[14]) + ',')
		fresults.write(str(row[15]) + ',')
		fresults.write(str(row[16]) + ',')
		fresults.write(str(row[17]) + ',')
		fresults.write(str(row[18]) + ',')
		fresults.write(str(row[19]) + ',')
		fresults.write(str(row[20]) + '\n')		

	#End loop through candidates#
	fresults.close()
	
	return
#-------------------------------------------------------------------------------------------------------------------#
#--- Split output files into sub-sets for each process -------------------------------------------------------------#

def split_output_files(seq, num):
	
	if len(seq) < num:
		num = len(seq) / 2
	
	avg = len(seq) / float(num)
	out = []
	last = 0.0

	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg
		
	if len(out[-1]) == 1:
		del out[-1]

	return out
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: Single candidate, expanded data files, number of total units -----------------------------------------------#
#OUTPUT: List of features for current candidate --------------------------------------------------------------------#

def process_unitwise_feature_vector(candidate_info_list, 
										candidate_frequency_dict, 
										lemma_frequency, 
										lemma_list, 
										pos_frequency, 
										pos_list, 
										category_frequency, 
										category_list, 
										total_units,
										freq_weighted
										):
	
	vector_list = []
	
	candidate_id = candidate_info_list[0]
	candidate_length = candidate_info_list[1]
	candidate_frequency = candidate_info_list[2]
	full_candidate_str = str(candidate_id)
		
	#NOT PAIRWISE#
	# 11. Beginning-Divided ΔP, Left-to-Right#
	# 12. Beginning-Divided ΔP, Right-to-Left#
	# 13. End-Divided ΔP, Left-to-Right#
	# 14. End-Divided ΔP, Right-to-Left#
	
	if candidate_length < 3:
	
		divided_beginning_lr = 0
		divided_beginning_rl = 0
		divided_end_lr = 0
		divided_end_rl = 0
		
	else:
		
		#Break candidate into appropriate chunks#
		beginning_divided = [[candidate_id[0]], candidate_id[1:]]
		end_divided = [candidate_id[0:(len(candidate_id) -1)], [candidate_id[-1]]]
		
		beginning_list = get_unitwise_abcd(candidate_frequency, 
											beginning_divided, 
											candidate_frequency_dict, 
											lemma_frequency, 
											lemma_list, 
											pos_frequency, 
											pos_list, 
											category_frequency, 
											category_list, 
											total_units
											)
											
		end_list = get_unitwise_abcd(candidate_frequency, 
										end_divided, 
										candidate_frequency_dict, 
										lemma_frequency, 
										lemma_list, 
										pos_frequency, 
										pos_list, 
										category_frequency, 
										category_list, 
										total_units
										)
		
		if beginning_list == [[0,0,0,0]]:
			divided_beginning_lr = 0
			divided_beginning_rl = 0
			
		else:
			divided_beginning_lr_temp = calculate_summed_lr(beginning_list, freq_weighted)
			divided_beginning_lr = divided_beginning_lr_temp[0]
			
			divided_beginning_rl_temp = calculate_summed_rl(beginning_list, freq_weighted)
			divided_beginning_rl = divided_beginning_rl_temp[0]
		
		if end_list == [[0,0,0,0]]:
		
			divided_end_lr = 0
			divided_end_rl = 0
			
		else:
			divided_end_lr_temp = calculate_summed_lr(end_list, freq_weighted)
			divided_end_lr = divided_end_lr_temp[0]
			
			divided_end_rl_temp = calculate_summed_rl(end_list, freq_weighted)
			divided_end_rl = divided_end_rl_temp[0]
	
	vector_list	= [full_candidate_str, divided_beginning_lr, divided_beginning_rl, divided_end_lr, divided_end_rl]
	
	return vector_list
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: Single candidate, expanded data files, number of total units -----------------------------------------------#
#OUTPUT: List of features for current candidate --------------------------------------------------------------------#

def process_pairwise_feature_vector(candidate_info_list, pairwise_dictionary, freq_weighted):

	candidate_id = candidate_info_list[0]
	candidate_length = candidate_info_list[1]
	candidate_frequency = candidate_info_list[2]
	
	if candidate_length == 2:
		vector_list = create_pairwise_single(candidate_id, candidate_frequency, pairwise_dictionary, freq_weighted)
	
	elif candidate_length > 2:
		vector_list = create_pairwise_multiple(candidate_id, candidate_frequency, pairwise_dictionary, freq_weighted)
		
	return vector_list
#-------------------------------------------------------------------------------------------------------------------#
#--Load and merge candidate files ----------------------------------------------------------------------------------#

def process_merge_output(output_files, action = "Load"):

	#Initialize data structures#
	lemma_frequency = {}
	pos_frequency = {}
	category_frequency = {}
	number_of_words_total = 0
	candidate_dictionary = {}
	dictionary_key_list = []
	final_dictionary = {}
	
	#First, read in one output file to create the baseline items#
	if action == "Load":
		current_dictionary = read_candidates(output_files[0])
		
	elif action == "Pass":
		current_dictionary = output_files[0]
		
	#These items are shared and need to be saved once and checked for consistency only#
	sequence_list = current_dictionary['sequence_list']
	Grammar = current_dictionary['Grammar']
					
	#These items need to be merged across all output files#
	dictionary_key_list = list(current_dictionary['candidate_dictionary'].keys())
	lemma_frequency = current_dictionary['lemma_frequency']
	pos_frequency = current_dictionary['pos_frequency']
	category_frequency = current_dictionary['category_frequency']
	number_of_words = current_dictionary['number_of_words']
		
	if len(output_files) > 1:
		
		#Second, for all other files, make sure they have same info and merge results#
		for file in output_files[1:]:
			
			print("\tAdding files.")
				
			current_dictionary = {}
				
			if action == "Load":
				current_dictionary = read_candidates(file)
				
			elif action == "Pass":
				current_dictionary = file
				
			temp_dictionary = {}
			
			if "Grammar" in current_dictionary:
				
				#Check to ensure the otuput files were created with the same grammar file#
				if Grammar.POS_List == current_dictionary['Grammar'].POS_List and Grammar.Lemma_List == current_dictionary['Grammar'].Lemma_List:
					
					#These items need to be merged across all output files#
					lemma_frequency = ct.merge_with(sum, [lemma_frequency, current_dictionary['lemma_frequency']])
					pos_frequency = ct.merge_with(sum, [pos_frequency, current_dictionary['pos_frequency']])
					category_frequency = ct.merge_with(sum, [category_frequency, current_dictionary['category_frequency']])
						
					number_of_words_total += current_dictionary['number_of_words']	
						
					dictionary_key_list += list(current_dictionary['candidate_dictionary'].keys())
					dictionary_key_list = list(set(dictionary_key_list))

					for key in sequence_list:
						
						key = str(list(key))
							
						if key not in candidate_dictionary:
							candidate_dictionary[key] = {}
							
						if key not in current_dictionary['candidate_dictionary']:
							current_dictionary['candidate_dictionary'][key] = {}

						temp_dictionary[key] = ct.merge_with(sum, [candidate_dictionary[key], current_dictionary['candidate_dictionary'][key]])
							
					del candidate_dictionary
					candidate_dictionary = temp_dictionary
					del temp_dictionary
						
				else:
					print("\t\tFile did not match grammar elements. Not compatible.")
			else:
				print("Candidates not saved from this file.")
				
		final_candidate_dictionary = candidate_dictionary
			
		#Count total candidates#
		total = 0
		for key in final_candidate_dictionary.keys():
			total += len(final_candidate_dictionary[key].keys())
		#Done counting#
		
		print("")
		print("Done merging data: " + str(total) + " candidates before frequency threshold.")
		print("")
		
		final_dictionary['Grammar'] = Grammar
		
		final_dictionary['lemma_frequency'] = lemma_frequency
		final_dictionary['pos_frequency'] = pos_frequency
		final_dictionary['category_frequency'] = category_frequency
		final_dictionary['number_of_words'] = number_of_words_total
		final_dictionary['candidate_dictionary'] = final_candidate_dictionary
		final_dictionary['sequence_list'] = sequence_list

		return final_dictionary
		
	else:
		return current_dictionary
#-------------------------------------------------------------------------------------------------------------------#
#OUTPUT: Dictionary with all acceptable candidates, their frequency, other info ------------------------------------#

def merge_output(output_files, frequency_threshold, number_of_cpus, run_parameter = 0):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
		
		print("Starting to load and merge candidate files.")

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
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: list of all candidates -------------------------------------------------------------------------------------#
#OUTPUT: Dictionary with frequency of each candidate ---------------------------------------------------------------#
# Take full candidate list and return frequency dictionary ---------------------------------------------------------#

def get_unitwise_abcd(candidate_frequency, 
						sequence_list, 
						candidate_frequency_dict, 
						lemma_frequency, 
						lemma_list, 
						pos_frequency, 
						pos_list, 
						category_frequency, 
						category_list, 
						total_units
						):

	flag = 0
	
	#First, the total candidate frequency is occurrences of both elements together#
	a = candidate_frequency
	
	#Second, the solitary unit first is "b" and the solitary unit second is "c"#
	first = sequence_list[0]
	second = sequence_list[1]
	
	if len(first) == 1:
		if first[0][0] == "Lex":
			try:
				temp_name = lemma_list[first[0][1]]
				b = lemma_frequency[temp_name] - a
			except:
				b = 0
				print("First Lem Flag", end="")
				print(": ", end="")
				print(first)

		elif first[0][0] == "Pos":
			try:
				temp_name = pos_list[first[0][1]]
				b = pos_frequency[temp_name] - a
			except:
				b = 0
				print("First POS Flag", end="")
				print(": ", end="")
				print(first)

		elif first[0][0] == "Cat":
			try:
				temp_name = category_list[first[0][1]]
				b = category_frequency[temp_name] - a
			except:
				b = 0
				print("First Cat Flag", end="")
				print(": ", end="")
				print(first)
	
	elif len(second) == 1:
		if second[0][0] == "Lex":
			try:
				temp_name = lemma_list[second[0][1]]
				c = lemma_frequency[temp_name] - a
			except:
				c = 0
				print("Second Lemma flag", end="")
				print(": ", end="")
				print(second)

		elif second[0][0] == "Pos":
			try:
				temp_name = pos_list[second[0][1]]
				c = pos_frequency[temp_name] - a
			except:
				c = 0
				print("Second Pos flag", end="")
				print(": ", end="")
				print(second)

		elif second[0][0] == "Cat":
			try:
				temp_name = category_list[second[0][1]]
				c = category_frequency[temp_name] - a
			except:
				c = 0
				print("Second Category flag", end="")
				print(": ", end="")
				print(second)
	
	#Third, the non-solitary sequence first is "b" and the non-solitary sequence second is "c"#
	if len(first) > 1:
		try:
			b = ct.get(str(first), candidate_frequency_dict)
			b = b - a
		except:
			b = 0
			flag = 1
			
	elif len(second) > 1:
		try:
			c = ct.get(str(second), candidate_frequency_dict)
			c = c - a
		except: 
			c = 0
			flag = 1
		
	#Fourth, "d" is the total minus everything else#
	d = total_units - a - b - c
	
	
	if flag == 0:
		
		abcd_list = [[a, b, c, d]]
		
		if b < 0 or c < 0:
			print("Co-occurrence cannot be negative: ", end="")
			print(str(first) + " and " + str(second))
			abcd_list = [[0, 0, 0, 0]]

	elif flag == 1:
		
		abcd_list = [[0, 0, 0, 0]]		
	
	return abcd_list
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: List of unexpanded data files ------------------------------------------------------------------------------#
#OUTPUT: Total number of units in corpus ---------------------------------------------------------------------------#

def get_phrase_count(original_df, Grammar, lemma_frequency, pos_frequency):
	
	phrase_names = [x for x in Grammar.POS_List if x not in pos_frequency.keys()]
	
	try:
		del phrase_names[phrase_names.index('n/a')]
	except:
		print("")
	
	#print("")
	#print("Finding frequency for phrase types.")
	
	for key in phrase_names:
		
		pos_frequency[key] = 0
		lemma_frequency[key] = 0	
		
	#Count Phrases#
	for unit in phrase_names:
		
		print("\tCurrent phrase: " + str(unit))
		
		try:
			unit_index = Grammar.Lemma_List.index(unit)
			
		except:
			Grammar.Lemma_List.append(unit)
			unit_index = Grammar.Lemma_List.index(unit)
		
		unit_mask = original_df.loc[:, "Lex"] == unit_index
		unit_df = original_df.loc[unit_mask, ['Sent', 'Unit']]
		unit_df = unit_df.drop_duplicates(keep='first')
		unit_list = unit_df.values.tolist()
			
		current_count = len(unit_list)
		pos_frequency[unit] += current_count
		lemma_frequency[unit] += current_count
			
	return lemma_frequency, pos_frequency
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: Dictionary of pairs and their co-occurrences, unit index and frequency lists -------------------------------#
#OUTPUT: Dataframe with co-occurrence data (a, b, c) for each pair -------------------------------------------------#

def get_pairwise_df(pairwise_candidate, 
						lemma_frequency, 
						lemma_list, 
						pos_frequency, 
						pos_list, 
						category_frequency, 
						category_list, 
						total_units
						):

	candidate = str(pairwise_candidate[0])
	frequency = pairwise_candidate[2]
	
	#Assign label and index values#
	unit1_label = pairwise_candidate[0][0][0]
	unit1_index = pairwise_candidate[0][0][1]
	
	unit2_label = pairwise_candidate[0][1][0]
	unit2_index = pairwise_candidate[0][1][1]
	
	#Choose list for each pair type#
	if unit1_label == "Lex":
		unit1_list = lemma_frequency
		unit1_key = lemma_list[unit1_index]
	
	elif unit1_label == "Pos":
		unit1_list = pos_frequency
		unit1_key = pos_list[unit1_index]
		
	elif unit1_label == "Cat":
		unit1_list = category_frequency
		unit1_key = category_list[unit1_index]
		
	if unit2_label == "Lex":
		unit2_list = lemma_frequency
		unit2_key = lemma_list[unit2_index]
	
	elif unit2_label == "Pos":
		unit2_list = pos_frequency
		unit2_key = pos_list[unit2_index]
		
	elif unit2_label == "Cat":
		unit2_list = category_frequency
		unit2_key = category_list[unit2_index]
	
	#Calculate co-occurence frequencies#
	unit1_freq = unit1_list[unit1_key]
	unit2_freq = unit2_list[unit2_key]
	
	a = frequency
	b = unit1_freq - a
	c = unit2_freq - a
	d = total_units - a - b - c
	
	return [candidate, a, b, c, d]
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: list of all candidates -------------------------------------------------------------------------------------#
#OUTPUT: Dictionary with frequency of each candidate ---------------------------------------------------------------#
# Take full candidate list and return frequency dictionary ---------------------------------------------------------#

def get_frequency_dict(full_candidate_list):
	
	candidate_frequency_dict = {}
	
	for i in range(len(full_candidate_list)):
		
		candidate_id = str(full_candidate_list[i][0])
		candidate_length = full_candidate_list[i][1]
		candidate_frequency = full_candidate_list[i][2]
		
		candidate_frequency_dict[candidate_id] = candidate_frequency
			
	return candidate_frequency_dict
#-------------------------------------------------------------------------------------------------------------------#
#INPUT: list of all candidates -------------------------------------------------------------------------------------#
#OUTPUT: List of all pairs -----------------------------------------------------------------------------------------#
# Take full candidate list and return all two item pairs -----------------------------------------------------------#

def get_formatted_candidates(full_candidate_dictionary):
	
	candidate_list_formatted = []
	candidate_list_all = []
	candidate_list_pairs = []
	
	for key in full_candidate_dictionary.keys():
		
		current_template = eval(key)
		current_dictionary = full_candidate_dictionary[key]
		current_candidate_list = list(current_dictionary.keys())
		
		for j in range(len(current_candidate_list)):
			
			current_candidate = current_candidate_list[j]
			formatted_candidate = []
			candidate_length = 0
			
			for k in range(len(current_candidate)):
				temp_tuple = (current_template[k], current_candidate[k])
				formatted_candidate.append(temp_tuple)
				candidate_length += 1
			
			candidate_list_formatted.append(formatted_candidate)
			candidate_frequency = current_dictionary[current_candidate_list[j]]
			
			candidate_list_all.append([formatted_candidate, candidate_length, candidate_frequency])
			
			if candidate_length == 2:
				candidate_list_pairs.append([formatted_candidate, candidate_length, candidate_frequency])
				
	return [candidate_list_formatted, candidate_list_all, candidate_list_pairs]
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def calculate_endpoint(co_occurrence_list, pairwise_dictionary, candidate_id, freq_weighted):
		
	unit1 = str(candidate_id[0])
	unit2 = str(candidate_id[len(candidate_id)-1])
	
	a = 0
	b = 0
	c = 0
	d = 0
	
	candidate_str = "[" + unit1 + ", " + unit2 + "]"
				
	try:
		current_pair = ct.get(candidate_str, pairwise_dictionary)
	except:
		current_pair = []
			
	if current_pair !=[]:
		
		a = current_pair[0]
		b = current_pair[1]
		c = current_pair[2]
		d = current_pair[3]
		
	if a == 0:
		lr_measure = 0.0
		rl_measure = 0.0
		
	else:
		
		lr_measure = calculate_summed_lr([[a, b, c, d]], freq_weighted)
		lr_measure = lr_measure[0]
		
		rl_measure = calculate_summed_rl([[a, b, c, d]], freq_weighted)
		rl_measure = rl_measure[0]

	return lr_measure, rl_measure
#-------------------------------------------------------------------------------------------------------------------#