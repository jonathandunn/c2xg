#---------------------------------------------------------------------------------------------#
#INPUT: Input DF, pos_list, and directional remove dictionaries ------------------------------#
#OUTPUT: DataFrame with fully schematic versions of sentences --------------------------------#
#TProduce full schematic representation with all largest constituents reduced ----------------#
#---------------------------------------------------------------------------------------------#
def process_schematic_representation(single_df, 
										pos_list, 
										lemma_list,
										full_remove_dictionary_lr, 
										full_remove_dictionary_rl,
										counter
										):
	
	import cytoolz as ct
	from operator import itemgetter
	
	counter += 1
	
	tuple_list = []
	
	full_remove_dictionary = ct.merge(full_remove_dictionary_lr, full_remove_dictionary_rl)
	
	lr_heads = list(full_remove_dictionary_lr.keys())
	rl_heads = list(full_remove_dictionary_rl.keys())
	
	#Get index, length tuples for each head#
	for head in list(full_remove_dictionary.keys()):
		
		#Add (Head, Index, Length) tuples to list for merging#
		for index, length in full_remove_dictionary[head].items():
		
			current_tuple = (head, index, length)
			tuple_list.append(current_tuple)
			
	#Done creating (Head, Index, Length) tuples#
	
	#Sort from longest to shortest constituents#
	sorted_tuples = sorted(tuple_list, key=itemgetter(2))
	
	covered_indexes = []
	largest_constituents = []
	
	#Create collection of largest constituents not contained or overlapping with larger constituents#
	for constituent in sorted_tuples:
	
		current_head = constituent[0]
		current_index = constituent[1]
		current_length = constituent[2]
		
		if current_length > 1:
			current_covered = [x for x in range(current_index, current_index + current_length)]
			
		else:
			current_covered = [current_index]
		
		check = any(True for x in current_covered if x in covered_indexes)
		
		if check == False:
			covered_indexes += current_covered
			largest_constituents.append(constituent)
			
	del covered_indexes

	#Create DataFrame representing this fully schematic version#
	copy_df = single_df.copy("Deep")
	remove_list = []
	
	for constituent in largest_constituents:
		
		current_head = constituent[0]
		current_index = constituent[1]
		current_length = constituent[2]
		
		#Get identity of head index#
		if current_head in lr_heads:
			current_head_index = current_index
			
		elif current_head in rl_heads:
			current_head_index = current_index + current_length - 1
		
		#Get list of indexes constituting current constituent#
		if current_length > 1:
			current_covered = [x for x in range(current_index, current_index + current_length)]
		else:
			current_covered = [current_index]

		current_covered.remove(current_head_index)
		remove_list += current_covered
		
		current_pos_index = current_head
		copy_df.loc[copy_df.Mas == current_head_index, 'Pos'] = current_pos_index
			
	#Done changing phrase head information#
	#Now remove non-head members of constituents#
	copy_df = copy_df[~copy_df.Mas.isin(remove_list)]
	
	copy_df.loc[:,'Alt'] = counter

	return copy_df, counter
#---------------------------------------------------------------------------------------------#