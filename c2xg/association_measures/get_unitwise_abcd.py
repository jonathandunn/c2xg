#---------------------------------------------------------------------------------------------#
#FUNCTION: get_unitwise_abcd: ---------------------------------------------------------------#
#INPUT: list of all candidates ---------------------------------------------------------------#
#OUTPUT: Dictionary with frequency of each candidate -----------------------------------------#
# Take full candidate list and return frequency dictionary -----------------------------------#
#---------------------------------------------------------------------------------------------#
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

	import cytoolz as ct
	
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
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#