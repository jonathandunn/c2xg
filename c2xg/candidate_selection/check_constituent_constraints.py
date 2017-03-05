#---------------------------------------------------------------------------------#
def check_constituent_constraints(sequence_list, head_dictionary):

	total_sequences =  len(sequence_list)
	allowed_sequences = []
	
	#Now determine which sequences are well-formed#
	for sequence in sequence_list:
	
		left_head = sequence[0]
		right_head = sequence[-1]
		
		#Check if the left end-point is filled by a left head#
		if head_dictionary[left_head]["Status"] == 1:
			if head_dictionary[left_head]["Direction"] == 1:
			
				#Possible left-headed phrase: Check constraints#
				
				#-----Right endpoint not a head -----------------------or not a left head--------------------------or it is independent ------------------------#
				if head_dictionary[right_head]["Status"] != 1 or head_dictionary[right_head]["Direction"] != 1 or head_dictionary[right_head]["Independence"] == 1:
					allowed_sequences.append(sequence)
					#print("Left-headed: " + str(sequence))
			
		#Otherwise check if the right end-point is filled by a right head#
		elif head_dictionary[right_head]["Status"] == 1:
			if head_dictionary[right_head]["Direction"] == -1:
			
				#Possible right-headed phrase: Check constraints#
				
				#-----Left endpoint not a head -----------------------or not a right head--------------------------or it is independent ------------------------#
				if head_dictionary[left_head]["Status"] != 1 or head_dictionary[left_head]["Direction"] != -1 or head_dictionary[left_head]["Independence"] == 1:
					allowed_sequences.append(sequence)
					#print("Right-headed: " + str(sequence))
					
	return allowed_sequences
#----------------------------------------------------------------------------------#