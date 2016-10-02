#-------------------------------------------------------------#
#-------------------------------------------------------------#
def check_constraints(pair, 
						pair_status, 
						head_status,
						pair_status_dictionary,
						pair_head_dictionary,
						head_status_dictionary
						):
	
	#print("Checking constraints on current set of catenae pairs.")
	
	#Constraint 1: Converse must not be catenae pair
	#Constraint 2: Heads must occupy same end-point
	
	pair_reverse = (pair[1], pair[0])
	rejection_flag = 0
	
	#Get head information#
	if head_status == "L":
		current_head = pair[0]
		current_non_head = pair[1]
		current_non_direction = "R"
		
	elif head_status == "R":
		current_head = pair[1]
		current_non_head = pair[0]
		current_non_direction = "L"
	
	#Check if identified head disagrees with previous head predictions#		
	if current_head in head_status_dictionary:
		
		#Consistent head prediction#
		if head_status_dictionary[current_head] == head_status:
			#print("Predicted head matches previous identifications.")
			null_counter = 0
			
		#Inconsistent head prediction#
		else:
			#print("Compatibility error: predicted head has opposite direction. Rejecting.")
			null_counter = 0
			
			#Try alternate head#
			if current_non_head in head_status_dictionary:
				if head_status_dictionary[current_non_head] == current_non_direction:
					#print("Based on previous identifications, changing predicted head.")
					head_status_dictionary[current_non_head] = current_non_direction
					pair_head_dictionary[pair] = current_non_direction
					
				elif head_status_dictionary[current_non_head] != current_non_direction:
					#print("No head matches previous direction. Rejecting as catenae pair.")
					pair_status_dictionary[pair] = "Non-Catenae"
					pair_head_dictionary[pair] = "n\a"
					rejection_flag = 1
			
	else:
		head_status_dictionary[current_head] = head_status
		pair_head_dictionary[pair] = head_status
		
	#Check if reverse is already identified as catenae#
	if rejection_flag == 0:
	
		if pair_reverse in pair_status_dictionary:
		
			if pair_status_dictionary[pair_reverse] == "Catenae":
				#print("Compatibility error: reverse pair is already catenae. Rejecting.")
				rejection_flag = 1
				
		else:
			#print("Adding pair and its reverse to pair status dictionary")
			pair_status_dictionary[pair_reverse] = "Non-Catenae"
			pair_head_dictionary[pair_reverse] = "n\a"
			pair_status_dictionary[pair] = "Catenae"
			
			if pair not in pair_head_dictionary:
				pair_head_dictionary[pair] = head_status
	
	return pair_status_dictionary, pair_head_dictionary, head_status_dictionary
#------------------------------------------------------------#	