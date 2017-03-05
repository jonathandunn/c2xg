#---------------------------------------------------------------------------------------------#
#FUNCTION: write_results ---------------------------------------------------------------------#
#INPUT: Full vector DataFrame, index lists, and file name ------------------------------------#
#OUTPUT: File with readable candidate vectors ------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def write_results(full_vector_df, 
					lemma_list, 
					pos_list, 
					category_list, 
					output_file_name, 
					encoding_type
					):
	
	import pandas as pd
	
	from process_input.create_category_dictionary import create_category_dictionary
	
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
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#