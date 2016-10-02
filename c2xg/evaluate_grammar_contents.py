#--------------------------------------------------------------#
#-- Take final construction grammar model and test set --------#
#-- Output word-level evaluation of coverage / size -----------#
#--------------------------------------------------------------#
def evaluate_by_word(model_file, 
					testing_file, 
					number_of_cpus,
					input_folder = "../../../../data/",
					encoding_type = "utf-8"
					):

	#Prevent pool workers from starting here#
	if __name__ == '__main__':
	#---------------------------------------#
		
		from functions_candidate_extraction.read_candidates import read_candidates
		from functions_candidate_pruning.process_coverage import process_coverage
		import math
		
		print("Loading model file.")
		write_dictionary = read_candidates(model_file)

		candidate_list = write_dictionary['candidate_list']
		lemma_list = write_dictionary['lemma_list']
		pos_list = write_dictionary['pos_list']
		word_list = write_dictionary['word_list']
		category_list = write_dictionary['category_list']
		semantic_category_dictionary = write_dictionary['semantic_category_dictionary']
		sequence_list = write_dictionary['sequence_list']
		max_construction_length = write_dictionary['max_construction_length']
		annotation_types = write_dictionary['annotation_types']
		
		phrase_constituent_list = write_dictionary['phrase_constituent_list']
		lemma_dictionary = write_dictionary['lemma_dictionary']
		pos_dictionary = write_dictionary['pos_dictionary']
		category_dictionary = write_dictionary['category_dictionary']
		emoji_dictionary = write_dictionary['emoji_dictionary']
		
		fo = open("Grammar_Description.txt", "w")
		
		result_dict = {}
		result_dict[2] = {}
		result_dict[3] = {}
		result_dict[4] = {}
		result_dict[5] = {}
		
		for i in [2, 3, 4, 5]:
			result_dict[i]["Contains_Lem"] = 0
			result_dict[i]["All_Lem"] = 0
			result_dict[i]["Contains_Pos"] = 0
			result_dict[i]["All_Pos"] = 0
			result_dict[i]["Contains_Cat"] = 0
			result_dict[i]["All_Cat"] = 0
		
		for construction in candidate_list:
			
			current_length = len(construction)
			current_labels = [label for (label, index) in construction]
			
			if "Lem" in current_labels:
				result_dict[current_length]["Contains_Lem"] += 1

				if "Pos" not in current_labels and "Cat" not in current_labels:
					result_dict[current_length]["All_Lem"] += 1

			if "Pos" in current_labels:
				result_dict[current_length]["Contains_Pos"] += 1
					
				if "Lem" not in current_labels and "Cat" not in current_labels:
					result_dict[current_length]["All_Pos"] += 1
					
			if "Cat" in current_labels:
				result_dict[current_length]["Contains_Cat"] += 1
					
				if "Pos" not in current_labels and "Lem" not in current_labels:
					result_dict[current_length]["All_Cat"] += 1
					
		#Now compute descriptives#
		for key in result_dict:
			print("Current length: " + str(key))
			
			print("")
			print("\tSyntactic, In: " + str(result_dict[key]["Contains_Pos"]))
			print("\tSyntactic, All: " + str(result_dict[key]["All_Pos"]))
			print("\tSyntactic, Mixed: " + str(result_dict[key]["Contains_Pos"] - result_dict[key]["All_Pos"]))
			
			print("")
			print("\tSemantic, In: " + str(result_dict[key]["Contains_Cat"]))
			print("\tSemantic, All: " + str(result_dict[key]["All_Cat"]))
			print("\tSemantic, Mixed: " + str(result_dict[key]["Contains_Cat"] - result_dict[key]["All_Cat"]))			
			
			print("")
			print("\tLexical, In: " + str(result_dict[key]["Contains_Lem"]))
			print("\tLexical, All: " + str(result_dict[key]["All_Lem"]))
			print("\tLexical, Mixed: " + str(result_dict[key]["Contains_Lem"] - result_dict[key]["All_Lem"]))
			
			print("")
			
			
						
			
				

			
		
		fo.close()
		
		
		return
#--------------------------------------------------------------#

evaluate_by_word("COLING.2.Constructions.model", "Testing.conll", 8)