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
		
		add_list = []
		ps_counter = 0
		for direction in phrase_constituent_list:
			for head in direction.keys():
				for rule in direction[head]:
					ps_counter += 1
					current_construction = []
					for slot in rule:
						current_construction.append(("Pos", slot))
						
					current_construction = tuple(current_construction)
					add_list.append(current_construction)
				
		candidate_list = [tuple(x) for x in candidate_list]
		
		add_counter = 0
		
		for ps_rule in add_list:
			if ps_rule not in candidate_list:
				add_counter += 1
				candidate_list.append(ps_rule)
				
			
		print("Added Ps rules as constructions: " + str(add_counter) + ", out of " + str(ps_counter))
		
		print("Starting process_coverage")
		coverage_dictionary = process_coverage(str(input_folder + "Input/Temp/" + testing_file), 
													candidate_list, 
													max_construction_length, 
													word_list,
													lemma_list, 
													pos_list, 
													category_list,
													lemma_dictionary, 
													pos_dictionary, 
													category_dictionary,
													semantic_category_dictionary,
													phrase_constituent_list,
													encoding_type,
													number_of_cpus
													)
													
		coverage_list = [coverage_dictionary[x] for x in candidate_list if coverage_dictionary[x] > 0]
		print(coverage_list)
		
		current_coverage = sum(coverage_list)
		current_size = len(candidate_list)
		
		current_quality = (current_coverage**2) / float(current_size)
		current_quality = -(math.log(current_quality, 10))
		
		print("")
		print("")
		print("Size: " + str(current_size))
		print("Coverage: " + str(current_coverage))
		print("Metric: " + str(current_quality))
		
		return
#--------------------------------------------------------------#

evaluate_by_word("COLING.2.Constructions.model", "Testing.conll", 8)