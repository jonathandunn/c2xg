#---------------------------------------------------------------------------------------------#
#INPUT: List of files containing formatted corpus, max_sentences, unit frequency threshold ---#
#OUTPUT: Lists of frequency reduced unit labels ----------------------------------------------#
#Open files, loop through lines, send lines to other functions, write list of sentences ------#
#---------------------------------------------------------------------------------------------#
def create_unit_index(input_files, Parameters):

	import csv
	import cytoolz as ct
	
	from process_input.process_line_ingest import process_line_ingest
	
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