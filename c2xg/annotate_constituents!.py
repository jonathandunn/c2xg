#-----C2xG, v 1.0 ----------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#---- Copyright, 2015-2016 Jonathan E. Dunn --------------------------------------------------#
#---------- www.jdunn.name -------------------------------------------------------------------#
#---------- jonathan.edwin.dunn@gmail.com ----------------------------------------------------#
#---------- Illinois Institute of Technology, Department of Computer Science -----------------#
#---------------------------------------------------------------------------------------------#

constituent_len_dictionary = ct.groupby(len, list(cfg_dictionary.keys()))
		length_list = list(constituent_len_dictionary.keys())
		length_list = sorted(length_list, reverse=False)
		
		for input_file in input_files:
		
			output_file = debug_file + "SentExamples"
			
			get_sent_examples(input_file,
							output_file,
							cfg_dictionary,
							constituent_len_dictionary,
							length_list,
							pos_list,
							encoding_type,
							semantic_category_dictionary,
							word_list,
							lemma_list,
							lemma_dictionary,
							pos_dictionary,
							category_dictionary,
							delete_temp
							)
			
			output_file = debug_file + "TypeExamples"
			
			get_type_examples(input_file,
							output_file,
							cfg_dictionary,
							constituent_len_dictionary,
							length_list,
							pos_list,
							encoding_type,
							semantic_category_dictionary,
							word_list,
							lemma_list,
							lemma_dictionary,
							pos_dictionary,
							category_dictionary,
							delete_temp
							)