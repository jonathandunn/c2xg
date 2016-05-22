#---------------------------------------------------------------------------------------------#
#INPUT: Current direction, dictionary of longest constituents, DF, head, examples file -------#
#OUTPUT: List of reductions of head last phrases ---------------------------------------------#
#Take formatted line and return versions with head last phrases reduced ----------------------#
#---------------------------------------------------------------------------------------------#
def constituents_print(pos_label, 
						head_list, 
						remove_list, 
						lemma_list, 
						original_df, 
						match_df, 
						direction, 
						examples_file, 
						encoding_type
						):

	import pandas as pd
	import codecs	
	
	fw = codecs.open(examples_file, "a", encoding = encoding_type)
	fw.write(str(pos_label).upper())
	fw.write(str("\n\n"))
	
	sentence_list = original_df.Sent.drop_duplicates().values.tolist()
	
	for sentence in sentence_list:
		original_sentence_df = original_df.query("Sent == @sentence", parser='pandas', engine='numexpr')
		reduced_sentence_df = match_df.query("Sent == @sentence", parser='pandas', engine='numexpr')
		
		fw.write(str("Current Sentence: " + str(sentence) + ":\n\t"))
		
		for row in original_sentence_df.itertuples():
			
			if row[7] in head_list:
			
				if direction == "LR":
					fw.write(str(" ["))
				
				fw.write(str(row[3]))
				
				if direction == "RL":
					fw.write(str("] "))
				
			elif row[7] in remove_list:
				
				if direction == "LR":
					fw.write(str("_"))
					fw.write(str(row[3]))
				
				elif direction == "RL":
					fw.write(str(" "))
					fw.write(str(row[3]))
					fw.write(str("_"))					
				
			else:
				fw.write(str(" "))
				fw.write(str(row[3]))				
					
		fw.write(str("\n\t"))
		
		for row in reduced_sentence_df.itertuples():
			fw.write(str(" "))
			fw.write(str(row[3]))
			
					
		fw.write(str("\n"))
	
	fw.close()
	
	return
#---------------------------------------------------------------------------------------------#