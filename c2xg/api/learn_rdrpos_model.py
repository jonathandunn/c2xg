#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

#-- Wrapper function for training and testing new RDRPOS models from tagged text

def learn_rdrpos_model(Parameters):

	print("")
	print("Starting C2xG.Learn_RDRPOS_Model")
	print("")

	#Import required modules#
	import codecs
	from process_input.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
	from process_input.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import unwrap_self_RDRPOSTagger
	from process_input.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import printHelp
	from process_input.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import train_model
	from process_input.rdrpos_tagger.Utility.Utils import readDictionary
	from process_input.rdrpos_tagger.Utility.Eval import computeAccuracies

	r = RDRPOSTagger()
	print("Loaded RDRPOS Tagger")

	train_model(Parameters.POS_Training_Folder)

	#Now, take gold test file and create unannotated version#
	fo = codecs.open(Parameters.POS_Testing_Folder, "r", encoding = Parameters.Encoding_Type)
	fw = codecs.open(Parameters.POS_Testing_Folder + ".raw", "w", encoding = Parameters.Encoding_Type)

	for line in fo:
	
		line_list = line.split()
		line_list_raw = []
	
		for unit in line_list:
			slash_index = unit.rfind("/")
			unit = unit[:slash_index]
			line_list_raw.append(unit)
		
		line_string = ""
	
		for unit in line_list_raw:
			line_string += str(unit) + " "
		
		line_string = line_string[:len(line_string)-1]
		fw.write(str(line_string))
		fw.write(str("\n"))
	
	fw.close()
	fo.close()

	#Now, use model to tag test corpus#
	r = RDRPOSTagger()
	
	model_string = training_dir + ".RDR"
	dict_string = training_dir + ".DICT"

	r.constructSCRDRtreeFromRDRfile(model_string)
	DICT = readDictionary(dict_string)

	fo = codecs.open(Parameters.POS_Testing_Folder + ".raw", "r", encoding = Parameters.Encoding_Type)
	fw = codecs.open(Parameters.POS_Testing_Folder + ".annotated", "w", encoding = Parameters.Encoding_Type)

	for line in fo:
		line_annotated = r.tagRawSentence(DICT, line)
		fw.write(str(line_annotated))
		fw.write(str("\n"))
	
	fo.close()
	fw.close()

	results = computeAccuracies(Parameters.POS_Training_Folder + ".DICT", Parameters.POS_Testing_Folder, Parameters.POS_Testing_Folder + ".annotated")

	known_results = results[0]
	unknown_results = results[1]
	total_results = results[2]

	print("Accuracy on known words: " + str(known_results))
	print("Accuracy on unknown words: " + str(unknown_results))
	print("Overall accuracy: " + str(total_results))
	
	return
#-------------------------------------------------------------------------------------------#