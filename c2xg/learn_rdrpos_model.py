#-----C2xG, v 1.0 ----------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#---- Copyright, 2015-2016 Jonathan E. Dunn --------------------------------------------------#
#---------- www.jdunn.name -------------------------------------------------------------------#
#---------- jonathan.edwin.dunn@gmail.com ----------------------------------------------------#
#---------- Illinois Institute of Technology, Department of Computer Science -----------------#
#---------------------------------------------------------------------------------------------#
# learn_rdrpos_model -------------------------------------------------------------------------#
#INPUT: Training data and testing data (tagged part-of-speech text) --------------------------#
#OUTPUT: DICT and RDR files for tagging new data ---------------------------------------------#
#---------------------------------------------------------------------------------------------#

def train_rdrpos_model(training_dir, test_dir):

	#Import required modules#
	import codecs
	from functions_annotate.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
	from functions_annotate.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import unwrap_self_RDRPOSTagger
	from functions_annotate.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import printHelp
	from functions_annotate.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import train_model
	from functions_annotate.rdrpos_tagger.Utility.Utils import readDictionary
	from functions_annotate.rdrpos_tagger.Utility.Eval import computeAccuracies

	r = RDRPOSTagger()
	print("Loaded RDRPOS Tagger")

	train_model(training_dir)

	#Now, take gold test file and create unannotated version#
	fo = codecs.open(test_dir, "r", encoding = "utf-8")
	fw = codecs.open(test_dir + ".raw", "w", encoding = "utf-8")

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

	fo = codecs.open(test_dir + ".raw", "r", encoding = "utf-8")
	fw = codecs.open(test_dir + ".annotated", "w", encoding = "utf-8")

	for line in fo:
		line_annotated = r.tagRawSentence(DICT, line)
		fw.write(str(line_annotated))
		fw.write(str("\n"))
	
	fo.close()
	fw.close()

	results = computeAccuracies(training_dir + ".DICT", test_dir, test_dir + ".annotated")

	known_results = results[0]
	unknown_results = results[1]
	total_results = results[2]

	print("Accuracy on known words: " + str(known_results))
	print("Accuracy on unknown words: " + str(unknown_results))
	print("Overall accuracy: " + str(total_results))
	
	return
#-------------------------------------------------------------------------------------------#

#Put Train and Test files in "construction_induction/functions_annotate/rdrpos_tagger/data"
training_dir = ""
test_dir = ""

train_rdrpos_model(training_dir, test_dir)