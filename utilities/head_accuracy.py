#----------------------------------------------------#
def head_accuracy(input_file, gold_dictionary):

	fo = open(input_file, "r")
	
	correct_counter = 0
	incorrect_counter = 0
	total_counter = 0
	
	for line in fo:
	
		line_list = line.strip().split()
		
		tag = line_list[0].replace(":", "")
		status = line_list[1]
		
		if tag not in ["", "#"]:
			
			if status == gold_dictionary[tag]:
				correct_counter += 1
				total_counter += 1
			
			else:
				incorrect_counter += 1
				total_counter += 1
				
	accuracy = correct_counter / float(total_counter)
	
	print(input_file + ": " + str(accuracy))
		
	return
#---------------------------------------------------#
gold_dictionary = {}

gold_dictionary['ex'] = "Head-First"
gold_dictionary['jjs'] = "Non-Head"
gold_dictionary['nnp'] = "Head-Last"
gold_dictionary['vb']  = "Head-First"
gold_dictionary['cd'] = "Non-Head"
gold_dictionary['md'] = "Head-First"
gold_dictionary['pos'] = "Non-Head"
gold_dictionary['rbs'] = "Non-Head"
gold_dictionary['vbp'] = "Head-First"
gold_dictionary['fw'] = "Non-Head"
gold_dictionary['jj'] = "Head-First"
gold_dictionary['rb'] = "Non-Head"
gold_dictionary['vbn'] = "Head-First"
gold_dictionary[''] = "Non-Head"
gold_dictionary['ls'] = "Non-Head"
gold_dictionary['nns'] = "Head-Last"
gold_dictionary['rp'] = "Non-Head"
gold_dictionary['vbz'] = "Head-First"
gold_dictionary['in'] = "Head-First"
gold_dictionary['prp'] = "Non-Head"
gold_dictionary['uh'] = "Non-Head"
gold_dictionary['wdt'] = "Head-First"
gold_dictionary['#'] = "Non-Head"
gold_dictionary['jjr'] = "Head-First"
gold_dictionary['nnps'] = "Head-Last"
gold_dictionary['prp$'] = "Head-First"
gold_dictionary['to'] = "Head-First"
gold_dictionary['wp'] = "Head-First"
gold_dictionary['cc'] = "Head-Last"
gold_dictionary['nn'] = "Head-Last"
gold_dictionary['vbg'] = "Head-First"
gold_dictionary['wp$'] = "Head-First"
gold_dictionary['dt'] = "Head-First"
gold_dictionary['pdt'] = "Head-First"
gold_dictionary['rbr'] = "Non-Head"
gold_dictionary['vbd'] = "Head-First"
gold_dictionary['wrb'] = "Non-Head"

head_accuracy("1mil_2n_001.txt", gold_dictionary)
head_accuracy("1mil_2n_001.txt", gold_dictionary)
head_accuracy("1mil_2n_005.txt", gold_dictionary)
head_accuracy("1mil_2n_01.txt", gold_dictionary)
head_accuracy("1mil_2n_05.txt", gold_dictionary)
head_accuracy("1mil_3n_001.txt", gold_dictionary)
head_accuracy("1mil_3n_005.txt", gold_dictionary)
head_accuracy("1mil_3n_05.txt", gold_dictionary)
head_accuracy("1mil_3_01.txt", gold_dictionary)
head_accuracy("1mil_4n_001.txt", gold_dictionary)
head_accuracy("1mil_4n_005.txt", gold_dictionary)
head_accuracy("1mil_4n_01.txt", gold_dictionary)
head_accuracy("1mil_4n_05.txt", gold_dictionary)
head_accuracy("1mil_5n_001.txt", gold_dictionary)
head_accuracy("1mil_5n_005.txt", gold_dictionary)
head_accuracy("1mil_5n_01.txt", gold_dictionary)
head_accuracy("1mil_5n_05.txt", gold_dictionary)
head_accuracy("1mil_6n_001.txt", gold_dictionary)
head_accuracy("1mil_6n_005.txt", gold_dictionary)
head_accuracy("1mil_6n_01.txt", gold_dictionary)
head_accuracy("1mil_6n_05.txt", gold_dictionary)
head_accuracy("1mil_7n_001.txt", gold_dictionary)
head_accuracy("1mil_7n_005.txt", gold_dictionary)
head_accuracy("1mil_7n_01.txt", gold_dictionary)
head_accuracy("1mil_7n_05.txt", gold_dictionary)
head_accuracy("1mil_8n_001.txt", gold_dictionary)
head_accuracy("1mil_8n_005.txt", gold_dictionary)
head_accuracy("1mil_8n_01.txt", gold_dictionary)
head_accuracy("1mil_8n_05.txt", gold_dictionary)
head_accuracy("1mil_9n_001.txt", gold_dictionary)
head_accuracy("1mil_9n_005.txt", gold_dictionary)
head_accuracy("1mil_9n_01.txt", gold_dictionary)
head_accuracy("1mil_9n_05.txt", gold_dictionary)
head_accuracy("1mil_10n_001.txt", gold_dictionary)
head_accuracy("1mil_10n_005.txt", gold_dictionary)
head_accuracy("1mil_10n_01.txt", gold_dictionary)
head_accuracy("1mil_10n_05.txt", gold_dictionary)