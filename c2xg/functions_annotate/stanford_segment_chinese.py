#-------------------------------------------------------------------------------#
#INPUT: Memory limit, input file, working directory, encoding type -------------#
#OUTPUT: List of segmented lines as strings ------------------------------------#
#-------------------------------------------------------------------------------#
def stanford_segment_chinese(memory_limit,
								file,
								working_directory, 
								encoding_type
								):

	import subprocess
	import codecs
	import os
	
	slash_index = file.rfind("/")
	output_dir = file[:slash_index]
	
	run_string = 'javaw '
	run_string += ' -cp '
	run_string += '"*"'
	run_string += ' -Xmx'
	run_string += str(memory_limit)
	run_string += ' edu.stanford.nlp.pipeline.StanfordCoreNLP -props chinese.properties -annotators segment,ssplit -file '
	run_string += file
	run_string += ' -outputFormat text'
	run_string += ' -outputDirectory '
	run_string += output_dir
	
	subprocess.call(run_string, cwd = working_directory, shell=True)
	
	line_list = []
	line_string = ""	
	counter = 0
	
	fo = codecs.open(file + ".out", encoding = encoding_type)
	for line in fo:
	
		if line[0:5] == "[Text":
			
			temp_list = line.split(" ")
			current_word = temp_list[0].replace("[Text=","")
			
			if current_word != "EOL":
				line_string += current_word
				line_string += " "
			
			elif current_word == "EOL":
				line_string = line_string.replace("\n", "").replace("\r", "").replace("http : //", "http://").replace("https : //","https://")
				if line_string != "":
					line_list.append(line_string)
				line_string = ""
	
	line_string = line_string.replace("\n", "").replace("\r", "").replace("http : //", "http://").replace("https : //","https://")
	if line_string != "":
		line_list.append((counter, line_string))
		
	os.remove(file)
	
	return line_list
#--------------------------------------------------#