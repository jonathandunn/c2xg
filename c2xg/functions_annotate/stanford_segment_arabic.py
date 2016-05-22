#-------------------------------------------------------------------------------#
#INPUT: Memory limit, input file, working directory, encoding type -------------#
#OUTPUT: List of segmented lines as strings ------------------------------------#
#-------------------------------------------------------------------------------#
def stanford_segment_arabic(memory_limit, 
								file, 
								working_directory, 
								encoding_type
								):

	import subprocess
	import codecs
	import os
	
	run_string = 'javaw '
	run_string += ' -cp '
	run_string += '"*"'
	run_string += ' -Xmx'
	run_string += str(memory_limit)
	run_string += ' edu.stanford.nlp.international.arabic.process.ArabicSegmenter -loadClassifier ./Models/arabic-segmenter-atb+bn+arztrain.ser.gz -textFile '
	run_string += file
	run_string += ' -outputFormat text'
	run_string += ' > ' + file + '.out'
	
	subprocess.call(run_string, cwd = working_directory, shell=True)
	
	fo = codecs.open(file + ".out", "rb", encoding = encoding_type)
	counter = 0
	
	for line in fo:
		counter += 1
		line = line.decode(encoding_type, errors="replace")
		line_list.append((counter, line))
	
	fo.close()
	
	os.remove(file)
	
	return line_list
#--------------------------------------------------#