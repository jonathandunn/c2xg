#-------------------------------------------------------------------------------#
#INPUT: Memory limit, input file, working directory, encoding type -------------#
#OUTPUT: List of segmented lines as strings ------------------------------------#
#-------------------------------------------------------------------------------#
def stanford_segment_arabic(Parameters, file):

	import subprocess
	import codecs
	import os
	
	run_string = 'javaw '
	run_string += ' -cp '
	run_string += '"*"'
	run_string += ' -Xmx'
	run_string += str(Parameters.Stanford_Memory_Limit)
	run_string += ' edu.stanford.nlp.international.arabic.process.ArabicSegmenter -loadClassifier ./Models/arabic-segmenter-atb+bn+arztrain.ser.gz -textFile '
	run_string += file
	run_string += ' -outputFormat text'
	run_string += ' > ' + file + '.out'
	
	subprocess.call(run_string, cwd = Parameters.Stanford_Working_Directory, shell = True)
	
	fo = codecs.open(file + ".out", "rb", encoding = Parameters.Encoding_Type)
	counter = 0
	
	for line in fo:
		counter += 1
		line = line.decode(Parameters.Encoding_Type, errors="replace")
		line_list.append((counter, line))
	
	fo.close()
	
	os.remove(file)
	
	return line_list
#--------------------------------------------------#