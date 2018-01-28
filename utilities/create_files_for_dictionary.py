#-------------------------------------------------------#
#--create_files for dictionary--------------------------#
#INPUT: Files to prepare for making semantic dictionary-#
#OUTPUT: Corpus of files for making semantic dictioanry-#
#-------------------------------------------------------#

from construction_induction.functions_annotate.load_utf8 import load_utf8
from construction_induction.functions_annotate.process_lines import process_lines
from construction_induction.functions_input.create_emoji_dictionary import create_emoji_dictionary
import codecs

input_files = [
				file1,
				file2
]

encoding_type = "utf-8"
language = desired language
memory_limit = "8g"
working_directory = 
emoji_file = ""
output_file = "'

emoji_dictionary = create_emoji_dictionary(emoji_file)
line_list = load_utf8(input_file, encoding_type, language, memory_limit, working_directory)

fw = codecs.open(output_file, "w", encoding = encoding_type)

for line in line_list:
	line = process_lines(line, emoji_dictionary)
	fw.write(str(line))
	fw.write(str("\n"))
	
fw.close()
	

	

