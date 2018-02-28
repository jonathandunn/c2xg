import codecs
import pickle
import os

for file in os.listdir("."):
	if file.endswith(".txt"):
		
		domain_dict = {}

		with codecs.open(file, "r", encoding = "utf-8") as fo:

			for line in fo:
			
				line = line.strip().split(",")
				
				if len(line) == 2:
				
					word = line[0]
					domain = (int(line[1]) + 1)
					domain_dict[word] = domain

		with open(file.replace(".txt", ".p"), "wb") as fo:
			pickle.dump(domain_dict, fo, protocol = pickle.HIGHEST_PROTOCOL)
		
