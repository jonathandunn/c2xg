import codecs
import pickle

domain_dict = {}
file = "Spanish.Aranea.DIM=500.SG=1.HS=1.ITER=25"
counter = 0

with codecs.open(file + ".txt", "r", encoding = "utf-8") as fo:

	for line in fo:
	
		line = line.strip().split(",")
		
		if len(line) == 2:
		
			counter += 1
			word = line[0]
			domain = (int(line[1]) + 1)
			domain_dict[word] = {}
			domain_dict[word]["index"] = counter
			domain_dict[word]["domain"] = domain
			
with open(file + ".p", "wb") as fo:
	pickle.dump(domain_dict, fo)
		
