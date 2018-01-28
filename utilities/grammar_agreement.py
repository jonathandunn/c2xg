#-----------------------------------------------------#
#Calculate agreement between grammars using output CSV#
#-----------------------------------------------------#
def calculate_agreement(grammar1, grammar2):

	import codecs

	grammar1_list = []
	grammar2_list = []

	#Load first grammar output CSV#
	print("")
	print("")
	print("Loading  " + str(grammar1))
	f1 = codecs.open(grammar1, "r", encoding = "utf-8")

	for line in f1:
	
		line_list = line.split(",")
		construction = line_list[0]
		grammar1_list.append(construction)
	
	f1.close()

	#Load second grammar output CSV#
	print("Loading " + str(grammar2))
	f2 = codecs.open(grammar2, "r", encoding = "utf-8")

	for line in f2:
	
		line_list = line.split(",")
		construction = line_list[0]
		grammar2_list.append(construction)
	
	f2.close()

	#Calculate agreement between grammars#
	print("Calculating agreement between grammars")
	shared_items = 0

	for construction in grammar1_list:
		if construction in grammar2_list:
			shared_items += 1
		
	print("Total items in grammar 1: " + str(len(grammar1_list)))
	print("Total items shared with grammar 2: " + str(shared_items))
	print("Total agreement: " + str(float(shared_items) / len(grammar1_list)))
	
	return
#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#
grammar1 = "ukWac.Subset-1Results.FreqIndv=100.FreqCon=100.Length=7.csv"
grammar2 = "ukWac.Subset-2Results.FreqIndv=100.FreqCon=100.Length=7.csv"
calculate_agreement(grammar1, grammar2)