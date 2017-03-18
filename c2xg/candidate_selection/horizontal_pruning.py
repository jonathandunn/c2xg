#-----------------------------------------------#
def horizontal_pruning(final_grammar):

	pruned_grammar = []

	for construction1 in final_grammar:
	
		remove_flag = 0

		for construction2 in final_grammar:
			
			if construction2 != construction1 and len(construction2) > len(construction1):
				
				#Left check#
				if construction1 == construction2[0:len(construction1)]:
					remove_flag = 1
					
				#Right check#
				elif construction1 == construction2[-(len(construction1)):]:
					remove_flag = 1
					
		if remove_flag == 0:
			pruned_grammar.append(construction1)
			
	print("Length of unpruned grammar: " + str(len(final_grammar)))
	print("Length of pruned grammar: " + str(len(pruned_grammar)))		

	return pruned_grammar
#-----------------------------------------------#