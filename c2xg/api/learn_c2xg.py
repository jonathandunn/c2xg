#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

#-- Wrapper function for running the whole C2xG pipeline
#-- Learning RDRPOS models and word2vec dictionaries is considered outside the pipeline

def learn_c2xg(Parameters):

	import c2xg
	from api.learn_idioms import learn_idioms
	from api.learn_constituents import learn_constituents
	from api.learn_constructions import learn_constructions
	
	if Parameters.Run_Tagger == False:
		print("Warning: Idioms will not be detected and modeled without running the tagger at least once after learn_idioms.")
		sys.kill()

	print("Loading grammar")
	Grammar = c2xg.Grammar()
	print(Grammar.Idiom_List)
	Grammar = learn_idioms(Parameters, Grammar)
	Grammar = learn_constituents(Parameters, Grammar)
	Grammar = learn_constructions(Parameters, Grammar)
		
	return Grammar
#----------------------------------------------------------------------------------------------------#