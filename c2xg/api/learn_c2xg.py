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

	#Grammar = learn_idioms(Parameters)
	Grammar = c2xg.Grammar()
	Grammar = learn_constituents(Parameters, Grammar)
	Grammar = learn_constructions(Parameters, Grammar)
		
	return Grammar
#----------------------------------------------------------------------------------------------------#