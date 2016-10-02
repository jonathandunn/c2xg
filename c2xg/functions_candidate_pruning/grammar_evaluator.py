#--------------------------------------------------------------#
#--Take grammar and coverage data and return grammar quality --#
#--------------------------------------------------------------#

from numba import jit
import math

@jit
def grammar_evaluator(size, coverage):

	coverage = coverage**2
	
	score = coverage / float(size)
	
	if score > 0:
		score = -(math.log(score, 10))
		
	else:
		score = 100

	return score
#-------------------------------------------------------------#