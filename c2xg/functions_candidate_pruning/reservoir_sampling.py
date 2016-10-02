#----------------------------------------------#
def reservoir_sampling(iterator, K):

	import random

	result = []
	N = 0

	for item in iterator:
		N += 1
		if len( result ) < K:
			result.append( item )
		else:
			s = int(random.random() * N)
			if s < K:
				result[ s ] = item

	return result
#----------------------------------------------#