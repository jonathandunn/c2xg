#-----------------------------------------------------------------------------#
def move_generator_direct(tabu_search_df, checks_per_move, max_move_size):

	import random
	
	move_list = []
	index_list = list(tabu_search_df.index.values)
	
	for i in range(checks_per_move):
	
		move_size = random.randint(1,max_move_size)
		
		if move_size < len(index_list):
			move_list.append(random.sample(index_list, move_size))
			
		else:
			move_list.append(index_list)

	return move_list
#-----------------------------------------------------------------------------#