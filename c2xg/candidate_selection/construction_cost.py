#--------------------------------------------------------------#
def construction_cost(training_df, slot_r_cost, pos_unit_cost, lex_unit_cost, cat_unit_cost):

	cost_list = []
	
	for current_row in training_df.loc[:,"Candidate"].iteritems():
		
		construction = eval(current_row[1])
		cost = 0.0
		
		for slot in construction:
			
			slot_type = slot[0]
			cost += slot_r_cost
			
			if slot_type == "Pos":
				cost += pos_unit_cost
				
			elif slot_type == "Lex":
				cost += lex_unit_cost
				
			elif slot_type == "Cat":
				cost += cat_unit_cost
				
		cost_list.append(cost)
		
	training_df.loc[:,"Cost"] = cost_list
	
	return training_df
#--------------------------------------------------------------------------#