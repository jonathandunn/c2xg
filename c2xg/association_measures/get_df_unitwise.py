#----------------------------------------------------------------------------------------------------------#
def get_df_unitwise(vector_list, condition):

	import pandas as pd
	
	vector_df = pd.DataFrame(vector_list, columns=['Candidate', 
													'Beginning_Divided_LR_' + condition, 
													'Beginning_Divided_RL_' + condition, 
													'End_Divided_LR_' + condition, 
													'End_Divided_RL_' + condition
													])
	return vector_df
#------------------------------------------------------------------------------------------------------------#