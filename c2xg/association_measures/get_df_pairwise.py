#----------------------------------------------------------------------------------------------------------#
def get_df_pairwise(vector_list, condition):

	import pandas as pd

	vector_df = pd.DataFrame(vector_list, columns=['Candidate', 
													'Frequency', 
													'Summed_LR_' + condition,
													'Smallest_LR_' + condition,
													'Summed_RL_' + condition, 
													'Smallest_RL_' + condition,
													'Mean_LR_' + condition, 
													'Mean_RL_' + condition, 
													'Beginning_Reduced_LR_' + condition,
													'Beginning_Reduced_RL_' + condition,
													'End_Reduced_LR_' + condition,
													'End_Reduced_RL_' + condition,
													'Directional_Scalar_' + condition,
													'Directional_Categorical_' + condition,
													'Endpoint_LR_' + condition,
													'Endpoint_RL_' + condition
													])
	
	return vector_df
#------------------------------------------------------------------------------------------------------------#