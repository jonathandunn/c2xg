#---------------------------------------------------------------------------------------------#
#INPUT: Single sentence dataframe ------------------------------------------------------------#
#OUTPUT: single sentence dataframe with punctuation removed ----------------------------------#
#Take sentence and return dataframe without punctuation, etc. --------------------------------#
#---------------------------------------------------------------------------------------------#
def remove_punc(single_df, counter):

	import pandas as pd
	
	try:
		single_df = single_df[single_df.Pos != 0]
		
		pd.options.mode.chained_assignment = None
		single_df.loc[:,'Alt'] = counter
		pd.options.mode.chained_assignment = "warn"
		
	except:
		single_df = single_df
				
	return single_df
#---------------------------------------------------------------------------------------------#