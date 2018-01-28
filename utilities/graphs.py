import seaborn as sns
import matplotlib as plt
import pandas as pd
import random

#-------------------------------------------------------------------------------------------------------------------------------------------#
def quality_across_folds():

	#FOR FIGURE 8 and FIGURE 9#
	
	#FAKE DATA#
	df_list = []

	for fold in [1, 2, 3, 5, 6, 7, 8, 9, 10]:
		for lang in ["de", "en", "nl", "es", "fr", "it"]:
			for grammar in ["Lex", "Con", "Full"]:
			
				value = random.uniform(0, 1)
				df_list.append((fold, lang, grammar, value))
				
	data_df = pd.DataFrame(df_list)
	data_df.columns = ["Fold", "Language", "Grammar", "Value"]
	#DONE FAKE DATA#

	g = sns.FacetGrid(data_df, row = "Grammar")
	g = (g.map(sns.pointplot, 
				x = "Fold", 
				y = "Value", 
				hue = "Language", 
				hue_order = ["de", "en", "nl", "es", "fr", "it"],
				palette = {"de": "darkblue", "en": "blue", "nl": "lightblue", "es": "darkgreen", "fr": "green", "it": "lightgreen"}, 
				data = data_df, 
				markers = ".",
				ci = None,
				legends = False
				))
	
	g.fig.get_axes()[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,
           ncol=6, mode=None, borderaxespad=0.)
	sns.plt.show()
	
	return
#----------------------------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------------------------------#
def quality_across_grammars():

	#FOR FIGURE 7#
	
	#FAKE DATA#
	df_list = []

	for fold in [1, 2, 3, 5, 6, 7, 8, 9, 10]:
		for lang in ["de", "en", "nl", "es", "fr", "it"]:
			for grammar in ["Lex", "Con", "Full"]:
			
				value = random.uniform(0, 1)
				df_list.append((fold, lang, grammar, value))
				
	data_df = pd.DataFrame(df_list)
	data_df.columns = ["Fold", "Language", "Grammar", "Value"]
	#DONE FAKE DATA#

	g = sns.pointplot( 
				x = "Grammar", 
				y = "Value", 
				hue = "Language", 
				hue_order = ["de", "en", "nl", "es", "fr", "it"],
				palette = {"de": "darkblue", "en": "blue", "nl": "lightblue", "es": "darkgreen", "fr": "green", "it": "lightgreen"}, 
				data = data_df, 
				markers = ".",
				ci = None,
				legends = False
				)
	
	g.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,
           ncol=6, mode=None, borderaxespad=0.)
	sns.plt.show()
	
	return
#----------------------------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------------------------------#
def iterations_across_searches():

	#FOR FIGURE 10#
	
	#FAKE DATA#
	df_list = []

	for fold in ["Direct", "Indirect"]:
		for lang in ["de", "en", "nl", "es", "fr", "it"]:
			for grammar in ["Lex", "Con", "Full"]:
			
				value = random.uniform(20,20000)
				df_list.append((fold, lang, grammar, value))
				
	data_df = pd.DataFrame(df_list)
	data_df.columns = ["Fold", "Language", "Grammar", "Value"]
	#DONE FAKE DATA#

	g = sns.FacetGrid(data_df, row = "Fold")
	g = (g.map(sns.pointplot, 
				x = "Grammar", 
				y = "Value", 
				hue = "Language", 
				hue_order = ["de", "en", "nl", "es", "fr", "it"],
				palette = {"de": "darkblue", "en": "blue", "nl": "lightblue", "es": "darkgreen", "fr": "green", "it": "lightgreen"}, 
				data = data_df, 
				markers = ".",
				ci = None,
				legends = False
				))
	
	g.fig.get_axes()[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,
           ncol=6, mode=None, borderaxespad=0.)
	sns.plt.show()
	
	return
#----------------------------------------------------------------------------------------------------------------------------------------#

iterations_across_searches()	
