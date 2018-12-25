from c2xg import C2xG
import os

if __name__ == "__main__":

	language = "ara"
	
	#Initialize C2xG object
	CxG = C2xG(data_dir = "../../../Data", language = language)
		
	#Start or resume learning
	CxG.learn(nickname = language, 
				cycles = 2, 
				cycle_size = (1, 12, 120), 
				ngram_range = (3,6),
				freq_threshold = 25,
				turn_limit = 10,
				workers = 30,
				)