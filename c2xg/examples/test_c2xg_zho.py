from c2xg import C2xG
import os

if __name__ == "__main__":

	#Initialize C2xG object
	CxG = C2xG(data_dir = "Data", language = "zho")
		
	#Start or resume learning
	CxG.learn(nickname = "zho2", 
				cycles = 5, 
				cycle_size = (1, 5, 20), 
				ngram_range = (3,6),
				freq_threshold = 40,
				turn_limit = 10,
				workers = 30
				)