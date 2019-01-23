from c2xg import C2xG
import os

if __name__ == "__main__":

	language = "por"
	
	#Initialize C2xG object
	CxG = C2xG(data_dir = "../../../Data", language = language)
		
	#Start or resume learning
	CxG.learn(nickname = language, 
				cycles = 4, 
				cycle_size = (1, 5, 60), 
				freq_threshold = 25,
				beam_freq_threshold = 10,
				turn_limit = 10,
				workers = 20,
				mdl_workers = 1,
				# states = [
							# [2, "Background_State", "None"],
							# [2, "Candidate_State", "Threshold"],
							# [2, "MDL_State", "None"],
							# ]
				)