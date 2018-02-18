import pickle
import time

if __name__ == "__main__":

	from modules.Encoder import Encoder
	from modules.Loader import Loader

	#Load association measure module
	Load = Loader("./In", "./Out")
	Encoder = Encoder(language = "eng", Loader = Load)
	
	#Files to test on
	files = ["eng.test.txt"]
	
	for line in Encoder.load(files):
	
		print(line)
	