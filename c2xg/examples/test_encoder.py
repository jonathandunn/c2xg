import pickle
import time
import os

if __name__ == "__main__":

	from modules.Encoder import Encoder
	from modules.Loader import Loader

	#Load association measure module
	in_dir = os.path.join("..", "..", "..", "..", "Test", "In")
	out_dir = os.path.join("..", "..", "..", "..", "Test", "Out")
	Load = Loader(in_dir, out_dir, language = "eng")
	Encoder = Encoder(Loader = Load)
	
	for i in range(1, 10):
		#Files to test on
		files = ["eng.1.txt"]
		counter = 0
			
		starting = time.time()
		for line in Encoder.load_stream(files):
			counter += 1
			
		print(str(counter) + " in " + str(time.time() - starting))