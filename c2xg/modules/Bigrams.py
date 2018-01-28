from modules.Encoder import Encoder
import pandas as pd
import cytoolz as ct
import multiprocessing as mp
import time
import pickle
from collections import defaultdict
from functools import partial

#--------------------------------------------------------------------------------------------#
def process_bigrams(filename, Encoder, save = False):

	#Initialize bigram dictionary
	ngrams = defaultdict(int)
	unigrams = defaultdict(int)
			
	starting = time.time()
	total = 0

	for line in Encoder.load(filename):
	
		total += len(line)

		#Store unigrams
		for item in line:
			unigrams[(1, item[0])] += 1
			unigrams[(2, item[1])] += 1
			unigrams[(3, item[2])] += 1
		
		try:
			for bigram in ct.sliding_window(2, line):
				
				#Tuples are indexes for (LEX, POS, CAT)
				#Index types are 1 (LEX), 2 (POS), 3 (CAT)
				ngrams[((1, bigram[0][0]), (1, bigram[1][0]))] += 1	#lex_lex
				ngrams[((1, bigram[0][0]), (2, bigram[1][1]))] += 1	#lex_pos
				ngrams[((1, bigram[0][0]), (3, bigram[1][2]))] += 1	#lex_cat
				ngrams[((2, bigram[0][1]), (2, bigram[1][1]))] += 1	#pos_pos
				ngrams[((2, bigram[0][1]), (1, bigram[1][0]))] += 1	#pos_lex
				ngrams[((2, bigram[0][1]), (3, bigram[1][2]))] += 1	#pos_cat 
				ngrams[((3, bigram[0][2]), (3, bigram[1][2]))] += 1	#cat_cat
				ngrams[((3, bigram[0][2]), (2, bigram[1][1]))] += 1	#cat_pos
				ngrams[((3, bigram[0][2]), (1, bigram[1][0]))] += 1	#cat_lex
		
		except Exception as e:
			error = e

	#Reduce nonce ngrams
	size = len(list(ngrams.keys()))
	keepable = lambda x: x > 1
	ngrams = ct.valfilter(keepable, ngrams)
	unigrams = ct.valfilter(keepable, unigrams)
	
	#Reduce null indexes
	ngrams = {key: ngrams[key] for key in list(ngrams.keys()) if 0 not in key[0] and 0 not in key[1]}
	unigrams = {key: unigrams[key] for key in list(unigrams.keys()) if 0 not in key}
	
	ngrams = ct.merge([ngrams, unigrams])	
	ngrams["TOTAL"] = total
	
	del unigrams
	
	#Print status
	print("Time: ", end = "")
	print(time.time() - starting, end = "")
	print(" Full: " + str(size) + " ", end = "")
	print(" Reduced: ", end = "")
	print(len(list(ngrams.keys())), end = "")
	print(" with " + str(ngrams["TOTAL"]) + " words.")
	
	if save == True:
	
		with open(filename + ".ngrams.p", "wb") as handle:
			pickle.dump(ngrams, handle, protocol = pickle.HIGHEST_PROTOCOL)
			
	else:
		return ngrams
#--------------------------------------------------------------------------------------------#

if __name__ == "__main__":

	language = "eng"
	workers = 4
	files = [
		"eng.1.txt",
		"eng.2.txt",
		"eng.3.txt",
		"eng.4.txt",
		"eng.5.txt",
		"eng.6.txt",
		"eng.7.txt",
		"eng.8.txt"
		]
		
	#Initialize Ingestor
	Encoder = Encoder(language = language)

	starting = time.time()
	
	#Multi-process#
	pool_instance = mp.Pool(processes = workers, maxtasksperchild = 1)
	pool_instance.map(partial(process_bigrams, Encoder = Encoder, save = True), files, chunksize = 1)
	pool_instance.close()
	pool_instance.join()
	
	#Merge individual bigram dictionaries
	
	#ngrams = ct.merge_with(sum, [x for x in ngrams])
	#print("TOTAL TIME: " + str(time.time() - starting))
	#print("TOTAL NGRAMS: " + str(len(list(ngrams.keys()))))
	#print("TOTAL WORDS: " + str(ngrams["TOTAL"]))