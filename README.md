c2xg 1.0
=============

Computational Construction Grammar, or c2xg, is a Python package for learning of CxGs and working with CxGs. 

Why CxG? Constructions are grammatical entities that support a straight-forward quantification of linguistic structure.

This package currently support 35 languages: ara, bul, cat, ces, dan, deu, ell, eng, est, fas, fin, fra, glg, heb, hin, hun, ind, ita, kor, lav, nld, nor, pol, por, ron, rus, slv, spa, swe, tgl, tha, tur, ukr, urd, vie

Usage: Initializing
---------------------

The first task is to initialize an instance of c2xg:

		from c2xg import C2xG
		CxG = C2xG(language = eng)
	
The class initialization accepts the following variables:

	data_dir (str)	 		Either the path to the main data folder or, if using s3, the prefix name
	language (str)	 		Currently supports 35 languages (listed above)
	nickname (str)	 		If learning a new model, this creates a new namespace for saving temp files
	model (str)			 	If provided, loads a specific model; otherwise loads default grammar for the language
	zho_split(boolean)	 	Chinese text needs to be segmented into words; if False, the input text is already split
	max_words = (False, int)	The maximum number of words to process from a file; useful for experimenting with different amounts of training data
	fast_parse = (boolean)	If True, use the faster parser; recommended in most cases
		
Usage: Parsing
---------------

The Parse method takes a text or string and returns a sparse matrix with construction frequencies.

		vectors = CxG.parse_return(input, mode, workers)
		
This references the following settings:
	
	input (str / list of [strs])	The input, either filenames or texts, specified using **mode**
	mode (str)						files assumes input as filenames; lines takes a list of texts
	workers (int)					Number of processes to use
	
A generator function is also available.

      for vector in CxG.parse_yield(input, mode, workers):
            print(vector)
    
This references the following settings:

    input (str / list of [strs])	The input, either filenames or texts, specified using **mode**
	mode (str)						files assumes input as filenames; lines takes a list of texts

Usage: Showing Constructions
----------------------------
This function will show each construction, together with its index. The index corresponds with the column when extracting construction frequencies.

	CxG.print_constructions()

This function will read a text file and write a different text file with examples of each construction from that file. $n$ refers to the number of tokens per construction.

	CxG.print_examples(input_file, output_file, n)


Usage: Getting Association Values (Delta P)
-----------------------------

	association_csv = CxG.get_association(input_data, freq_threshold = 1, smoothing = False, lex_only = False)

Usage: Learning New Grammars
-----------------------------
The second task is to learn a new CxG. Most users will not need to train a new model.

		CxG.learn()
	
This references the following variables:

	nickname (str): Creates a unique namespace for saving temp files
	cycles (int): Number of unique folds to use; final grammars are merged across fold-specific grammars
	cycle_size (tuple of ints): The number of files to use for optimization data, for candidate extraction, and for background data
	freq_threshold (int): The number of occurrences required before a candidate construction is considered
	beam_freq_threshold (int): The frequency threshold used when searching for the best beam search parameters
	turn_limit (int): For the tabu search, how many turns to evaluate for making each move (x3 for the direct tabu search)
	workers (int): Number of processes to use; not every stage distributes well.
	mdl_workers (int): Number of processes to use for evaluating MDL during construction search; uses more memory
	fixed_set (list): Use a fixed set of files for each step in the algorithm; useful for experimenting with different types of input
	no_mdl (boolean): Limit the use of the MDL metric to the beam search component

Each learning fold consists of three tasks: (i) estimating association values from background data; this requires a large amount of data (e.g., 20 files); (ii) extracting candidate constructions; this requires a moderate amount of data (e.g., 5 files); (iii) evaluating potential grammars against a test set; this requires a small amount of data (e.g., 1 file or 10 mil words).

The freq_threshold is used to control the number of potential constructions to consider. It can be set at 20. The turn limit controls how far the search process can go. It can be set at 10.

Usage: Comparing Grammars
--------------
These section looks at some convenience functions for comparing or evaluating grammars. The first finds the fuzzy overlap between two grammars.

	fuzzy_jaccard(grammar1, grammar2, threshold = 0.70, workers = 2)

The second return the MDL for a given grammar using a given test corpus.

	get_mdl(candidates, file, workers = 2)

The third provides a method for pruning grammars using additional corpora. Constructions are slowly forgotten unless they reoccur regularly.

	forget_constructions(grammar, datasets, workers = None, threshold = 1, adjustment = 0.25, increment_size = 100000)

Installation
--------------

Installation files will be provided in the **c2xg/whl** directory.

		pip install <whl file>
		
