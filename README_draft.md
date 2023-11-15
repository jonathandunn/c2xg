# c2xg 2.0
=============

Computational Construction Grammar, or *c2xg*, is a Python package for learning and working with construction grammars. 

Why CxG? Constructions are grammatical entities that support a straight-forward quantification of linguistic structure.

This package currently support 18 languages: English (eng), Arabic (ara), Danish (dan), German (deu), Greek (ell), Farsi (fas), Finnish (fin), French (fra), Hindi (hin), Indonesian (ind), Italian (ita), Dutch (nld), Polish (pol), Portuguese (por), Russian (rus), Spanish (spa), Swedish (swe), Turkish (tur)

## Further Documentation
-----------------------

More detailed linguistic documentation is available in the draft book *Computational Construction Grammar: A Usage-Based Approach* available at https://www.jdunn.name/cxg

Usage examples in a working environment are available at https://doi.org/10.24433/CO.9944630.v1

Detailed descriptions of each pre-trained grammar are available at https://doi.org/10.17605/OSF.IO/SA6R3

## Download Models
--------------------

A number of pre-trained grammar models are available for download and use with C2xG.

Note: The *download_model()* function must be used to install pre-trained grammars.

	from c2xg import download_model

Models can be downloaded as follows: 

	download_model(model = False, data_dir = None, out_dir = None)

These parameters are as follows:

	model (str)		Name of a pre-trained grammar model or its shortcut
	data_dir (str)	Main data directory, creates 'data' in current directory if none given
	out_dir (str)	Output data directory, creates 'OUT' in main data directory if none given

<details>
	<summary>Model Files</summary>

	"BL": "cxg_corpus_blogs_final_v2.eng.1000k_words.model.zip",
	"NC": "cxg_corpus_comments_final_v2.eng.1000k_words.model.zip",
	"EU": "cxg_corpus_eu_final_v2.eng.1000k_words.model.zip",
	"PG": "cxg_corpus_pg_final_v2.eng.1000k_words.model.zip",
	"PR": "cxg_corpus_reviews_final_v2.eng.1000k_words.model.zip",
	"OS": "cxg_corpus_subs_final_v2.eng.1000k_words.model.zip",
	"TW": "cxg_corpus_tw_final_v2.eng.1000k_words.model.zip",
	"WK": "cxg_corpus_wiki_final_v2.eng.1000k_words.model.zip",
	"ara": "cxg_multi_v02.ara.1000k_words.model.zip",
	"dan": "cxg_multi_v02.dan.1000k_words.model.zip",
	"deu": "cxg_multi_v02.deu.1000k_words.model.zip",
	"ell": "cxg_multi_v02.ell.1000k_words.model.zip",
	"eng": "cxg_multi_v02.eng.1000k_words.model.zip",
	"fas": "cxg_multi_v02.fas.1000k_words.model.zip",
	"fin": "cxg_multi_v02.fin.1000k_words.model.zip",
	"fra": "cxg_multi_v02.fra.1000k_words.model.zip",
	"hin": "cxg_multi_v02.hin.1000k_words.model.zip",
	"ind": "cxg_multi_v02.ind.1000k_words.model.zip",
	"ita": "cxg_multi_v02.ita.1000k_words.model.zip",
	"nld": "cxg_multi_v02.nld.1000k_words.model.zip",
	"pol": "cxg_multi_v02.pol.1000k_words.model.zip",
	"por": "cxg_multi_v02.por.1000k_words.model.zip",
	"rus": "cxg_multi_v02.rus.1000k_words.model.zip",
	"spa": "cxg_multi_v02.spa.1000k_words.model.zip",
	"swe": "cxg_multi_v02.swe.1000k_words.model.zip",
	"tur": "cxg_multi_v02.tur.1000k_words.model.zip",

</details>

## Installation
--------------

For the full package:

	pip install git+https://github.com/jonathandunn/c2xg.git

For only deu, eng, fra, ita, por, spa:

	pip install c2xg

## Usage: Initializing
---------------------

The first task is to initialize an instance of c2xg:

Depending on your machine, this may take a few minutes.

	from c2xg import C2xG
	CxG = C2xG(model = "BL") 

The class initialization accepts the following variables:

	model (str)				Pre-trained grammar file name in the out directory, or corresponding shortcut
						See "Download Models" and "Further Documentation" for more
	data_dir (str)			Working directory, creates 'data' in current directory if none given
	in_dir (str)			Input directory name, creates 'IN' in 'data_dir' if none given
	out_dir (str)			Output directory name, creates 'OUT' in 'data_dir' if none given
	language (str)			Language for file names, default 'N/A'
	nickname (str) 			Nickname for file names, default 'cxg'
	max_sentence_length (int) 	Cutoff length for loading a given sentence, 50 by default
	normalization (bool)		Normalize frequency by ngram type and frequency strata, True by default
	max_words (bool) 			Limit the number of words when reading input data, False by default
	cbow_file (str)			Name of cbow file to load or create
	sg_file (str) 			Name of skip-gram file to load or create

## Usage: Grammar Parsing
---------------------------

### CxG.parse()

The *parse()* method takes a text, file name, or list of file names and returns a sparse matrix with construction token frequencies.

	CxG.parse(self, input, input_type = "files", mode = "syn", third_order = False)

This references the following settings:

	input (str or list of str)	A filename or list of filenames to be parsed, sourced from 'in' directory
	input_type (str)			"files" if input contains filenames or "lines" if input contains data
	mode (str, default "syn")	Type(s) of constructions to be parsed ("lex", "syn", "full", or "all")
	third_order (bool)		Whether third-order constructions are used, False by default

### CxG.parse_types()

The *parse_types()* method takes a text, file name, or list of file names and returns a sparse matrix with construction type frequencies.

	CxG.parse_types(self, input, input_type = "files", mode = "syn", third_order = False)

Which takes the following parameters:

	input (str or list of str)	A filename or list of filenames to be parsed, sourced from 'in' directory
	input_type (str)			"files" if input contains filenames or "lines" if input contains data
	mode (str, default "syn")	Type(s) of representations to be parsed ("lex", "syn", "full", or "all")
	third_order (bool)		Whether third-order constructions are used, False by default

## Usage: Grammar Analysis
----------------------------

### CxG.get_type_token_ratio()

The *get_type_token_ratio()* method takes a text, file name, or list of file names and returns the type/token ration for each construction. 

	get_type_token_ratio(self, input_data, input_type, mode = "syn", third_order = False)

Which takes the following parameters:

	input (str or list of str)	A filename or list of filenames to be parsed, sourced from 'in' directory
	input_type (str)			"files" if input contains filenames or "lines" if input contains data
	mode (str, default "syn")	Type(s) of representations to be parsed ("lex", "syn", "full", or "all")
	third_order (bool)		Whether third-order constructions are used, False by default

### CxG.get_association()

The *get_association()* method returns a list of association measure (Delta P) statistics for word pairs in a given input. Also creates a .txt file in the out directory with a list of constructions.

For more on these measures, see https://arxiv.org/abs/2104.01297

	get_association(self, freq_threshold = 1, normalization = True, grammar_type = "full", lex_only = False, data = False)

Which takes the following parameters:

	freq_threshold (int) 		Only consider bigrams above this frequency threshold, 1 by default
	normalization (bool) 		Normalize frequency by ngram type and frequency strata, True by default
	grammar_type (str)		Suffix for pickle file name for file containing discounts, default "full"
	lex_only (bool)			Limit n-grams examined to lexical entries only, False by default
	data (str or list of str)	A filename or list of filenames to be parsed, sourced from 'in' directory

## Usage: Grammar Exploration
-------------------------------

### CxG.print_constructions()

The *print_constructions()* function prints, returns, and creates a .txt file with a list of constructions in the loaded model.

	print_constructions(self, mode="lex")

Which takes the following parameters:

	mode (str, default "lex")	Type(s) of representations to be examined ("lex", "syn", "full", or "all")

### CxG.print_examples()

The *print_examples()* function creates a .txt file with a list of constructions in the loaded model with examples from given data.

	print_examples(self, grammar, input_file, n = 50, output = False, send_back = False)

Which takes the following parameters:

	grammar (str or CxG.grammar)	Type of grammar to examine ("lex", "syn", "full", "all")
							Alternatively, grammars can be specified with:
							'C2xG.{type}_grammar.loc[:,"Chunk"].values'
	input_file (str)			A filename or list of filenames to be parsed, sourced from 'in' directory
	n (int)				Limit examples per construction, 50 by default
	output (bool)			Print examples in console, False by default
	send_back (bool)			Return examples as variable, False by default

## Usage: Learning
----------------------------

### CxG.learn()

The *learn()* function creates and returns new grammar models from the ground-up using input data. 

It returns dataframes with lexical, syntactic, and full grammars.

	learn(self, input_data, npmi_threshold = 0.75, starting_index = 0, min_count = None, 
		max_vocab = None, cbow_range = False, sg_range = False, get_examples = True, 
		increments = 50000, learning_rounds = 20, forgetting_rounds = 40, cluster_only = False)

Which takes the following parameters:

	input_data (str or list of str) 	A filename or list of filenames to be parsed, sourced from 'in' directory
	npmi_threshold (int)		Normalised pointwise mutual information threshold value for use with 'gensim.Phrases', 0.75 by default
							for more information, see: https://radimrehurek.com/gensim/models/phrases.html
	starting_index (int)		Index in input to begin learning, if not the beginning, 0 by default
	min_count (int)			Minimum ngram token count to maintain. If none, derived from 'max_words' during initialisation
	max_vocab (int)			Maximum vocabulary size, no maximum by default
	cbow_range (int)			Maximum cbow clusters, 250 by default
	sg_range (int) 			Maximum skip-gram clusters, 2500 by default
	get_examples (bool)		If true, also run 'get_examples'. Use 'help(C2xG.get_examples)' for more.
	increments (int)			Defines both the number of words to discard and where to stop, 50000 by default
	learning_rounds (int) 		Number of learning rounds to build/refine vocabulary, 20 by default
	forgetting_rounds (int) 	Number of forgetting rounds to prune vocabulary, 40 by default
	cluster_only (bool) 		Only use clusters from embedding models, False by default

Each learning fold consists of three tasks: (i) estimating association values from background data; this requires a large amount of data (e.g., 20 files); (ii) extracting candidate constructions; this requires a moderate amount of data (e.g., 5 files); (iii) evaluating potential grammars against a test set; this requires a small amount of data (e.g., 1 file or 10 mil words).

The freq_threshold is used to control the number of potential constructions to consider. It can be set at 20. The turn limit controls how far the search process can go. It can be set at 10.

### CxG.learn_embeddings()

The *learn_embeddings()* function creates new cbow and skip-gram embeddings using input data.

	learn_embeddings(self, input_data, name="embeddings")

Which takes the following parameters:

	input_data (str or list of str)	A filename or list of filenames to be parsed, sourced from 'in' directory
	name (str) 					The nickname to use when saving models, 'embeddings' by default.

