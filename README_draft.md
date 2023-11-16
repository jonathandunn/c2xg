## c2xg 2.0
=============

Computational Construction Grammar, or *c2xg*, is a Python package for learning and working with construction grammars. 

Why CxG? Constructions are grammatical entities that support a straight-forward quantification of linguistic structure.

This package currently support 18 languages: English (eng), Arabic (ara), Danish (dan), German (deu), Greek (ell), Farsi (fas), Finnish (fin), French (fra), Hindi (hin), Indonesian (ind), Italian (ita), Dutch (nld), Polish (pol), Portuguese (por), Russian (rus), Spanish (spa), Swedish (swe), Turkish (tur)

### Further Documentation
-----------------------

More detailed linguistic documentation is available in the draft book *Computational Construction Grammar: A Usage-Based Approach* available at https://www.jdunn.name/cxg

Usage examples in a working environment are available at https://doi.org/10.24433/CO.9944630.v1

Detailed descriptions of each pre-trained grammar are available at https://doi.org/10.17605/OSF.IO/SA6R3

### Installation
--------------

To install the full package and its dependencies with *pip*, use:

	pip install git+https://github.com/jonathandunn/c2xg.git

### Download Models
--------------------

A number of pre-trained grammar models are available for download and use with C2xG.

Note: The *download_model()* function must be used to install pre-trained grammars.

	from c2xg import download_model

Models can be downloaded as follows: 

	download_model(model = False, data_dir = None, out_dir = None)

These parameters are as follows:

	model (str)	Name of a pre-trained grammar model or its shortcut
	data_dir (str)	Main data directory, creates 'data' in current directory if none given
	out_dir (str)	Output data directory, creates 'OUT' in main data directory if none given

For example, to download the model pre-trained on an English blogs corpus, use:

	download_model(model = "BL", data_dir = "CxG_data")

Models have been pre-trained for each of the languages above, with additional English models trained on separate corpora. To see the models available, please view the following lists:

<details><summary>View: General Models</summary>
	
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

<details><summary>View: English Corpus Models</summary>

	"BL": "cxg_corpus_blogs_final_v2.eng.1000k_words.model.zip",
	"NC": "cxg_corpus_comments_final_v2.eng.1000k_words.model.zip",
	"EU": "cxg_corpus_eu_final_v2.eng.1000k_words.model.zip",
	"PG": "cxg_corpus_pg_final_v2.eng.1000k_words.model.zip",
	"PR": "cxg_corpus_reviews_final_v2.eng.1000k_words.model.zip",
	"OS": "cxg_corpus_subs_final_v2.eng.1000k_words.model.zip",
	"TW": "cxg_corpus_tw_final_v2.eng.1000k_words.model.zip",
	"WK": "cxg_corpus_wiki_final_v2.eng.1000k_words.model.zip",

</details>

### Usage: Initialising
---------------------

To use C2xG, it must first be initialised with the following command.

_note:_ This process may take a few minutes depending on your machine. 

	from c2xg import C2xG
	CxG = C2xG(model = "BL") 

Initialisation accepts the following parameters:

	model (str)			Pre-trained grammar file name in the out directory, or corresponding shortcut
						See "Download Models" and "Further Documentation" for more
	data_dir (str)			Working directory, creates 'data' in current directory if none given
	in_dir (str)			Input directory name, creates 'IN' in 'data_dir' if none given
	out_dir (str)			Output directory name, creates 'OUT' in 'data_dir' if none given
	language (str)			Language for file names, default 'N/A'
	nickname (str) 			Nickname for file names, default 'cxg'
	max_sentence_length (int) 	Cutoff length for loading a given sentence, 50 by default
	normalization (bool)		Normalize frequency by ngram type and frequency strata, True by default
	max_words (bool) 		Limit the number of words when reading input data, False by default
	cbow_file (str)			Name of cbow file to load or create
	sg_file (str) 			Name of skip-gram file to load or create

For example, to initialise an instance of C2xG with the English Wiki corpus in the folder "CxG_data", use:

	CxG_wiki = C2xG(model = "WK", data_dir = "CxG_data")

### Usage: Grammar Parsing
---------------------------

***CxG.parse()***

The *parse()* function takes a text, file name, or list of file names and returns a sparse matrix with construction frequencies for each line in the text. 

	CxG.parse(self, input, input_type = "files", mode = "syn", third_order = False)

Which accepts the following parameters: 

	input (str or list of str)	A filename or list of filenames to be parsed, sourced from 'in' directory
	input_type (str)		"files" if input contains filenames or "lines" if input contains data
	mode (str, default "syn")	Type(s) of constructions to be parsed ("lex", "syn", "full", or "all")
	third_order (bool)		Whether third-order constructions are used, False by default

For example, to take a text file in the 'in' directory and parse it for lexical constructions, use:

	parse_lex = CxG.parse(input = "my_sentences.txt", input_type = "files", mode = "lex")

***CxG.parse_types()***

The *parse_types()* function takes a text, file name, or list of file names and returns a sparse matrix with construction type frequencies over all inputs. 

	CxG.parse_types(self, input, input_type = "files", mode = "syn", third_order = False)

Which takes the following parameters:

	input (str or list of str)	A filename or list of filenames to be parsed, sourced from 'in' directory
	input_type (str)		"files" if input contains filenames or "lines" if input contains data
	mode (str, default "syn")	Type(s) of representations to be parsed ("lex", "syn", "full", or "all")
	third_order (bool)		Whether third-order constructions are used, False by default

For example, to take a text file in the 'in' directory and parse it for all constructions types, use:

	parse_all_types = CxG.parse_types(input = "my_sentences.txt", input_type = "files", mode = "all")

### Usage: Grammar Analysis
----------------------------

***CxG.get_type_token_ratio()***

The *get_type_token_ratio()* method takes a text, file name, or list of file names and returns the following: the type and token counts for all inputs, and the ratio thereof.

	get_type_token_ratio(self, input_data, input_type, mode = "syn", third_order = False)

Which takes the following parameters:

	input (str or list of str)	A filename or list of filenames to be parsed, sourced from 'in' directory
	input_type (str)		"files" if input contains filenames or "lines" if input contains data
	mode (str, default "syn")	Type(s) of representations to be parsed ("lex", "syn", "full", or "all")
	third_order (bool)		Whether third-order constructions are used, False by default

 For example, to take a text file in the 'in' directory and obtain the lexical construction type/token counts and type-token ratio, use:

	get_ratio = CxG.get_type_token_ratio(input = "my_sentences.txt", input_type = "files", mode = "lex")

***CxG.get_association()***

The *get_association()* function returns a dataframe with assocation measures for word pairs in the input data. This dataframe includes the words in the pair, their Delta-P scores in both left and right directions, the difference in scores, the maximum score, and the frequency within the data. 

_note:_ For more on these measures, see https://arxiv.org/abs/2104.01297

	get_association(self, freq_threshold = 1, normalization = True, grammar_type = "full", lex_only = False, data = False)

Which takes the following parameters:

	freq_threshold (int) 		Only consider bigrams above this frequency threshold, 1 by default
	normalization (bool) 		Normalize frequency by ngram type and frequency strata, True by default
	grammar_type (str)		Suffix for pickle file name for file containing discounts, default "full"
	lex_only (bool)			Limit n-grams examined to lexical entries only, False by default
	data (str or list of str)	A filename or list of filenames to be parsed, sourced from 'in' directory

For example, to examples word pairs that occur at least ten times, use:

	delta_association = CxG.get_association(input = "my_sentences.txt", freq_threshold = 10)

### Usage: Grammar Exploration
-------------------------------

***CxG.print_constructions()***

The *print_constructions()* function prints, returns, and creates the file "temp.txt" in the 'out' directory containing a list of constructions of the selected type and their IDs from the initialised model.

	print_constructions(self, mode="lex")

Which takes the following parameters:

	mode (str, default "lex")	Type(s) of representations to be examined ("lex", "syn", "full", or "all")

For example, to print all constructions in the model loaded, use

	CxG_wiki = C2xG(model = "WK", data_dir = "CxG_data") # initialise model, as above
 	all_wiki_constructions = CxG.print_constructions(mode = "all)

***CxG.print_examples()***

The *print_examples()* function creates the file "temp.txt" with a list of constructions in the 'out' directory containing a list of constructions of the selected type and their IDs from the initialised model (or grammar) with examples from the selected data.

	print_examples(self, grammar, input_file, n = 50, output = False, send_back = False)

Which takes the following parameters:

	grammar (str or CxG.grammar)	Type of grammar to examine ("lex", "syn", "full", "all")
						Alternatively, grammars can be specified with:
						C2xG.{type}_grammar.loc[:,"Chunk"].values'
	input_file (str)		A filename or list of filenames to be parsed, sourced from 'in' directory
	n (int)				Limit examples per construction, 50 by default
	output (bool)			Print examples in console, False by default
	send_back (bool)		Return examples as variable, False by default

For example, to print and return 10 examples of each syntactic construction in the chosen data, use:

	syn_examples = CxG.print_examples(grammar = "syn", input_file = "my_sentences.txt", n = 10,
 					output = True, send_back = true)

### Usage: Learning
----------------------------

***CxG.learn()***

The *learn()* function creates and returns new grammar models like those obtained using the *download_model()* function, using a given data input. This function returns three separate dataframes for lexical, syntactic, and full grammars.

_note:_ This function is likely to take some time, especially with more learning/forgetting rounds. 

	learn(self, input_data, npmi_threshold = 0.75, starting_index = 0, min_count = None, 
		max_vocab = None, cbow_range = False, sg_range = False, get_examples = True, 
		increments = 50000, learning_rounds = 20, forgetting_rounds = 40, cluster_only = False)

Which takes the following parameters:

	input_data (str or list of str) 	A filename or list of filenames to be parsed, sourced from 'in' directory
	npmi_threshold (int)		Normalised pointwise mutual information threshold value, 0.75 by default. 
 						For use with 'gensim.Phrases', for more information see:
						https://radimrehurek.com/gensim/models/phrases.html
	starting_index (int)		Index in input to begin learning, if not the beginning, 0 by default
	min_count (int)			Minimum ngram token count to maintain. If none, derived from 'max_words' during initialisation
	max_vocab (int)			Maximum vocabulary size, no maximum by default
	cbow_range (int)		Maximum cbow clusters, 250 by default
	sg_range (int) 			Maximum skip-gram clusters, 2500 by default
	get_examples (bool)		If true, also run 'get_examples'. Use 'help(C2xG.get_examples)' for more.
	increments (int)		Defines both the number of words to discard and where to stop, 50000 by default
	learning_rounds (int) 		Number of learning rounds to build/refine vocabulary, 20 by default
	forgetting_rounds (int) 	Number of forgetting rounds to prune vocabulary, 40 by default
	cluster_only (bool) 		Only use clusters from embedding models, False by default

Each learning fold consists of three tasks: (i) estimating association values from background data; this requires a large amount of data (e.g., 20 files); (ii) extracting candidate constructions; this requires a moderate amount of data (e.g., 5 files); (iii) evaluating potential grammars against a test set; this requires a small amount of data (e.g., 1 file or 10 mil words).

The freq_threshold is used to control the number of potential constructions to consider. It can be set at 20. The turn limit controls how far the search process can go. It can be set at 10.

For example, to create a simple model with only two rounds, eight rounds of forgetting, a cbow range of 50, and a skip-gram range of 500, while also getting a list of examles use:

	lex_gram, syn_gram, full_gram = CxG.learn(input_file = "my_sentences.txt", get_examples = True,
 							learning_rounds = 2, forgetting_rounds = 8, 
							cbow_range = 50, sg_range = 500)

***CxG.learn_embeddings()***

The *learn_embeddings()* function creates new cbow and skip-gram embeddings using input data. The _learn()_ function will do this automatically by default, but this function generates them in isolation. 

_note:_ Embeddings are stored in the class as 'self.cbow_model' and 'self.sg_model'.

	learn_embeddings(self, input_data, name="embeddings")

Which takes the following parameters:

	input_data (str or list of str)	A filename or list of filenames to be parsed, sourced from 'in' directory
	name (str) 			The nickname to use when saving models, 'embeddings' by default.

For example, learn embeddings and save them with the nickname "new_embeddings":

	CxG.learn_embeddings(input_file = "my_sentences.txt", name = "new_embeddings)
