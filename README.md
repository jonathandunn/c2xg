
Introduction
=============

C2xG (for Computational Construction Grammar) is two things: 

(1) A theory of linguistic structure that learns CCxG-like grammatical representations from observed unannotated text data

(2) An implemented grammar induction algorithm for the unsupervised learning of construction-based representations along with tools for vectorizing these representations for computational tasks

Installation
--------------

Installation is not currently automated with either PyPi or Anaconda installation processes.


Directory Structure
------------------------

The C2xG directory is the container for all GitHub material for the C2xG package.

It contains the following folders:

	docs			:Contains general documentation and published articles describing the algorithms
	c2xg			:Contains the codebase for the C2xG package as well as resources necessary for running
	utilities		:Contains stand-alone scripts helpful for various related tasks
	
All input data and temporary files are kept in a separate, not-Github indexed location, specified via parameter.

		
Environment and Dependencies
----------------------------------

This package is meant to run in Python 3.5 with a number of dependencies.
The easiest way to maintain the necessary environment is to use Anaconda Python (https://www.continuum.io/downloads)

Dependencies:
		
	Python 3.5
	cytoolz 0.7.4
	gensim 0.12.2
	matplotlib 1.5.0
	numexpr 2.5
	numpy 1.10.4
	pandas 0.18.0
	requests 2.9.1
	scipy 0.16.0
	sklearn 0.17

API
====

* learn_dictionary
* learn_constituents
* learn_candidates
* learn_constructions
* learn_usage
* learn_rdrpos_model
* annotate_pos
* *annotate_constituents (pending)*
* *annotate_constructions (pending)*
* examples_constituents
* examples_constructions
* evaluate_grammar
* extract_vectors


learn_dictionary
-----------------------
		

**Description**
		
Takes as input unannotated sentence-divided text. Provides as output a dictionary in which lexical items are assigned to semantic clusters, using GenSim's implementation of word2vec.
			
**Parameters**
		
	nickname			str: Base name for created resources
	input_directory		str: Path to directory containing corpora in possibly many files
	output_directory	str: Directory for saving models to
	min_threshold		str: Minimum frequency for including a lexical item in the dictionary
	num_dimensions		str: Number of lexical dimensions to use for GenSim
	num_clusters		str: Number of clusters for the final dictionary 
	number_of_cpus		[Optional] str: Number of processes to use 	

learn_constituents
-----------------------

**Description**
	
Takes as input a parameter file, a semantic dictionary, and a set of input texts. Provides as output a Grammar file for later scripts that contains the resources necessary for candidate extraction.
		
This function saves its intermediate data to file because it needs to process more data than can usually fit into memory.
		
This function performs the following tasks:
	
	(1) If necessary, calls RDRPOs or Stanford CoreNLP for POS-tagging.
	(2) Loads the semantic dictionary and creates unit indexes.
	(3) Loads pos-tagged files into categorical pandas matrix.
	(4) Learns phrase structure rules for the dataset.
	(5)	Writes a grammar file containing phrase structure rules, etc as a ".Constituents.model" file
	
**Parameters**

	input_folder						str: Folder containing input files
	output_folder						str: Folder for placing output files
	emoji_file							str: Path to file containing emoji dictionary
	input_files							list: List of strings containing input filenames (no path)
	semantic_dictionary_file			str: Path to file containing semantic dictionary
	frequency_threshold_individual		int: Minimum threshold for including individual units in representation
	illegal_pos							list: List of strings of pos tags to be discarded; dependent on pos system
	phrase_structure_ngram_length		int: N-gram range to use when determining phrase structure rules
	significance						float: Significance value for two-tailed t-test when determining head status
	independence_threshold				int: Number of co-occurrences required to consider a head independent
	data_file_grammar					str: Filename for output model file
	constituent_threshold				float: Frequency threshold for phrase structure rules, formulated as [Mean + (Std.Dev * threshold)]
	annotate_pos						[Optional] True/False: Whether or not input files need to be pos-tagged
	encoding_type						[Optional] str: For example, "utf-8"
	docs_per_file						[Optional] int: Number of documents / sentences to include for each CoNLL file, if pos_annotating
	settings_dictionary					[Optional] dict: Dictionary containing parameters for pos-tagging
	number_of_cpus_annotate 			[Optional] int: Number of processes for pos-tagging
	number_of_cpus_prepare				[Optional] int: Number of processes for learning phrase structure rules
	delete_temp							[Optional] True/False: Whether or not to delete unnecessary temp files when finished with them
	debug								[Optional] True/False: Whether or not to write debug information (e.g., unit indexes and phrase structure rules)
	debug_file							[Optional] str: Base filename for debug files; required if debug == True			
	
learn_candidates
------------------------

**Description**

Takes as input a parameter file and a grammar file from learn_constituents. Provides as output a candidate file for each input text. This candidate file contains the grammar and all other data necessary for merging candidates across a large number of separate files.
		
This function is ideally multi-processed on a large number of short files, so that it never deals with more data than fits into memory.
		
This script performs the following tasks:
	
	(1) If necessary, calls RDRPOS or Stanford CoreNLP for POS-tagging.
	(2) Loads necessary data from the input grammar file.
	(3)	Loads the pos-tagged file into categorical pandas matrix.
	(4) Uses phrase structure rules to reduce complex constituents.
	(5) Search for candidate constructions and keep count.
	(6) Save candidates and other data to output file.
			
This function works on large numbers of smaller texts and produces files that are later	merged. This is because it requires more data than can fit into memory.

**Parameters**

	input_files										list: List of strings of input filenames, without path
	input_folder									str: Path to input folder
	output_folder									str: Path to output folder
	data_file_grammar								str: Filename for ".Constituent.model" file used for extraction
	max_construction_length							int: Max units for searching for constructions; search space grows quickly
	frequency_threshold_constructions_perfile		int: Frequency threshold per file for constructions to be stored
	number_of_cpus_processing						[Optional] int: Number of processes to use for extraction
	annotate_pos									[Optional] True/False: Whether or not input files need to annotate_pos
	encoding_type									[Optional] str: For example, "utf-8"
	annotation_types								[Optional] list: List of strings containing annotation types -> ["Lem", "Pos", "Cat"]
	settings_dictionary								[Optional] dict: Dictionary of settings for pos_tagger; required if annotating pos
	docs_per_file									[Optional] int: Number of lines / docs to include in each CoNLL file, if annotating pos
	delete_temp										[Optional] True/False: Whether or not to remove temp files when finished with them

learn_constructions
---------------------

**Description**
	
Takes as input a list of candidate files from learn_candidates. utputs a construction grammar model that can be used for feature extraction.
		
This function performs the following tasks:
	
	(1) Merges list of candidate files and checks for consistent grammar models.
	    Multi-processing not possible here; pre-merge candidates if necessary using utilities.
	(2) Builds association strength measures for all candidates.
	(3) Prunes candidates to produce the optimum grammar.
	(4) Saves grammar as a model containing all necessary data for feature extraction.
	
**Parameters**

	input_folder						str: Path to folder containing input files
	output_folder						str: Path to folder for output files
	output_files						list: List of strings of filenames (with path) containing candidates from learn_candidates
	frequency_threshold_constructions	int: Frequency required to consider a candidate construction further
	max_construction_length				int: Maximum number of units allowed in a construction; search space grows quickly
	pairwise_threshold_lr				float: Value for pruning weak links in candidates, formulated as [Mean + (Std.Dev * threshold)]
	pairwise_threshold_rl				float: Value for pruning weak links in candidates, formulated as [Mean + (Std.Dev * threshold)]
	nickname							str: Base filename for saving output model, vectors, and csv files
	encoding_type						[Optional] str: For example, "utf-8"
	annotation_types					[Optional] list: List of strings of allowed annotation types -> ["Lem", "Pos", "Cat"]
	number_of_cpus_pruning				[Optional] int: Number of processes to used whenever multi-processing is possible (e.g., not for merging)
	
learn_usage
----------------------

**Description**

Takes as input a grammar model from learn_constructions and a list of files. Outputs a single row containing expected (average) frequencies for each feature. Saves a grammar and usage model for feature extraction.
		
	
extract_vectors
-----------------------

**Description**

Takes as input a grammar model and a list of input files. Outputs vectors of relative frequency for each construction and unit in files, as a pandas DataFrame saved in a compressed HDF5 file.
		
This function performs the following tasks:
	
	(1) Extracts and counts all constructions and individual units.
	(2) Converts frequency to relative frequency for document length.
	(3) Adjusts frequencies relative to centroid (expected frequencies).
	(4) Saves feature vector.


evaluate_grammar
---------------------

**Description**

Takes as input a grammar model and a list of test files. Outputs coverage metrics: how much of the test set is represented by the grammar?
	
This function performs the following tasks:
			
	(1) Creates feature vectors for all test files.
	(2) Creates DataFrame with Number of Features and Number of Words for each row
	(3) Produces visualization of coverage and its relation to text length
			

examples_constituents
---------------------
		
**Description**

Takes as input a grammar model a list of test files. Outputs observed instances of each constituent in the grammar.
		
This function performs the following tasks:
	(1) Extracts instances of constituents from the grammar in the test files.
	(2) Prints instances of constituents
			

examples_constructions
---------------------
		
**Description**

Takes as input a grammar model a list of test files. Outputs observed instances of each construction in the grammar. 

This function performs the following tasks:
			
	(1) Extracts instances of constituents from the grammar in the test files.
	(2) Prints instances of constituents


learn_rdrpos_model
------------------

**Description**

Wrapper updated for Python 3 for running and testing RDRPOSTagger models.
	
Command-Line Usage
==================

Each function in the API can be called or run from the command-line. If run from the command line, each script requires a command-line specified parameter file that contains the input variables. These parameter files are kept in the folder "files_parameters". These parameter files are imported using "files_parameters.FILENAME". For example: 

	python learn_constituents.py files_parameters.learning_ukwac
\*Note that the extension ".py" is excluded from the command.
	
More information about the parameter files is found in the file "model_parameters.py".
	

Creating a Pipeline
======================

This section describes how to use these scripts to create a construction grammar and to create vectors of relative frequency adjusted for expected frequencies.

	(1) Create semantic dictionary:
		(a) Use learn_dictionary.py in "C2xG" folder
		(b) Dictionary is in the form <WORD>,<CATEGORY> with one word per line
		(c) Save this dictionary in files_data folder
		
	(2) Get unit indexes and learn phrase structure rules:
		(a) Use learn_constituents.py in "C2xG" folder
		(b) This step saves temporary files to disk during processing, which can be deleted after use
		(c) Outputs a grammar file containing indexes and phrase structure rules (.Grammar)
		
	(3) Get candidates and their frequencies:
		(a) Use learn_candidates.py in "C2xG" folder
		(b) This step multi-processes by file
		(c) Files should stay within-memory, but otherwise faster with more text per file
		(d) This step does not save temporary files to disk
		(e) Output is an index of candidates and their frequency
		
	(4) Merge candidates across files and select optimum representations
		(a) Use learn_constructions.py in "C2xG" folder
		(b) Takes a list of output files from (3) and merges them
		(c) Calculates association measures
		(d) Prunes using calculated association measures
		(e) Outputs a model containing the previous grammar and the selected constructions(.Constructions)
		
	(5) Get expected frequencies for feature vectors
		(a) Use learn_usage.py in "C2xG" folder
		(b) Usually use same corpus as for learning constructions
		(c) Finds the average frequency of each construction per text
		(d) Output is a model that includes these average frequencies as a measure of relative usage (.Usage)
		
	Steps (1) through (5) are learning tasks to produce a model of constructions and their usage. 
	As such, they need to be performed only once to produce a model of grammar and usage. 
	Production with pre-made models begins with step (6).
		
	Create vectors (optionally of difference from expected frequencies)
		(a) Use extract_vectors.py in "C2xG" folder
		(b) Use new corpus of interest
		(c) Provide model of grammar and/or usage
		(d) Output is an HDF5 file containing a sparse, compressed pandas DataFrame
		
	Evaluate grammar coverage
		(a) Use evaluate_grammar.py in "C2xG" folder
		(b) Use test set distinct from learning corpus
		(c) Output is a cumulative histogram of per-sentence coverage
		(d) To evaluate stability, use scripts in "Utilities" folder
	
	See examples of constituents
		(a) Use examples_constituents.py in "C2xG" folder
		(b) Outputs lists of predicted constituents for each head-type with sentence for context
		
	See examples of constructions
		(a) Use examples_constructions.py in "C2xG"folder
		(b) Outputs list of predicted constructions with predicted / observed constructs for each representation		
	
	Note: annotate_constituents and annotate_constructions not currently implemented. Focus is on vector representations.	

Input Formats
===================

This section describes the input formats for the different components.

(1) Creating Semantic Dictionary

	Input: Unannotated text, one sentence per line
	
(2) Creating Models of Grammar and Usage
	
	Input: Annotated: CoNLL format of tab-separate fields [Word-Form, Lemma, POS, Index]. 
	Use <s:ID> to assign ids to documents.
	
	Input: Unannotated: Plain text with line breaks for documents / sentences as desired. 
	[In both cases, each line is assumed to be a "text" or the containing unit of analysis.]
			
(3) Extracting Feature Vectors
	
	Input with Meta-Data: 		Field:Value,Field:Value\tText
	Input without Meta-Data:	Plain text with line breaks (\n) for documents / sentences.
	
	
Feature Extraction
=========================

Given a model of the construction grammar of a language, the extract_vectors and learn_usage functions convert that grammar into a vector representation of texts or sentences (i.e., one unit per line in the input files). There are two modes and three quantification methods for creating vectors:

	full_scope == True: Construction and lexical features (from allowed word list)
	full_scope == False: Only lexical features; as a baseline for comparing usefulness of full construction grammar
	
	relative_freq == True: Quantify using the relative frequency of the feature in given sentence or text (as negative logarithms)
	relative_freq == False: Quantify using unadjusted raw frequency of the feature
	use_centroid == True: Extract vectors with centriod normalization learned using learn_usage
	
	Centroid normalization first finds the probability of a given feature in the background corpus. This is stored after running learn_usage in separate centroid_df models for the full grammar and for the lexical-only features. During extraction, if centroids are used for representation, this is converted into negative logarithms of the inverted joint probability of each feature occuring as many times as it does in a message.