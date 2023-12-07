'''
--------------------------------------------------
A small suite of tests to check C2xG functionality
--------------------------------------------------

First, navigate to C2xG working directory. 
	note: this should include a folder, 'data', with 'IN' and 'OUT' therein.

Next, run the following command from command line:

>>> python "c2xg_testing.py"

'''

## initialisation input and testing files
data_folder = "data" # in "./"
model_path = "cxg_corpus_blogs_final_v2.eng.1000k_words.model.zip" # in "./data/OUT"
testing_file = "testing_corpus100.txt" # in "./data/IN", first 20 lines from the blogs corpus
testing_corpus = "testing_corpus5000.txt" # in "./data/IN", first 5000 lines from the blogs corpus

## string input data
testing_lines = [
"surely somebody must ve seen something going on at one time or other",
"if thats the case then i would struggle to find something that is not material",
"this will be a new ongoing feature here at the question",
"his usual tree in the park helped keep away some of the rain but did nothing for the cold",
"this was after gwen and long before j lo so he still had some cred then"]

## import and initialise C2xG
from c2xg import C2xG
test_cxg = C2xG(data_dir = data_folder, model = model_path)

# ---------------------------------------

print("\nTest C2xG Functions\n")

## run a given function with given inputs
def c2xg_test(function,arguments):
	print("\t...attempting '{}()' with arguments: {}".format(function.__name__,arguments))
	try:
		function(**arguments)
	except Exception as e:
		print("\t\tFailure:", e)

# ---------------------------------------

# PARSING

def test_parse():

	## list of arguments to test 'parse()'
	parse_args = [
	{"input" : testing_file},
	{"input" : testing_file, "mode" : "all"},
	{"input" : testing_file, "third_order" : True},
	{"input" : testing_lines, "input_type" : "lines"}]

	## perform tests 
	for arguments in parse_args:
		c2xg_test(test_cxg.parse,arguments)

# ----------

def test_parse_types():

	## list of arguments to test 'parse_types()'
	parse_types_args = [
	{"input" : testing_file},
	{"input" : testing_file, "mode" : "all"},
	{"input" : testing_file, "third_order" : True},
	{"input" : testing_lines, "input_type" : "lines"}]

	## perform tests 
	for arguments in parse_types_args:
		c2xg_test(test_cxg.parse_types,arguments)

# ---------------------------------------

# ANALYSIS

def test_get_type_token_ratio():

	## list of arguments to test 'get_type_token_ratio()'
	get_type_token_ratio_args = [
	{"input_data" : testing_file, "input_type" : "files"},
	{"input_data" : testing_file, "input_type" : "files", "mode" : "all"},
	{"input_data" : testing_file, "input_type" : "files", "third_order" : True},
	{"input_data" : testing_lines, "input_type" : "lines"}]

	## perform tests 
	for arguments in get_type_token_ratio_args:
		c2xg_test(test_cxg.get_type_token_ratio,arguments)

# ----------

def test_get_association():

	## list of arguments to test 'get_association()'
	get_association_args = [
	{"data" : testing_file},
	{"data" : testing_file, "freq_threshold" : 2},
	{"data" : testing_file, "normalization" : False},
	{"data" : testing_file, "grammar_type" : "full"},
	{"data" : testing_file, "lex_only" : True},
	{"data" : testing_lines}]

	## perform tests 
	for arguments in get_association_args:
		c2xg_test(test_cxg.get_association,arguments)

# ---------------------------------------

# GRAMMAR EXPLORATION

def test_print_constructions():

	## list of arguments to test 'print_constructions()'
	print_constructions_args = [
	{},
	{"mode" : "all"}]

	## perform tests 
	for arguments in print_constructions_args:
		c2xg_test(test_cxg.print_constructions,arguments)

# ----------

def test_print_examples():

	## acquire test grammar
	syn_grammar = test_cxg.syn_grammar.loc[:,"Chunk"].values
	
	## list of arguments to test 'print_examples()'
	print_examples_args = [
	{"grammar" : syn_grammar, "input_file" : testing_file, "n" : 10}]

	## perform tests 
	for arguments in print_examples_args:
		c2xg_test(test_cxg.print_examples,arguments)

# ---------------------------------------

# LEARNING

def test_learn(full_test=False):

	## list of arguments to test 'learn()' with
	if full_test == False:
		learn_args = [
		{"input_data" : testing_corpus, "learning_rounds" : 4, "forgetting_rounds" : 8}]
	else:
		learn_args = [
		{"input_data" : testing_corpus, "learning_rounds" : 4, "forgetting_rounds" : 8},
		{"input_data" : testing_corpus, "learning_rounds" : 4, "forgetting_rounds" : 8, "npmi_threshold" : 0.8},
		{"input_data" : testing_corpus, "learning_rounds" : 4, "forgetting_rounds" : 8, "starting_index" : 100},
		{"input_data" : testing_corpus, "learning_rounds" : 4, "forgetting_rounds" : 8, "min_count" : 100},
		{"input_data" : testing_corpus, "learning_rounds" : 4, "forgetting_rounds" : 8, "max_vocab" : 1000},
		{"input_data" : testing_corpus, "learning_rounds" : 4, "forgetting_rounds" : 8, "cbow_range" : 200},
		{"input_data" : testing_corpus, "learning_rounds" : 4, "forgetting_rounds" : 8, "sg_range" : 2000},
		{"input_data" : testing_corpus, "learning_rounds" : 4, "forgetting_rounds" : 8, "cluster_only" : True}]

	## perform tests 
	for arguments in learn_args:
		c2xg_test(test_cxg.learn,arguments)

# ----------

def test_learn_embeddings():

	## list of arguments to test 'learn_embeddings()' with
	learn_embeddings_args = [
	{"input_data" : testing_corpus}]

	## perform tests 
	for arguments in learn_embeddings_args:
		c2xg_test(test_cxg.learn_embeddings,arguments)

# ---------------------------------------

def main_test():

	## test parsing functions
	test_parse()
	test_parse_types()

	## test analysis functions
	test_get_type_token_ratio()
	test_get_association()

	## test grammar exploration functions
	test_print_constructions()
	test_print_examples()

	## test learning functions
	test_learn()
	# test_learn(test_all=True) # A more comprehensive, time-intensive version
	test_learn_embeddings()

main_test()
