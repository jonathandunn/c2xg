import os
import random
import numpy as np
import pandas as pd
import copy
import operator
import pickle
import codecs
from collections import defaultdict
import multiprocessing as mp
import cytoolz as ct
from functools import partial
from pathlib import Path
from gensim.models.fasttext import load_facebook_model

from .Loader import Loader
from .Parser import Parser
from .Association import Association
from .Candidates import Candidates
from .MDL_Learner import MDL_Learner
from .Parser import parse_examples
from .Word_Classes import Word_Classes
#-------------------------------------------------------------------------------

class C2xG(object):
    
    def __init__(self, data_dir = None, language = "eng", nickname = "cxg", model = None, 
                    normalization = True, max_words = False, fast_parse = False, 
                    cbow_file = "", sg_file = "", workers = 1):
    
        #Initialize
        self.nickname = nickname
        self.workers = workers

        if max_words != False:
            self.nickname += "." + language + "." + str(int(max_words/1000)) + "k_words" 

        print("Current nickname: " + self.nickname)

        self.data_dir = data_dir
        self.language = language

        #Set data location
        if data_dir != None:
            in_dir = os.path.join(data_dir, "IN")
            out_dir = os.path.join(data_dir, "OUT")
        else:
            in_dir = None
            out_dir = None
        
        #Set global variables
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.max_words = max_words
        self.normalization = normalization
        
        #Set embeddings files
        self.cbow_file = os.path.join(self.out_dir, cbow_file)
        self.sg_file = os.path.join(self.out_dir, sg_file)
        
        if os.path.exists(self.cbow_file):
            print("Using for local word embeddings: ", self.cbow_file)
            self.cbow_model = self.load_embeddings(self.cbow_file)
        else:
            self.cbow_model = False
        
        if os.path.exists(self.sg_file):
            print("Using for non-local word embeddings: ", self.sg_file)
            self.sg_model = self.load_embeddings(self.sg_file)
        else:
            self.sg_model = False

        #Initialize modules
        self.Load = Loader(in_dir, out_dir, language = self.language, max_words = max_words, nickname = self.nickname, sg_model = self.sg_model, cbow_model = self.cbow_model, workers = self.workers)
        self.Association = Association(Load = self.Load, nickname = self.nickname)
        self.Parse = Parser(self.Load)
        self.Word_Classes = Word_Classes(self.Load)
        
        #Try to load default or specified model
        if model == "default":
            model = self.language + ".Grammar.v3.p"
            print("Using default grammar")

        #Try to load grammar from file
        if isinstance(model, str):

            try:
                modelname = None
                if os.path.isfile( model ) :
                    modelname = model
                else :
                    modelname = Path(__file__).parent / os.path.join("data", "models", model)

                with open(modelname, "rb") as handle:
                    self.model = pickle.load(handle)
        
            except Exception as e:
                print("Using empty grammar")
                self.model = None
            
        #Take model as input
        elif isinstance(model, list):
            self.model = model

        if fast_parse: 
            self._detail_model() ## self.detailed_model set by this. 
        else: 
            self.detailed_model = None

    #------------------------------------------------------------------
    def load_embeddings(self, model_file):
    
        #Load and prep word embeddings
        if isinstance(model_file, str):
            if os.path.exists(model_file):  
                model = load_facebook_model(model_file)                
                return model     

            else:
                print("Error: model doesn't exist. Use learn_embeddings.")
                print(model_file)
                return None
            
    #-----------------------------------------------------------------
    def learn_embeddings(self, input_data, name="embeddings"):

        print("Starting local embeddings (cbow)")
        self.cbow_model = self.Word_Classes.learn_embeddings(input_data, model_type="cbow", name=name)

        print("Finished with cbow emeddings. Starting sg embeddings")
        self.sg_model = self.Word_Classes.learn_embeddings(input_data, model_type="sg", name=name)
        
    #------------------------------------------------------------------

    def learn(self, input_data, npmi_threshold = 0.75, min_count = None, cbow_range = False, sg_range = False):

        #Adjust min_count to be 1 parts per million using max_words parameter
        if min_count == None:
            if self.max_words == None:
                min_count = 5
            elif self.max_words <= 1000000:
                min_count = int(1000000/self.max_words * 1)
            elif self.max_words > 1000000:
                min_count = int(self.max_words/1000000 * 1)
            print("Setting min_count to 1 parts per million (min_count = " + str(min_count) + ") (max_words = " + str(self.max_words) + ")")

        print("Starting to learn: lexicon")
        self.min_count = min_count
        lexicon, phrases, unique_words = self.Load.get_lexicon(input_data, npmi_threshold, self.min_count)

        n_phrases = len([x for x in lexicon.keys() if " " in x])
        print("Finished with " + str(len(lexicon)-n_phrases) + " words and " + str(n_phrases) + " phrases")

        #Save phrases and lexicon
        self.phrases = phrases
        self.lexicon = lexicon
        
        #Check embeddings
        if self.cbow_model == False and self.sg_model == False:
            self.learn_embeddings(input_data)

        #Check for syntactic clusters and form them if necessary
        cbow_df_file = os.path.join(self.out_dir, self.nickname + ".categories_cbow.csv")
        if not os.path.exists(cbow_df_file):
            print("Starting cbow word categories")
            cbow_df, cbow_mean_dict = self.Word_Classes.learn_categories(self.cbow_model, self.lexicon, unique_words = unique_words, variety = "cbow", top_range = cbow_range)
            cbow_df.to_csv(os.path.join(self.out_dir, self.nickname + ".categories_cbow.csv"), index = False)
            self.Load.save_file(cbow_mean_dict, self.nickname+".categories_cbow.means.p")   
        else:
            cbow_df = pd.read_csv(cbow_df_file)
            cbow_mean_dict = self.Load.load_file(self.nickname+".categories_cbow.means.p")
        
        #Now print syntactic clusters
        print(cbow_df)
        
        #check for semantic clusters and form them in necessary
        sg_df_file = os.path.join(self.out_dir, self.nickname + ".categories_sg.csv")
        if not os.path.exists(sg_df_file):
            print("Starting sg word categories")
            sg_df, sg_mean_dict = self.Word_Classes.learn_categories(self.sg_model, self.lexicon, unique_words = unique_words, variety = "sg", top_range = sg_range)
            sg_df.to_csv(os.path.join(self.out_dir, self.nickname + ".categories_sg.csv"), index = False)
            self.Load.save_file(sg_mean_dict, self.nickname+".categories_sg.means.p")
        else:
            sg_df = pd.read_csv(sg_df_file)
            sg_mean_dict = self.Load.load_file(self.nickname+".categories_sg.means.p")
         
        #Now print semantic clusters
        print(sg_df)
            
        #Add clusters to loader
        self.Load.cbow_centroids = cbow_mean_dict
        self.Load.sg_centroids = sg_mean_dict
        self.Load.add_categories(cbow_df, sg_df)
        
        #Now that we have clusters, enrich input data and save
        if not os.path.exists(os.path.join(self.out_dir, self.nickname+".input_enriched.p")):
            print("Enriching input using syntactic and semantic categories")
            self.data = self.Load.load(input_data)  #Save the enriched data once gotten
            self.Load.save_file(self.data, self.nickname+".input_enriched.p")
        else:
            print("Loading enriched input")
            self.data = self.Load.load_file(self.nickname+".input_enriched.p")
            
        #Get pairwise association with Delta P
        association_file = os.path.join(self.out_dir, self.nickname + ".association.gz")
        if not os.path.exists(association_file):
            association_df = self.get_association(freq_threshold = self.min_count, normalization = self.normalization, lex_only = False)
            association_df.to_csv(association_file, compression = "gzip")
        else:
            association_df = pd.read_csv(association_file, index_col = 0)
            
        #Now print association data
        print(association_df)
        
        #Convert to dict
        self.get_decoder()
        self.assoc_dict = self.get_association_dict(association_df)
        
        #Initialize candidates module
        self.Candidates = Candidates(language = self.language, Load = self.Load, freq_threshold = self.min_count, association_dict = self.assoc_dict)
        
        #Get chunks
        chunks = self.Candidates.get_candidates(self.data)
        
        
        return

    #------------------------------------------------------------------

    def _detail_model(self) : 

        ## Update model so we can access grammar faster ... 
        ## Want to make `if construction[0][1] == unit[construction[0][0]-1]` faster
        ## Dict on construction[0][1] which is self.model[i][0][1] (Call this Y)
        ## BUT unit[ construction[0][0] - 1 ] changes with unit ... 
        ## construction[0][0] values are very limited.  (call this X)
        ## dict[ construction[0][0] ][ construction[0][1] ] = list of constructions
        
        model_expanded = dict()

        X = list( set( [ self.model[i][0][0] for i in range(len(self.model)) ] ) )
        
        for x in X : 
            model_expanded[ x ] = defaultdict( list ) 
            this_x_elems = list()
            for k, elem in enumerate( self.model ) : 
                if elem[0][0] != x : 
                    continue
                elem_trunc = [ i for i in elem if i != (0,0) ]
                model_expanded[ x ][ elem[0][1] ].append( ( elem, elem_trunc, k ) )
        
        self.detailed_model = ( X, model_expanded ) 

    #------------------------------------------------------------------
        
    def parse_return(self, input, mode = "files", workers = None):
            
        #Make sure grammar is loaded
        if self.model == None:
            print("Unable to parse: No grammar model provided.")
            sys.kill()
            
        #Accepts str of filename or list of strs of filenames
        if isinstance(input, str):
            input = [input]
        
        #Text as input
        if mode == "lines":
            lines = self.Parse.parse_idNet(input, self.model, workers, self.detailed_model )
            return np.array(lines)    
                    
        #Filenames as input
        elif mode == "files":
            features = self.Parse.parse_batch(input, self.model, workers, self.detailed_model )
            return np.array(features)

    #-------------------------------------------------------------------------------

    def parse_validate(self, input, workers = 1):
        self.Parse.parse_validate(input, grammar = self.model, workers = workers, detailed_grammar = self.detailed_model)
        
    #-------------------------------------------------------------------------------
    
    def parse_yield(self, input, mode = "files"):

        #Make sure grammar is loaded
        if self.model == None:
            print("Unable to parse: No grammar model provided.")
            sys.kill()
            
        #Accepts str of filename or list of strs in batch/stream modes
        if isinstance(input, str):
            input = [input]
        
        #Filenames as input
        if mode == "files":
            for features in self.Parse.parse_stream(input, self.model, detailed_grammar = self.detailed_model):
                yield np.array(features)

        #Texts as input
        elif mode == "lines":
        
            for line in input:
                line = self.Parse.parse_line_yield(line, self.model, detailed_grammar = self.detailed_model)
                yield np.array(line)            
            
    #-------------------------------------------------------------------------------
    def print_constructions(self):

        return_list = []

        for i in range(len(self.model)):
            
            x = self.model[i]
            printed_examples = []

            #Prune to actual constraints
            x = [y for y in x if y[0] != 0]
            length = len(x)
            construction = self.Encode.decode_construction(x)

            print(i, construction)
            return_list.append(str(i) + ": " + str(construction))

        return return_list
    #-------------------------------------------------------------------------------
    def print_examples(self, input_file, output_file, n):

        #Read and write in the default data directories
        output_file = os.path.join(self.out_dir, output_file)

        #Save the pre-processed lines, to save time later
        line_list = []
        for line, encoding in self.Encode.load_examples(input_file):
            line_list.append([line, encoding])

        with codecs.open(output_file, "w", encoding = "utf-8") as fw:
            for i in range(len(self.model)):
            
                x = self.model[i]
                printed_examples = []

                #Prune to actual constraints
                x = [y for y in x if y[0] != 0]
                length = len(x)
                construction = self.Encode.decode_construction(x)

                print(i, construction)
                fw.write(str(i) + "\t")
                fw.write(construction)
                fw.write("\n")

                #Track how many examples have been found
                counter = 0

                for line, encoding in line_list:

                    construction_thing, indexes, matches = parse_examples(x, encoding)

                    if matches > 0:
                        for index in indexes:
                            
                            text = line.split()[index:index+length]

                            if text not in printed_examples:
                                counter += 1
                                printed_examples.append(text)
                                fw.write("\t" + str(counter) + "\t" + str(text) + "\n")
                    
                    #Stop looking for examples at threshold
                    if counter > n:
                        break
                
                #End of examples for this construction
                fw.write("\n\n")
    #-------------------------------------------------------------------------------

    def get_association(self, freq_threshold = 1, normalization = True, lex_only = False):
        
        #For smoothing, get discounts by constraint type
        if self.normalization == True:
            discount_dict = self.Association.find_discounts(self.data)
            self.Load.save_file(discount_dict, self.nickname+".discounts.p")
            print(discount_dict)
            print("Discounts ", self.nickname)

        else:
            discount_dict = False

        ngrams = self.Association.find_ngrams(self.data, lex_only = False, n_gram_threshold = 1)
        association_dict = self.Association.calculate_association(ngrams = ngrams, normalization = self.normalization, discount_dict = discount_dict)
        
        #Reduce to bigrams
        keepable = lambda x: len(x) > 1
        all_ngrams = ct.keyfilter(keepable, association_dict)

        #Convert to readable CSV
        pairs = []
        for pair in all_ngrams:
            
            val1 = self.Load.decode(pair[0])
            val2 = self.Load.decode(pair[1])

            if val1 != "UNK" and val2 != "UNK":
                maximum = max(association_dict[pair]["LR"], association_dict[pair]["RL"])
                difference = association_dict[pair]["LR"] - association_dict[pair]["RL"]
                pairs.append([val1, val2, maximum, difference, association_dict[pair]["LR"], association_dict[pair]["RL"], association_dict[pair]["Freq"]])

        #Make dataframe
        df = pd.DataFrame(pairs, columns = ["Word1", "Word2", "Max", "Difference", "LR", "RL", "Freq"])
        df = df.sort_values("Max", ascending = False)

        return df
 
    #-------------------------------------------------------------------------------
    
    def get_decoder(self):
    
        #Get reverse dictionaries for decoding
        self.cbow_dict = {val: key for (key, val) in self.Load.cbow_names.items()}
        self.sg_dict = {val: key for (key, val) in self.Load.sg_names.items()}
        self.word_dict = {val: key for (key, val) in self.Load.indexes.items()}
        
    #-------------------------------------------------------------------------------

    def get_association_dict(self, df):
    
        assoc_dict = {}
        
        #Process dataframe by row
        for row in df.itertuples():
            word1 = row[1]
            word2 = row[2]
            maximum = row[3]
            difference = row[4]
            lr = row[5]
            rl = row[6]
            freq = row[7]
            
            #Get categories instead of names, 1
            if "sem: " in word1:
                word1 = (3, self.sg_dict[word1.replace("sem: ", "")])
            elif "syn: " in word1:
                word1 = (2, self.cbow_dict[word1.replace("syn: ", "")])
            else:
                word1 = (1, self.word_dict[word1])
            
            #Get categories instead of names, 2
            if "sem: " in word2:
                word2 = (3, self.sg_dict[word2.replace("sem: ", "")])
            elif "syn: " in word2:
                word2 = (2, self.cbow_dict[word2.replace("syn: ", "")])
            else:
                word2 = (1, self.word_dict[word2])
            
            if word1 not in assoc_dict:
                assoc_dict[word1] = {}
                
            if word2 not in assoc_dict[word1]:
                assoc_dict[word1][word2] = {}
                assoc_dict[word1][word2]["Max"] = maximum
                assoc_dict[word1][word2]["Difference"] = difference
                assoc_dict[word1][word2]["LR"] = lr
                assoc_dict[word1][word2]["RL"] = rl
                assoc_dict[word1][word2]["Frequency"] = freq
     
        return assoc_dict
  
    #-------------------------------------------------------------------------------
    
    def fuzzy_jaccard(self, grammar1, grammar2, threshold = 0.70, workers = 2):

        umbrella = set(grammar1 + grammar2)
        
        #First grammar
        pool_instance = mp.Pool(processes = workers, maxtasksperchild = None)
        matches1 = pool_instance.map(partial(self.fuzzy_match, grammar = grammar1, threshold = threshold), umbrella, chunksize = 100)
        pool_instance.close()
        pool_instance.join()
            
        #Second gammar
        pool_instance = mp.Pool(processes = workers, maxtasksperchild = None)
        matches2 = pool_instance.map(partial(self.fuzzy_match, grammar = grammar2, threshold = threshold), umbrella, chunksize = 100)
        pool_instance.close()
        pool_instance.join()
                
        result = 1 - jaccard(matches1, matches2)

        return result

    #-----------------------------------------------
    
    def fuzzy_match(self, construction, grammar, threshold = 0.70):

        match = 0
            
        #Check for exact match
        if construction in grammar:
            match = 1
            
        #Or fall back to highest overlap
        else:

            for u_construction in grammar:
                
                s = difflib.SequenceMatcher(None, construction, u_construction)
                length = max(len(construction), len(u_construction))
                overlap = sum([x[2] for x in s.get_matching_blocks()]) / float(length)
                    
                if overlap >= threshold:
                    match = 1
                    break
                    
        return match

    #-----------------------------------------------    

    def get_mdl(self, candidates, file, workers = 2, freq_threshold = -1):

        result = eval_mdl([file], 
                    workers = workers, 
                    candidates = candidates, 
                    Load = self.Load, 
                    Encode = self.Encode, 
                    Parse = self.Parse, 
                    freq_threshold = freq_threshold, 
                    report = True
                    )

        return result
        
    #-----------------------------------------------    
    
    def forget_constructions(self, grammar, datasets, workers = None, threshold = 1, adjustment = 0.25, increment_size = 100000):

        round = 0
        weights = [1 for x in range(len(grammar))]
        
        for i in range(20):
            
            print(round, len(grammar))
            round += 1
                
            for i in range(len(datasets)):
            
                dataset = datasets[i]
            
                data_parse, data_keep = self.step_data(dataset, increment_size)
                datasets[i] = data_keep
                
                if len(dataset) > 25:
                    self.model = grammar
                    self._detail_model()
                    vector = np.array(self.parse_return(data_parse, mode = "lines"))
                    vector = np.sum(vector, axis = 0)
                    weights = [1 if vector[i] > threshold else weights[i]-adjustment for i in range(len(weights))]
                    
            grammar = [grammar[i] for i in range(len(grammar)) if weights[i] >= 0.0001]
            weights = [weights[i] for i in range(len(weights)) if weights[i] >= 0.0001]
                
        return grammar
    #-----------------------------------------------