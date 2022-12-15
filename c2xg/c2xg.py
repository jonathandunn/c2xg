import os
import numpy as np
import pandas as pd
import pickle
import codecs
import multiprocessing as mp
import cytoolz as ct
from functools import partial
from pathlib import Path
from gensim.models.fasttext import load_facebook_model

from .Loader import Loader
from .Parser import Parser
from .Association import Association
from .Candidates import Candidates
from .MDL import Minimum_Description_Length
from .Word_Classes import Word_Classes
#-------------------------------------------------------------------------------

class C2xG(object):
    
    def __init__(self, data_dir = None, language = "eng", nickname = "cxg", model = None, 
                    normalization = True, max_words = False, cbow_file = "", sg_file = ""):
    
        #Initialize
        self.nickname = nickname
        self.workers = mp.cpu_count()

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
        
        #Load existing cbow embeddings
        if os.path.exists(self.cbow_file):
            print("Using for local word embeddings: ", self.cbow_file)
            self.cbow_model = self.load_embeddings(self.cbow_file)
        else:
            self.cbow_model = False
        
        #Load existing sg embeddings
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

    def learn(self, input_data, npmi_threshold = 0.75, min_count = None, cbow_range = False, sg_range = False, get_examples = True):

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
        self.Load.min_count = min_count
        lexicon, phrases, unique_words = self.Load.get_lexicon(input_data, npmi_threshold, self.Load.min_count)

        n_phrases = len([x for x in lexicon.keys() if " " in x])
        print("Finished with " + str(len(lexicon)-n_phrases) + " words and " + str(n_phrases) + " phrases")

        #Save phrases and lexicon
        self.Load.phrases = phrases
        self.Load.lexicon = lexicon
        
        #Check embeddings
        if self.cbow_model == False and self.sg_model == False:
            self.learn_embeddings(input_data)

        #Check for syntactic clusters and form them if necessary
        cbow_df_file = os.path.join(self.out_dir, self.nickname + ".categories_cbow.csv")
        if not os.path.exists(cbow_df_file):
            print("Starting cbow word categories")
            self.Load.cbow_df, self.Load.cbow_mean_dict = self.Word_Classes.learn_categories(self.cbow_model, self.Load.lexicon, unique_words = unique_words, variety = "cbow", top_range = cbow_range)
            self.Load.cbow_df.to_csv(os.path.join(self.out_dir, self.nickname + ".categories_cbow.csv"), index = False)
            self.Load.save_file(self.Load.cbow_mean_dict, self.nickname+".categories_cbow.means.p")   
        else:
            self.Load.cbow_df = pd.read_csv(cbow_df_file)
            self.Load.cbow_mean_dict = self.Load.load_file(self.nickname+".categories_cbow.means.p")
        
        #Now print syntactic clusters
        print(self.Load.cbow_df)
        
        #check for semantic clusters and form them in necessary
        sg_df_file = os.path.join(self.out_dir, self.nickname + ".categories_sg.csv")
        if not os.path.exists(sg_df_file):
            print("Starting sg word categories")
            self.Load.sg_df, self.Load.sg_mean_dict = self.Word_Classes.learn_categories(self.sg_model, self.Load.lexicon, unique_words = unique_words, variety = "sg", top_range = sg_range)
            self.Load.sg_df.to_csv(os.path.join(self.out_dir, self.nickname + ".categories_sg.csv"), index = False)
            self.Load.save_file(self.Load.sg_mean_dict, self.nickname+".categories_sg.means.p")
        else:
            self.Load.sg_df = pd.read_csv(sg_df_file)
            self.Load.sg_mean_dict = self.Load.load_file(self.nickname+".categories_sg.means.p")
         
        #Now print semantic clusters
        print(self.Load.sg_df)
            
        #Add clusters to loader
        self.Load.cbow_centroids = self.Load.cbow_mean_dict
        self.Load.sg_centroids = self.Load.sg_mean_dict
        self.Load.add_categories(self.Load.cbow_df, self.Load.sg_df)
        
        #Now that we have clusters, enrich input data and save
        if not os.path.exists(os.path.join(self.out_dir, self.nickname+".input_enriched.p")):
            print("Enriching input using syntactic and semantic categories")
            self.Load.data = self.Load.load(input_data)  #Save the enriched data once gotten
            self.Load.save_file(self.Load.data, self.nickname+".input_enriched.p")
        else:
            print("Loading enriched input")
            self.Load.data = self.Load.load_file(self.nickname+".input_enriched.p")
            
        #Get pairwise association with Delta P
        association_file = os.path.join(self.out_dir, self.nickname + ".association.gz")
        if not os.path.exists(association_file):
            self.Load.association_df = self.get_association(freq_threshold = self.Load.min_count, normalization = self.normalization, lex_only = False)
            self.Load.association_df.to_csv(association_file, compression = "gzip")
        else:
            self.Load.association_df = pd.read_csv(association_file, index_col = 0)
            
        #Now print association data
        print(self.Load.association_df)
        
        #Convert to dict
        self.Load.assoc_dict = self.get_association_dict(self.Load.association_df)

        #Set grammar output filenames
        cost_file = os.path.join(self.out_dir, self.nickname + ".grammar_costs.csv")
        slot_cost_file = os.path.join(self.out_dir, self.nickname + ".slot_costs.csv")
        
        #Check if grammar output exists
        if not os.path.exists(cost_file):
        
            #Search for best grammar
            best_delta, best_freq, best_candidates, best_cost, best_cost_df, best_chunk_df, encoding_pruning = self.grid_search()
            print("")
            print("Best Delta Threshold: ", best_delta)
            print("Best Freq Threshold: ", best_freq)
            print("Encoding-based Pruning: ", encoding_pruning)
            print("Final Grammar Size ", len(best_candidates))
        
            #Save grammar and cost info
            best_cost_df.loc[:,"Construction"] = self.decode(best_cost_df.loc[:,"Chunk"].values)
            best_cost_df.to_csv(cost_file)
            best_chunk_df.to_csv(slot_cost_file)
        
        #Or load existing grammar
        else:
            best_cost_df = pd.read_csv(cost_file, index_col = 0)
            best_chunk_df = pd.read_csv(slot_cost_file, index_col = 0)
            
        print(best_cost_df)
        print(best_chunk_df)
        
        #Get examples if requested
        if get_examples == True:
            self.print_examples(grammar = best_cost_df.loc[:,"Chunk"].values, input_file = input_data, n = 25)
        
        return

    #------------------------------------------------------------------
    def decode(self, constructions):
    
        return_constructions = []
        
        #Iterate over items in grammar
        for construction in constructions:
        
            #Decode current construction
            construction = self.Load.decode_construction(construction)
            return_constructions.append(construction)
    
        return return_constructions
        
    #------------------------------------------------------------------
    def grid_search(self):
    
        print("Starting grid search for beam search parameters.")
        best_mdl = 999999999999 #High initial value to start search
        
        #Define frequency thresholds up to the minimum slot frequency
        freq_thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50]
        freq_thresholds = [self.Load.min_count - (self.Load.min_count * x) for x in freq_thresholds]
        freq_thresholds = list(set([int(x) for x in freq_thresholds]))
        
        #Initialize candidates module
        for delta_threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            
            print("Starting delta ", delta_threshold)
            #Initialize MDL
            mdl = Minimum_Description_Length(self.Load, self.Parse)

            #Get candidates without frequency threshold
            self.Candidates = Candidates(language = self.language, Load = self.Load, freq_threshold = self.Load.min_count, delta_threshold = delta_threshold, association_dict = self.Load.assoc_dict)

            #Get chunks
            chunks = self.Candidates.get_candidates(self.Load.data)

            first_chunks = True

            #Test frequency threshold within current delta p threshold
            for construction_freq in freq_thresholds:
                if construction_freq >= 0:
                
                    print("\n\t\t", self.nickname, "Delta: ", delta_threshold, "Freq: ", construction_freq)
                    
                    #For the first time, reduce to min frequency and save
                    if first_chunks == True:
                        #Reduce chunk set to those above frequency threshold
                        above_threshold = lambda x: x >= construction_freq
                        chunks = ct.valfilter(above_threshold, chunks)
                        grammar_fixed = list(chunks.keys()) #Fix order of keys
                        current_chunks = chunks
                    
                        #Get mask to support caching MDL parsing
                        chunk_mask = False
                        first_chunks = False
                    
                    #For after the first time, use mask to avoid re-parsing
                    else:
                        #Reduce chunk set to those above frequency threshold
                        above_threshold = lambda x: x >= construction_freq
                        current_chunks = ct.valfilter(above_threshold, chunks)
                    
                        #Get mask to support caching MDL parsing
                        chunk_mask = [1 if chunks[x] > construction_freq else 0 for x in chunks]
                        
                    #Recalculate constraint costs
                    mdl.get_constraint_cost()
                
                    #Cost of encoding the grammar
                    chunks_cost, chunk_df = mdl.get_grammar_cost(current_chunks)
                    
                    #Cost of encoding the data
                    total_mdl, l1_cost, l2_match_cost, l2_regret_cost = mdl.evaluate_grammar(current_chunks, grammar_fixed, chunks_cost, chunk_mask)
                    
                    #Check if this is the best version
                    if total_mdl < best_mdl:
                        print("\tNew best: Delta " + str(delta_threshold) + " and Freq: " + str(construction_freq))
                        best_delta = delta_threshold
                        best_freq = construction_freq
                        best_candidates = current_chunks
                        best_cost = chunks_cost
                        best_cost_df = chunk_df
                        best_mdl = total_mdl
                        best_grammar_fixed = grammar_fixed
                        best_chunks = chunks
                        best_slot_df = mdl.cost_df
                
        #Done with loop
        print("Best delta: " + str(best_delta) + " and best freq: " + str(best_freq))
        
        #Determine which constructions are worth encoding
        print("Checking encoding-cost pruning")
        best_cost_df.loc[:,"Cost"] = best_cost_df.loc[:,"Pointer"].mul(best_cost_df.loc[:,"Frequency"])
        test_cost_df = best_cost_df[best_cost_df.loc[:,"Cost"] > best_cost_df.loc[:,"Encoding"]]
        
        #Get subset of candidates that pass encoding test
        current_chunks = {}
        for row in test_cost_df.itertuples():
            chunk = row[1]
            freq = row[2]
            current_chunks[chunk] = freq
            
        test_grammar_fixed = list(current_chunks.keys()) #Fix order of keys
     
        #Initialize MDL
        mdl = Minimum_Description_Length(self.Load, self.Parse)
                
        #Recalculate constraint costs
        mdl.get_constraint_cost()
                
        #Cost of encoding the grammar
        chunks_cost, chunk_df = mdl.get_grammar_cost(current_chunks)
                    
        #Cost of encoding the data
        total_mdl, l1_cost, l2_match_cost, l2_regret_cost = mdl.evaluate_grammar(current_chunks, test_grammar_fixed, chunks_cost)
                    
        #Check if this is the best version
        if total_mdl < best_mdl:
            print("\tNew best: " + str(delta_threshold) + " with encoding-based pruning")
            best_delta = delta_threshold
            best_freq = construction_freq
            best_candidates = current_chunks
            best_cost = chunks_cost
            best_chunk_df = chunk_df
            best_cost_df = test_cost_df
            best_mdl = total_mdl
            encoding_pruning = "Yes"
            best_slot_df = mdl.cost_df
        else:
            print("\tEncoding-based pruning did not improve grammar.")
            encoding_pruning = "No"
        
        return best_delta, best_freq, best_candidates, best_cost, best_cost_df, best_slot_df, encoding_pruning
            
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
            construction = self.Load.decode_construction(x)

            print(i, construction)
            return_list.append(str(i) + ": " + str(construction))

        return return_list
    #-------------------------------------------------------------------------------
    def print_examples(self, grammar, input_file, n, output = None):

        #Default output
        if output == None:
            output = self.nickname + ".examples.txt"
            
        #Read and write in the default data directories
        output_file = os.path.join(self.out_dir, output)
        
        #Check if input is file or enriched data
        if isinstance(input_file, str):
            #Get text and enriched text
            lines_text = self.Load.read_file(input_file)
            lines_text = [self.Load.clean(x, encode = False) for x in lines_text]
            lines_enriched = [[self.Load.enrich(x) for x in y] for y in lines_text]

        #Open write file
        with codecs.open(output_file, "w", encoding = "utf-8") as fw:
            
            #Iterate over constructions
            for i in range(len(grammar)):
            
                x = grammar[i]
                printed_examples = []

                #Prune to actual constraints
                construction = self.Load.decode_construction(x)

                print(i, construction)
                fw.write(str(i) + "\t")
                fw.write(construction)
                fw.write("\n")
                
                #Input may be a string rather than tuple
                if isinstance(x, str):
                    x = eval(x)
                #Determine how long sequence should be 
                length = len(x)
                #Track how many examples have been found
                counter = 0

                #Iterate over lines
                for j in range(len(lines_text)):
                    if counter < n:

                        line = lines_text[j]
                        encoding = lines_enriched[j]
                        
                        #Parse examples
                        construction_thing, indexes, matches = self.Parse.parse_examples(x, encoding)

                        if matches > 0:
                            for index in indexes:
                                
                                text = line[index:index+length]
                                
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
            discount_dict = self.Association.find_discounts(self.Load.data)
            self.Load.save_file(discount_dict, self.nickname+".discounts.p")
            print(discount_dict)
            print("Discounts ", self.nickname)

        else:
            discount_dict = False

        ngrams = self.Association.find_ngrams(self.Load.data, lex_only = False, n_gram_threshold = 1)
        association_dict = self.Association.calculate_association(ngrams = ngrams, normalization = self.normalization, discount_dict = discount_dict)
        
        #Reduce to bigrams
        keepable = lambda x: len(x) > 1
        all_ngrams = ct.keyfilter(keepable, association_dict)

        #Convert to readable CSV
        pairs = []
        for pair in all_ngrams:
            
            #Decode to readable annotations
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
    def clean_label(self, label):
    
        start = label.find(":")
        label = label[start+1:]

        end = label.find("_")
        label = label[:end]
        
        return int(label)
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
            if "sem:" in word1:
                word1 = (3, self.clean_label(word1))
            elif "syn:" in word1:
                word1 = (2, self.clean_label(word1))
            else:
                word1 = (1, self.Load.lex_encode[word1])
            
            #Get categories instead of names, 2
            if "sem:" in word2:
                word2 = (3, self.clean_label(word2))
            elif "syn:" in word2:
                word2 = (2, self.clean_label(word2))
            else:
                word2 = (1, self.Load.lex_encode[word2])
            
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