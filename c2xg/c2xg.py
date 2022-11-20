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
        self.Candidates = Candidates(language = self.language, Load = self.Load)
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
        lexicon, phrases, unique_words = self.Load.get_lexicon(input_data, npmi_threshold, min_count)

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
        print("Enriching input using syntactic and semantic categories")
        self.data = self.Load.load(input_data)  #Save the enriched data once gotten
            
        #Get pairwise association with Delta P
        association_file = os.path.join(self.out_dir, self.nickname + ".association.gz")
        if not os.path.exists(association_file):
            association_df = self.get_association(freq_threshold = min_count, normalization = self.normalization, lex_only = False)
            association_df.to_csv(association_file, compression = "gzip")
        else:
            association_df = pd.read_csv(association_file)
            
        #Now print association data
        print(association_df)
        
        
        return

    #------------------------------------------------------------------

    def eval_mdl(files, workers, candidates, Load, Encode, Parse, freq_threshold = -1, report = False):
    
        print("Now initiating MDL evaluation: " + str(files))
        
        #Check if one file
        if isinstance(files, str):
            files = [files]
        
        for file in files:
            print("\tStarting " + str(file))        
            MDL = MDL_Learner(Load, Encode, Parse, freq_threshold = freq_threshold, vectors = {"na": 0}, candidates = candidates)
            MDL.get_mdl_data([file], workers = workers, learn_flag = False)
            total_mdl, l1_cost, l2_match_cost, l2_regret_cost, baseline_mdl = MDL.evaluate_subset(subset = False, return_detail = True)
                
        if report == True:
            return total_mdl, l1_cost, l2_match_cost, l2_regret_cost, baseline_mdl
    #------------------------------------------------------------        

    def delta_grid_search(candidate_file, test_file, workers, mdl_workers, association_dict, freq_threshold, language, in_dir, out_dir, max_words, nickname = "current"):
    
        print("\nStarting grid search for beam search settings.")
        result_dict = {}
            
        delta_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        
        if len(delta_thresholds) < workers:
            parse_workers = len(delta_thresholds)
        else:
            parse_workers = workers
            
        #Multi-process#
        pool_instance = mp.Pool(processes = parse_workers, maxtasksperchild = 1)
        distribute_list = [(x, candidate_file) for x in delta_thresholds]

        pool_instance.map(partial(process_candidates, 
                                    association_dict = association_dict.copy(),
                                    language = language,
                                    freq_threshold = freq_threshold,
                                    in_dir = in_dir,
                                    out_dir = out_dir,
                                    max_words = max_words,
                                    nickname = nickname
                                    ), distribute_list, chunksize = 1)
        pool_instance.close()
        pool_instance.join()
                    
        #Now MDL
        if language == "zho":
            zho_split = True
        else:
            zho_split = False
            
        Load = Loader(in_dir, out_dir, language, max_words = max_words)
        Encode = Encoder(Loader = Load, zho_split = zho_split)
        Parse = Parser(Load, Encode)
        
        for threshold in delta_thresholds:
            print("\tStarting MDL search for " + str(threshold))
            filename = str(candidate_file) + "." + nickname + ".delta." + str(threshold) + ".p"
            candidates = Load.load_file(filename)
            
            if len(candidates) < 5:
                print("\tNot enough candidates!")
            
            else:

                mdl_score = eval_mdl(files = test_file, 
                                        candidates = candidates, 
                                        workers = mdl_workers, 
                                        Load = Load, 
                                        Encode = Encode, 
                                        Parse = Parse, 
                                        freq_threshold = freq_threshold, 
                                        report = True
                                        )
                
                result_dict[threshold] = mdl_score
                print("\tThreshold: " + str(threshold) + " and MDL: " + str(mdl_score))
            
        #Get threshold with best score
        print(result_dict)
        best = min(result_dict.items(), key=operator.itemgetter(1))[0]

        #Get best candidates
        filename = str(candidate_file) + "." + nickname + ".delta." + str(best) + ".p"
        best_candidates = Load.load_file(filename)
            
        return best, best_candidates

    #------------------------------------------------------------

    def process_candidates(input_tuple, association_dict, language, in_dir, out_dir, freq_threshold = 1, mode = "", max_words = False, nickname = "current"):

        threshold =  input_tuple[0]
        candidate_file = input_tuple[1]
        
        print("\tStarting " + str(threshold) + " with freq threshold: " + str(freq_threshold))
        Load = Loader(in_dir, out_dir, language, max_words)
        C = Candidates(language = language, Loader = Load, association_dict = association_dict)
        
        if mode == "candidates":
            filename = str(candidate_file + ".candidates.p")
            
        else:
            filename = str(candidate_file) + "." + nickname + ".delta." + str(threshold) + ".p"
        
        if filename not in Load.list_output():
        
            candidates = C.process_file(candidate_file, threshold, freq_threshold, save = False)
            Load.save_file(candidates, filename)
        
        #Clean
        del association_dict
        del C
    
        return

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

        #Compatbility with idNet
        if mode == "idNet":
            mode = "lines"
            
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
    
    def learn_old(self, 
                nickname, 
                cycles = 1, 
                cycle_size = (1, 5, 20), 
                freq_threshold = 10, 
                beam_freq_threshold = 10,
                turn_limit = 10, 
                workers = 1,
                mdl_workers = 1,
                states = None,
                fixed_set = [],
                beam_threshold = None,
                no_mdl = False,
                ):
    
        self.nickname = nickname

        if self.data_dir == None:
            print("Error: Cannot train grammars without specified data directory.")
            sys.kill()

        #Check learning state and resume
        self.model_state_file = self.language + "." + self.nickname + ".State.p"
        
        try:
            loader_files = self.Load.list_output()
        except:
            loader_files = []

        if self.model_state_file in loader_files:
            print("Resuming learning state.")
            self.progress_dict, self.data_dict = self.Load.load_file(self.model_state_file)
            
            if states != None:
                print("Manual state change!")
                for state in states:
                    self.progress_dict[state[0]][state[1]] = state[2]
            
        else:
            print("Initializing learning state.")
            self.data_dict = self.divide_data(cycles, cycle_size, fixed_set)
            self.progress_dict = self.set_progress()
            self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
        
        #Check beam setting
        if beam_threshold != None:
            self.progress_dict["BeamSearch"] = beam_threshold

        #Learn each cycle
        for cycle in self.progress_dict.keys():
            if isinstance(cycle, int):
            
                if self.progress_dict[cycle]["State"] == "Complete":
                    print("\t Cycle " + str(cycle) + " already complete.")
                    
                #This cycle is not yet finished
                else:

                    #-----------------#
                    #BACKGROUND STAGE
                    #-----------------#
                    if self.progress_dict[cycle]["Background_State"] != "Complete":
                        
                        #Check if ngram extraction is finished
                        if self.progress_dict[cycle]["Background_State"] == "None":
                            check_files = self.Load.list_output(type = "ngrams")
                            pop_list = []
                            for i in range(len(self.progress_dict[cycle]["Background"])):
                                if self.progress_dict[cycle]["Background"][i] + "." + self.nickname + ".ngrams.p" in check_files:
                                    pop_list.append(i)

                            #Pop items separately in reverse order
                            if len(pop_list) > 0:
                                for i in sorted(pop_list, reverse = True):
                                    self.progress_dict[cycle]["Background"].pop(i)
                            
                            #If remaining background files, process them
                            if len(self.progress_dict[cycle]["Background"]) > 0:
                                print("\tNow processing remaining files: " + str(len(self.progress_dict[cycle]["Background"])))
                                self.Association.find_ngrams(self.progress_dict[cycle]["Background"], workers)
                                
                            #Change state
                            self.progress_dict[cycle]["Background_State"] = "Ngrams"
                            self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
                        
                        #Check if ngram merging is finished
                        if self.progress_dict[cycle]["Background_State"] == "Ngrams":
                            files = [filename + "." + self.nickname + ".ngrams.p" for filename in self.data_dict[cycle]["Background"]]
                        
                            print("\tNow merging ngrams for files: " + str(len(files)))
                            ngrams = self.Association.merge_ngrams(files = files, n_gram_threshold = freq_threshold)
                            
                            #Save data and state
                            self.Load.save_file(ngrams, nickname + ".Cycle-" + str(cycle) + ".Merged-Grams.p")
                            self.progress_dict[cycle]["Background_State"] = "Merged"
                            self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
                        
                        #Check if association_dict has been made
                        if self.progress_dict[cycle]["Background_State"] == "Merged":
                            ngrams = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".Merged-Grams.p")
                            association_dict = self.Association.calculate_association(ngrams = ngrams, smoothing = self.smoothing, save = False)
                            del ngrams
                            self.Load.save_file(association_dict, nickname + ".Cycle-" + str(cycle) + ".Association_Dict.p")
                            self.progress_dict[cycle]["Background_State"] = "Complete"
                            self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
                            self.association_dict = association_dict
                            
                    else:
                        print("\tLoading association_dict.")
                        self.association_dict = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".Association_Dict.p")
                        
                    #-----------------#
                    #CANDIDATE STAGE
                    #-----------------#    
                    
                    if self.progress_dict[cycle]["Candidate_State"] != "Complete":

                        print("Initializing Candidates module")
                        C = Candidates(self.language, self.Load, workers, self.association_dict)
                        
                        #Find beam search threshold
                        if self.progress_dict["BeamSearch"] == "None" or self.progress_dict["BeamSearch"] == {}:
                            print("Finding Beam Search settings.")

                            delta_threshold, best_candidates = delta_grid_search(candidate_file = self.data_dict["BeamCandidates"], 
                                                                    test_file = self.data_dict["BeamTest"], 
                                                                    workers = workers, 
                                                                    mdl_workers = mdl_workers,
                                                                    association_dict = self.association_dict, 
                                                                    freq_threshold = beam_freq_threshold,
                                                                    language = self.language, 
                                                                    in_dir = self.in_dir, 
                                                                    out_dir = self.out_dir,
                                                                    nickname = self.nickname,
                                                                    max_words = self.max_words,
                                                                    )
                            self.progress_dict["BeamSearch"] = delta_threshold
                            
                            self.progress_dict[cycle]["Candidate_State"] = "Threshold"
                            self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)

                        #If saved, load beam search threshold
                        else:
                            print("Loading Beam Search settings.")
                            delta_threshold = self.progress_dict["BeamSearch"]
                            self.progress_dict[cycle]["Candidate_State"] = "Threshold"

                        #For a fixed set experiment, we use the same data so we keep the best candidates
                        if fixed_set == []:
                        
                            #Check which files have been completed
                            if self.progress_dict[cycle]["Candidate_State"] == "Threshold":
                                check_files = self.Load.list_output(type = "candidates")
                                pop_list = []
                                for i in range(len(self.progress_dict[cycle]["Candidate"])):
                                    if self.progress_dict[cycle]["Candidate"][i] + ".candidates.p" in check_files:
                                        pop_list.append(i)                            
                                        
                                #Pop items separately in reverse order
                                if len(pop_list) > 0:
                                    for i in sorted(pop_list, reverse = True):
                                        self.progress_dict[cycle]["Candidate"].pop(i)
                                    
                                #If remaining candidate files, process them
                                if len(self.progress_dict[cycle]["Candidate"]) > 0:
                                    print("\n\tNow processing remaining files: " + str(len(self.progress_dict[cycle]["Candidate"])))
                                    
                                    #Multi-process#
                                    if workers > len(self.progress_dict[cycle]["Candidate"]):
                                        candidate_workers = len(self.progress_dict[cycle]["Candidate"])
                                    else:
                                        candidate_workers = workers
                                        
                                    pool_instance = mp.Pool(processes = candidate_workers, maxtasksperchild = 1)
                                    distribute_list = [(delta_threshold, x) for x in self.progress_dict[cycle]["Candidate"]]
                                    pool_instance.map(partial(process_candidates, 
                                                                            association_dict = self.association_dict.copy(),
                                                                            language = self.language,
                                                                            in_dir = self.in_dir,
                                                                            out_dir = self.out_dir,
                                                                            mode = "candidates",
                                                                            max_words = self.max_words,
                                                                            ), distribute_list, chunksize = 1)
                                    pool_instance.close()
                                    pool_instance.join()
                                    
                                self.progress_dict[cycle]["Candidate_State"] = "Merge"
                                self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)

                            #Merge and Save candidates
                            if self.progress_dict[cycle]["Candidate_State"] == "Merge":
                                output_files = [filename + ".candidates.p" for filename in self.data_dict[cycle]["Candidate"]]
                                candidates = self.Candidates.merge_candidates(output_files, freq_threshold)
                            
                                self.Load.save_file(candidates, nickname + ".Cycle-" + str(cycle) + ".Candidates.p")
                                self.progress_dict[cycle]["Candidate_State"] = "Dict"
                                self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
                                
                            #Make association vectors
                            if self.progress_dict[cycle]["Candidate_State"] == "Dict":
                                
                                candidates = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".Candidates.p")
                                candidate_dict = self.Candidates.get_association(candidates, self.association_dict)
                                self.Load.save_file(candidate_dict, nickname + ".Cycle-" + str(cycle) + ".Candidate_Dict.p")
                                
                                self.progress_dict[cycle]["Candidate_State"] == "Complete"
                                self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
                                
                            
                            else:
                                print("\tLoading candidate_dict.")
                                candidate_dict = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".Candidate_Dict.p")
                                candidates = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".Candidates.p")
                            
                            del self.association_dict
                    
                        #If there was a fixed set of training/testing files
                        elif fixed_set != []:

                            candidates = best_candidates
                            candidate_dict = self.Candidates.get_association(candidates, self.association_dict)
                            del self.association_dict
                            self.progress_dict[cycle]["Candidate_State"] == "Complete"

                    #-----------------#
                    #MDL STAGE
                    #-----------------#
                    if no_mdl == False:
                        if self.progress_dict[cycle]["MDL_State"] != "Complete":
                        
                            #Prep test data for MDL
                            if self.progress_dict[cycle]["MDL_State"] == "None":
                                MDL = MDL_Learner(self.Load, self.Encode, self.Parse, freq_threshold = 1, vectors = candidate_dict, candidates = candidates)
                                MDL.get_mdl_data(self.progress_dict[cycle]["Test"], workers = mdl_workers)
                                self.Load.save_file(MDL, nickname + ".Cycle-" + str(cycle) + ".MDL.p")
                                
                                self.progress_dict[cycle]["MDL_State"] = "EM"
                                self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
                            
                            #Run EM-based Tabu Search
                            if self.progress_dict[cycle]["MDL_State"] == "EM":
                                
                                try:
                                    MDL.search_em(turn_limit, mdl_workers)
                                except:
                                    MDL = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".MDL.p")
                                    MDL.search_em(turn_limit, mdl_workers)
                                    
                                self.Load.save_file(MDL, nickname + ".Cycle-" + str(cycle) + ".MDL.p")
                                self.progress_dict[cycle]["MDL_State"] = "Direct"
                                self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
                                
                            #Run direct Tabu Search
                            if self.progress_dict[cycle]["MDL_State"] == "Direct":
                                
                                try:
                                    MDL.search_direct(turn_limit*3, mdl_workers)
                                except:
                                    MDL = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".MDL.p")
                                    MDL.search_direct(turn_limit*3, mdl_workers)
                                
                                #Get grammar to save
                                grammar_dict = defaultdict(dict)
                                for i in range(len(MDL.candidates)):
                                    grammar_dict[i]["Constructions"] = MDL.candidates[i]
                                    grammar_dict[i]["Matches"] = MDL.matches[i]
                                        
                                #Save grammar
                                self.Load.save_file(grammar_dict, nickname + ".Cycle-" + str(cycle) + ".Final_Grammar.p")
                                
                                self.progress_dict[cycle]["MDL_State"] = "Complete"
                                self.progress_dict[cycle]["State"] = "Complete"
                                self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)    
                                
                                del MDL

                    elif no_mdl == True:
                        print("Calculating MDL")

                        self.progress_dict[cycle]["MDL_State"] = "Complete"
                        self.progress_dict[cycle]["State"] = "Complete"
                
        #-----------------#
        #MERGING STAGE
        #-----------------#
        if self.progress_dict[cycle]["State"] == "Complete":
            
            if no_mdl == False:
                print("Starting to merge fold grammars.")
                grammar_files = [nickname + ".Cycle-" + str(i) + ".Final_Grammar.p" for i in range(cycles)]
                final_grammar = self.merge_grammars(grammar_files)
                self.Load.save_file(final_grammar, self.language + ".Grammar.p")

            else:
                final_grammar = list(candidates.keys())
                self.Load.save_file(final_grammar, self.nickname + ".Grammar_BeamOnly.p")
                
    #-------------------------------------------------------------------------------
    
    def merge_grammars(self, grammar_files, no_mdl = False):
    
        all_grammars = {}
        
        if no_mdl == False:
            #Load all grammar files
            for file in grammar_files:
            
                current_dict = self.Load.load_file(file)
                
                #Iterate over constructions in current fold grammar
                for key in current_dict.keys():
                    current_construction = current_dict[key]["Constructions"]
                    current_construction = current_construction.tolist()
                    current_matches = current_dict[key]["Matches"]
                    
                    #Reformat
                    new_construction = []
                    for unit in current_construction:
                        new_type = unit[0]
                        new_index = unit[1]
                            
                        if new_type != 0:
                            new_construction.append(tuple((new_type, new_index)))
                    
                    #Make hashable
                    new_construction = tuple(new_construction)
                    
                    #Add to dictionary
                    if new_construction not in all_grammars:
                        all_grammars[new_construction] = {}
                        all_grammars[new_construction]["Matches"] = current_matches
                        all_grammars[new_construction]["Selected"] = 1
                    
                    else:
                        all_grammars[new_construction]["Matches"] += current_matches
                        all_grammars[new_construction]["Selected"] += 1
                        
            #Done loading grammars
            print("Final grammar for " + self.language + " contains "  + str(len(list(all_grammars.keys()))))
            final_grammar = list(all_grammars.keys())
            final_grammar = self.Parse.format_grammar(final_grammar)

        else:
            final_grammar = []
            for file in grammar_files:
                current_dict = self.Load.load_file(file)
                for key in current_dict:
                    if key not in final_grammar:
                        final_grammar.append(key)
        
        return final_grammar                
            
    #-------------------------------------------------------------------------------
    
    
   
    #-----------------------------------------------
    
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
    def step_data(self, data, step):

        return_data = []
        extra_data = []
        
        counter = 0
        
        for line in data:
            if len(line) > 5:
        
                if counter < step:
                    return_data.append(line)
                    counter += len(line.split())
                    
                else:
                    extra_data.append(line)
                
        return return_data, extra_data
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