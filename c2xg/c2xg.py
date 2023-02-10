import os
import time
import numpy as np
import pandas as pd
import pickle
import codecs
import difflib
from Levenshtein import distance
import multiprocessing as mp
import cytoolz as ct
from functools import partial
from pathlib import Path
from gensim.models.fasttext import load_facebook_model
from gensim.models.phrases import Phrases

from .Loader import Loader
from .Parser import Parser
from .Parser import parse_fast
from .Parser import detail_model
from .Association import Association
from .Candidates import Candidates
from .MDL import Minimum_Description_Length
from .Word_Classes import Word_Classes

#----------------------------------------------------------
    
def token_similarity(input_tuple, examples_dict):
    
    #Process input tuple
    construction1 = input_tuple[0]
    construction2 = input_tuple[1]
    
    #Initialize matrix
    overlaps = []
    
    #Get examples
    examples1 = examples_dict[construction1]
    examples2 = examples_dict[construction2]
    
    #No need to check the same examples against one another
    if examples1 == examples2:
        return 0.0
        
    elif len(examples1) < 1 or len(examples2) < 1:
        return 1.0
    
    elif None in examples1 or None in examples2:
        return 1.0
            
    else:
        #Look for overlaps between all pairs of examples and take the lowest one (low = similar)
        for string1 in examples1:
            for string2 in examples2:
                    
                #Sequence matching algorithm (by words)
                s = difflib.SequenceMatcher(None, string1, string2)
                length = max(len(string1), len(string2))
                overlap = sum([x[2] for x in s.get_matching_blocks()]) / float(length)
        
                #Only count as a match if sufficient overlap
                if overlap > 0.75:
                
                    #Return with the first match
                    return 0.0
          
        #Only process the entire cycle if no close matches are found
        return 1.0
             
#------------------------------------------------------------------
    
def process_clipping(input_tuple, grammar_list):
    
    #For multi-processing
    i = input_tuple[0]
    j = input_tuple[1]

    construction1 = grammar_list[i]
    construction2 = grammar_list[j]
    new_construction = False
                            
    #Check for 1-2 overlap
    if construction1[-1] == construction2[0]:
        new_construction = construction1 + construction2[1:]
        clip_index = len(construction1)
                            
    #Check for 2-1 overlap
    elif construction2[-1] == construction1[0]:
        new_construction = construction2 + construction1[1:]
        clip_index = len(construction2)
                                
    #Evaluate potential clipping
    if new_construction != False:
                            
        #Length check
        if len(new_construction) < 10:
            if new_construction not in grammar_list:
                return (new_construction, clip_index)
 
#-------------------------------------------------------------------------------

def process_clipping_parsing(input_tuple, data, min_count):

    #Initialize parser
    Parse = Parser()
    
    #Unpack input
    candidate_pool = input_tuple[0]
    clip_pool = input_tuple[1]
    
    #Parse
    frequencies = Parse.parse_enriched(data, grammar = candidate_pool)
    frequencies = np.sum(frequencies, axis=0)
    frequencies = frequencies.tolist()[0]
    
    clips = {}
    new_constructions = {}

    #Check candidates
    for i in range(len(candidate_pool)):
        new_construction = candidate_pool[i]
        freq = frequencies[i]
                
        #Freq check
        if freq > min_count:
            clip_index = clip_pool[i]
            new_constructions[new_construction] = freq
            clips[new_construction] = clip_index
            
    return new_constructions, clips

#-------------------------------------------------------------------------------

class C2xG(object):
    
    def __init__(self, data_dir = None, language = "eng", nickname = "cxg", model = None, 
                    normalization = True, max_words = False, starting_index = 0, cbow_file = "", sg_file = ""):
    
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
        self.Load = Loader(in_dir, out_dir, language = self.language, max_words = max_words, nickname = self.nickname, sg_model = self.sg_model, cbow_model = self.cbow_model)
        self.Load.starting_index = starting_index
        self.Association = Association(Load = self.Load, nickname = self.nickname)
        self.Parse = Parser(self.Load)
        self.Word_Classes = Word_Classes(self.Load)

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

    def learn(self, input_data, npmi_threshold = 0.75, min_count = None, cbow_range = False, sg_range = False, get_examples = True, increments = 50000, learning_rounds = 20, forgetting_rounds = 40, cluster_only = False):

        #Adjust min_count to be 1 parts per million using max_words parameter
        if min_count == None:
            if self.max_words == None:
                min_count = 5
            elif self.max_words <= 1000000:
                min_count = int(1000000/self.max_words * 1)
            elif self.max_words > 1000000:
                min_count = int(self.max_words/1000000 * 1)
            print("Setting min_count to 1 parts per million (min_count = " + str(min_count) + ") (max_words = " + str(self.max_words) + ")")
 
        #Filenames for lexicon and phrases
        lex_file = self.nickname + ".lexicon.p"
        phrase_file = self.nickname + ".phrases.p"
        unique_file = os.path.join(self.out_dir, self.nickname + ".unique_words.csv")
        self.Load.min_count = min_count
        
        #If lexicon and phrases don't exist
        if not os.path.exists(os.path.join(self.out_dir, lex_file)):
            print("Starting to learn: lexicon")
            lexicon, phrases, unique_words, self.Load.full_lexicon = self.Load.get_lexicon(input_data, npmi_threshold, self.Load.min_count)

            n_phrases = len([x for x in lexicon.keys() if " " in x])
            print("Finished with " + str(len(lexicon)-n_phrases) + " words and " + str(n_phrases) + " phrases")

            #Save phrases and lexicon
            self.Load.phrases = phrases
            self.Load.lexicon = lexicon
            
            #Store lexicon and phrases
            self.Load.save_file(lexicon, lex_file)
            self.Load.save_file(phrases.phrasegrams, phrase_file)
            unique_words.to_csv(unique_file)
            
        #Load existing lexicon and phrases, reconstitute phrases
        else:
            print("Loading lexicon and phrases")
            self.Load.full_lexicon = pd.read_csv(os.path.join(self.out_dir, self.nickname+".full_lexicon.csv"))
            self.Load.lexicon = self.Load.load_file(lex_file)
            unique_words = pd.read_csv(unique_file, index_col = 0)
            temp_phrases = self.Load.load_file(phrase_file)
            self.Load.phrases = Phrases(["holder"], min_count = min_count, threshold = npmi_threshold, scoring = "npmi", delimiter = " ")
            self.Load.phrases = self.Load.phrases.freeze()
            self.Load.phrases.phrasegrams = temp_phrases
            del temp_phrases
        
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
        self.Load.add_categories(self.Load.cbow_df, self.Load.sg_df, self.Load.lexicon, self.Load.phrases.phrasegrams, self.Load.full_lexicon, unique_words)
        
        #Stop here if necessary
        if cluster_only == True:
            return
        
        #STARTING ON-GOING LEARNING AFTER THIS
        base_nickname = self.nickname
        for learning_round in range(learning_rounds):
        
            #Update nickname
            self.nickname = base_nickname + "_round" + str(learning_round)
            self.Load.nickname = self.nickname
            print("Starting new learning round: " + self.nickname)  

            #First round doesn't merge or update lexicon
            if learning_round == 0:
            
                grammar_df_lex, grammar_df_syn, grammar_df_full, clips_lex, clips_syn, clips_full = self.learn_streaming(input_data, get_examples, forgetting_rounds, increments)
                
            #Otherwise merge with existing
            else:
            
                #Increment starting index, including for the data used for forgetting
                self.Load.starting_index += self.max_words + (forgetting_rounds * increments)
            
                #Load new data
                lexicon, phrases, unique_words, full_lexicon = self.Load.get_lexicon(input_data, npmi_threshold, min_count)
                
                #Update phrases
                current_phrases = self.Load.phrases.phrasegrams
                self.Load.phrases.phrasegrams = ct.merge(current_phrases, phrases.phrasegrams)
                print("\t Expanding lexical phrases from " + str(len(current_phrases)) + " to " + str(len(self.Load.phrases.phrasegrams)))
                
                #Update the lexicon
                start_size = len(self.Load.lexicon)
                max_index = max(list(self.Load.lex_decode.keys()))

                for word in lexicon:
                    #Update frequencies for existing words
                    if word in self.Load.lexicon:
                        self.Load.lexicon[word] += lexicon[word]
                    #Add new words
                    else:
                        max_index += 1
                        self.Load.lexicon[word] = lexicon[word]
                        #Update encoding
                        self.Load.lex_decode[max_index] = word
                        self.Load.lex_encode[word] = max_index

                print("\t Expanding lexicon from " + str(start_size) + " to " + str(len(self.Load.lexicon)))
                self.Load.add_categories(self.Load.cbow_df, self.Load.sg_df, self.Load.lexicon, self.Load.phrases.phrasegrams, full_lexicon, unique_words, update = True)
                
                #Get new and merged grammars
                grammar_df_lex, grammar_df_syn, grammar_df_full, clips_lex, clips_syn, clips_full = self.learn_streaming(input_data, get_examples, forgetting_rounds, increments, 
                                                                                                                            grammar_df_lex, grammar_df_syn, grammar_df_full, 
                                                                                                                            clips_lex, clips_syn, clips_full)
                
        #Finished with all learning rounds, now the final pruning and clustering
        #Update nickname
        self.nickname = base_nickname + "_final_round"
        self.Load.nickname = self.nickname
        print("Starting final forgetting round with clustering: " + self.nickname)  
        
        #Increment starting index, including for the data used for forgetting
        self.Load.starting_index += self.max_words + (forgetting_rounds * increments)
        
        grammar_df_lex, grammar_df_syn, grammar_df_full, clips_lex, clips_syn, clips_full = self.learn_final_steps(input_data, forgetting_rounds, increments,
                                                                                                                            grammar_df_lex, grammar_df_syn, grammar_df_full, 
                                                                                                                            clips_lex, clips_syn, clips_full)
        print(grammar_df_lex)
        print(grammar_df_syn)
        print(grammar_df_full)
        print("Finished!")
        
        return grammar_df_lex, grammar_df_syn, grammar_df_full
    #------------------------------------------------------------------------------------------------
        
    def learn_streaming(self, input_data, forgetting_rounds, increments, get_examples = False,
                            temp_grammar_df_lex = [], temp_grammar_df_syn = [], temp_grammar_df_full = [], 
                            temp_clips_lex = None, temp_clips_syn = None, temp_clips_full = None):
        
        #Now that we have clusters, enrich input data and save
        if not os.path.exists(os.path.join(self.out_dir, self.nickname+".input_enriched.p")):
            print("Enriching input using syntactic and semantic categories")
            self.Load.actual_data = self.Load.load(input_data)  #Save the enriched data once gotten
            self.Load.save_file(self.Load.actual_data, self.nickname+".input_enriched.p")
        else:
            print("Loading enriched input")
            self.Load.actual_data = self.Load.load_file(self.nickname+".input_enriched.p")
            
        #Get lexical only constructions
        if not os.path.exists(os.path.join(self.out_dir, self.nickname+".lex.grammar_clipping.csv")):
            print("Starting lexical only constructions.")
            grammar_df_lex, clips_lex = self.process_grammar(input_data, grammar_type = "lex", get_examples = get_examples)
        else:
            grammar_df_lex = pd.read_csv(os.path.join(self.out_dir, self.nickname+".lex.grammar_clipping.csv"), index_col = 0)
            clips_lex = self.Load.load_file(self.nickname+".lex.grammar_clipping_indexes.p")
        
        #Get syntactic only constructions
        if not os.path.exists(os.path.join(self.out_dir, self.nickname+".syn.grammar_clipping.csv")):
            print("Starting syntactic only constructions.")
            grammar_df_syn, clips_syn = self.process_grammar(input_data, grammar_type = "syn", get_examples = get_examples)
        else:
            grammar_df_syn = pd.read_csv(os.path.join(self.out_dir, self.nickname+".syn.grammar_clipping.csv"), index_col = 0)
            clips_syn = self.Load.load_file(self.nickname+".syn.grammar_clipping_indexes.p")
        
        #Get full constructions
        if not os.path.exists(os.path.join(self.out_dir, self.nickname+".full.grammar_clipping.csv")):
            print("Starting full constructions.")
            grammar_df_full, clips_full = self.process_grammar(input_data, grammar_type = "full", get_examples = get_examples)
        else:
            grammar_df_full = pd.read_csv(os.path.join(self.out_dir, self.nickname+".full.grammar_clipping.csv"), index_col = 0)
            clips_full = self.Load.load_file(self.nickname+".full.grammar_clipping_indexes.p")
        
        #If not the first round, now merge existing grammars before forgetting
        if temp_clips_lex != None:
            
            print("\tMerging lexical grammar: " + str(len(grammar_df_lex)) + " with " + str(len(temp_grammar_df_lex)))
            grammar_df_lex = pd.concat([grammar_df_lex, temp_grammar_df_lex], axis = 0, ignore_index = True)
            grammar_df_lex = grammar_df_lex.drop_duplicates(subset = "Chunk", keep = "first")
            grammar_df_lex.loc[:,"Type"] = "Lexical-Only"
            print(grammar_df_lex)
            clips_lex = ct.merge(clips_lex, temp_clips_lex)
                
            print("\tMerging syntactic grammar: " + str(len(grammar_df_syn)) + " with " + str(len(temp_grammar_df_syn)))
            grammar_df_syn = pd.concat([grammar_df_syn, temp_grammar_df_syn], axis = 0, ignore_index = True)
            grammar_df_syn = grammar_df_syn.drop_duplicates(subset = "Chunk", keep = "first")
            grammar_df_syn.loc[:,"Type"] = "Syntactic-Only"
            print(grammar_df_syn)
            clips_syn = ct.merge(clips_syn, temp_clips_syn)
                
            print("\tMerging full grammar: " + str(len(grammar_df_full)) + " with " + str(len(temp_grammar_df_full)))
            grammar_df_full = pd.concat([grammar_df_full, temp_grammar_df_full], axis = 0, ignore_index = True)
            grammar_df_full = grammar_df_full.drop_duplicates(subset = "Chunk", keep = "first")
            grammar_df_full.loc[:,"Type"] = "Full Grammar"
            print(grammar_df_full)
            clips_full = ct.merge(clips_full, temp_clips_full)
        
        #Check if forgetting is desired
        if increments != False:
        
            #Forgetting for lexical grammar
            forget_name_lex = os.path.join(self.out_dir, self.nickname + ".lex.grammar_forgetting.csv")
            if not os.path.exists(forget_name_lex):
                grammar_df_lex, clips_lex = self.forget_constructions(grammar_df_lex.loc[:,"Chunk"], input_data, threshold = 1, adjustment = 0.20, 
                                                                rounds = forgetting_rounds, increment_size = increments, name = "lex", clips = clips_lex, temp_grammar = temp_grammar_df_lex)
                grammar_df_lex.to_csv(forget_name_lex)
                self.Load.save_file(clips_lex, self.nickname+".clips_lex_forgetting.p")
            else:
                grammar_df_lex = pd.read_csv(forget_name_lex, index_col = 0)
                clips_lex = self.Load.load_file(self.nickname+".clips_lex_forgetting.p")
                
            print(grammar_df_lex)
            
            #Forgetting for syntactic grammar
            forget_name_syn = os.path.join(self.out_dir, self.nickname + ".syn.grammar_forgetting.csv")
            if not os.path.exists(forget_name_syn):
                grammar_df_syn, clips_syn = self.forget_constructions(grammar_df_syn.loc[:,"Chunk"], input_data, threshold = 1, adjustment = 0.20, 
                                                            rounds = forgetting_rounds, increment_size = increments, name = "syn", clips = clips_syn, temp_grammar = temp_grammar_df_syn)
                grammar_df_syn.to_csv(forget_name_syn)
                self.Load.save_file(clips_syn, self.nickname+".clips_syn_forgetting.p")
            else:
                grammar_df_syn = pd.read_csv(forget_name_syn, index_col = 0)
                clips_syn = self.Load.load_file(self.nickname+".clips_syn_forgetting.p")
                
            print(grammar_df_syn)
            
            #Forgetting for full grammar
            forget_name_full = os.path.join(self.out_dir, self.nickname + ".full.grammar_forgetting.csv")
            if not os.path.exists(forget_name_full):
                grammar_df_full, clips_full = self.forget_constructions(grammar_df_full.loc[:,"Chunk"], input_data, threshold = 1, adjustment = 0.20, 
                                                            rounds = forgetting_rounds, increment_size = increments, name = "full", clips = clips_full, temp_grammar = temp_grammar_df_full)
                grammar_df_full.to_csv(forget_name_full)
                self.Load.save_file(clips_full, self.nickname+".clips_full_forgetting.p")
            else:
                grammar_df_full = pd.read_csv(forget_name_full, index_col = 0)
                clips_full = self.Load.load_file(self.nickname+".clips_full_forgetting.p")
                
            print(grammar_df_full)
        
        #Combine grammars
        print("Merging scaffolded grammars")
        grammar_df_lex.loc[:,"Type"] = "Lexical-Only"
        grammar_df_syn.loc[:,"Type"] = "Syntactic-Only"
        grammar_df_full.loc[:,"Type"] = "Full Grammar"
        grammar_df = pd.concat([grammar_df_lex, grammar_df_syn, grammar_df_full], axis = 0, ignore_index = True)
        grammar_df = grammar_df.drop_duplicates(subset = "Chunk", keep = "first")
        self.Load.clips = ct.merge(clips_lex, clips_syn, clips_full)
        
        #Save grammars
        grammar_df.to_csv(os.path.join(self.out_dir, self.nickname + ".forgetting_merged_grammar.csv"))
        self.Load.save_file(self.Load.clips, self.nickname + ".forgetting_merged_grammar_clips.p")
        print(grammar_df)
        
        return grammar_df_lex, grammar_df_syn, grammar_df_full, clips_lex, clips_syn, clips_full
        
    #------------------------------------------------------------------
    def learn_final_steps(self, input_data, forgetting_rounds, increments, grammar_df_lex, grammar_df_syn, grammar_df_full, clips_lex, clips_syn, clips_full):
    
        #First, a final round of forgetting where all constructions are old
        #Forgetting for lexical grammar
        forget_name_lex = os.path.join(self.out_dir, self.nickname + ".lex.grammar_forgetting.csv")
        if not os.path.exists(forget_name_lex):
            grammar_df_lex, clips_lex = self.forget_constructions(grammar_df_lex.loc[:,"Chunk"], input_data, threshold = 1, adjustment = 0.20, 
                                                                rounds = forgetting_rounds, increment_size = increments, name = "lex", clips = clips_lex, temp_grammar = grammar_df_lex)
            grammar_df_lex.to_csv(forget_name_lex)
            self.Load.save_file(clips_lex, self.nickname+".clips_lex_forgetting.p")
        else:
            grammar_df_lex = pd.read_csv(forget_name_lex, index_col = 0)
            clips_lex = self.Load.load_file(self.nickname+".clips_lex_forgetting.p")
                
        print(grammar_df_lex)

        #Forgetting for syntactic grammar
        forget_name_syn = os.path.join(self.out_dir, self.nickname + ".syn.grammar_forgetting.csv")
        if not os.path.exists(forget_name_syn):
            grammar_df_syn, clips_syn = self.forget_constructions(grammar_df_syn.loc[:,"Chunk"], input_data, threshold = 1, adjustment = 0.20, 
                                                            rounds = forgetting_rounds, increment_size = increments, name = "syn", clips = clips_syn, temp_grammar = grammar_df_syn)
            grammar_df_syn.to_csv(forget_name_syn)
            self.Load.save_file(clips_syn, self.nickname+".clips_syn_forgetting.p")
        else:
            grammar_df_syn = pd.read_csv(forget_name_syn, index_col = 0)
            clips_syn = self.Load.load_file(self.nickname+".clips_syn_forgetting.p")
                
        print(grammar_df_syn)
            
        #Forgetting for full grammar
        forget_name_full = os.path.join(self.out_dir, self.nickname + ".full.grammar_forgetting.csv")
        if not os.path.exists(forget_name_full):
            grammar_df_full, clips_full = self.forget_constructions(grammar_df_full.loc[:,"Chunk"], input_data, threshold = 1, adjustment = 0.20, 
                                                            rounds = forgetting_rounds, increment_size = increments, name = "full", clips = clips_full, temp_grammar = grammar_df_full)
            grammar_df_full.to_csv(forget_name_full)
            self.Load.save_file(clips_full, self.nickname+".clips_full_forgetting.p")
        else:
            grammar_df_full = pd.read_csv(forget_name_full, index_col = 0)
            clips_full = self.Load.load_file(self.nickname+".clips_full_forgetting.p")
                
        print(grammar_df_full)
        
        #Clustering lexical constructions
        lex_cluster_examples_file = self.nickname + ".grammar_lex_clusters_examples.txt"
            
        if not os.path.exists(os.path.join(self.out_dir, lex_cluster_examples_file)):
            print("Starting to cluster lexical constructions.")
                
            lex_clusters_constructions_df = self.get_construction_similarity(grammar_df_lex.loc[:,"Chunk"].tolist())
            print("\t Getting examples for token similarity.")
            examples_dict = self.print_examples(grammar = grammar_df_lex.loc[:,"Chunk"], input_file = input_data, output = False, n = 25, send_back=True)
            lex_cluster_df = self.get_token_similarity(lex_clusters_constructions_df, examples_dict)
            lex_cluster_df.loc[:,"Construction"] = self.decode(lex_cluster_df.loc[:,"Chunk"].values, clips = clips_lex)
            print(lex_cluster_df)
            lex_cluster_df.to_csv(os.path.join(self.out_dir, self.nickname + ".grammar_lex_clusters.csv"))
            #Save examples
            self.print_examples_clusters(examples_dict, lex_cluster_df, clips_lex, output_file = lex_cluster_examples_file)
       
        #Clustering syntactic constructions
        syn_cluster_examples_file = self.nickname + ".grammar_syn_clusters_examples.txt"
            
        if not os.path.exists(os.path.join(self.out_dir, syn_cluster_examples_file)):
            print("Starting to cluster syntactic constructions.")
                
            syn_clusters_constructions_df = self.get_construction_similarity(grammar_df_syn.loc[:,"Chunk"].tolist())
            print("\t Getting examples for token similarity.")
            examples_dict = self.print_examples(grammar = grammar_df_syn.loc[:,"Chunk"], input_file = input_data, output = False, n = 25, send_back=True)
            syn_cluster_df = self.get_token_similarity(syn_clusters_constructions_df, examples_dict)
            syn_cluster_df.loc[:,"Construction"] = self.decode(syn_cluster_df.loc[:,"Chunk"].values, clips = clips_syn)
            print(syn_cluster_df)
            syn_cluster_df.to_csv(os.path.join(self.out_dir, self.nickname + ".grammar_syn_clusters.csv"))
            #Save examples
            self.print_examples_clusters(examples_dict, syn_cluster_df, clips_syn, output_file = syn_cluster_examples_file)
                
        #Clustering full constructions
        full_cluster_examples_file = self.nickname + ".grammar_full_clusters_examples.txt"
           
        if not os.path.exists(os.path.join(self.out_dir, full_cluster_examples_file)):
            print("Starting to cluster full constructions: " + str(len(grammar_df_full)))
            
            full_clusters_constructions_df = self.get_construction_similarity(grammar_df_full.loc[:,"Chunk"].tolist())
            print("\t Getting examples for token similarity.")
            examples_dict = self.print_examples(grammar = grammar_df_full.loc[:,"Chunk"], input_file = input_data, output = False, n = 25, send_back=True)
            full_cluster_df = self.get_token_similarity(full_clusters_constructions_df, examples_dict)
            full_cluster_df.loc[:,"Construction"] = self.decode(full_cluster_df.loc[:,"Chunk"].values, clips = clips_full)
            print(full_cluster_df)
            full_cluster_df.to_csv(os.path.join(self.out_dir, self.nickname + ".grammar_full_clusters.csv"))
            #Save examples
            self.print_examples_clusters(examples_dict, full_cluster_df, clips_full, output_file = full_cluster_examples_file)
            
        return lex_cluster_df, syn_cluster_df, full_cluster_df, clips_lex, clips_syn, clips_full
     
    #------------------------------------------------------------------
        
    def process_grammar(self, input_data, grammar_type = "full", get_examples = True):
    
        #Lexical only
        if grammar_type == "lex":
            self.Load.data = [[(unit[0], -1, -1) for unit in line] for line in self.Load.actual_data]
        #Syntax only
        if grammar_type == "syn":
            self.Load.data = [[(-1, unit[1], -1) for unit in line] for line in self.Load.actual_data]
        #Full lex/syn/sem
        elif grammar_type == "full":
            self.Load.data = self.Load.actual_data 

        #Get pairwise association with Delta P
        association_file = os.path.join(self.out_dir, self.nickname + "." + grammar_type + ".association.gz")
        if not os.path.exists(association_file):
            self.Load.association_df = self.get_association(freq_threshold = self.Load.min_count, normalization = self.normalization, grammar_type = grammar_type, lex_only = False)
            self.Load.association_df.to_csv(association_file, compression = "gzip")
        else:
            self.Load.association_df = pd.read_csv(association_file, index_col = 0)
            
        #Now print association data
        print(self.Load.association_df)
        
        #Convert to dict
        self.Load.assoc_dict = self.get_association_dict(self.Load.association_df)

        #Set grammar output filenames
        cost_file = os.path.join(self.out_dir, self.nickname + "." + grammar_type + ".grammar_costs.csv")
        slot_cost_file = os.path.join(self.out_dir, self.nickname + "." + grammar_type + ".slot_costs.csv")
        
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
        grammar_df = best_cost_df
        
        #Grammar file name
        grammar_file = os.path.join(self.out_dir, self.nickname + "." + grammar_type + ".grammar_clipping.csv")
        
        #Check if grammar file exists
        if not os.path.exists(grammar_file):
        
            #Clip constructions together
            grammar, self.Load.clips = self.clip_constructions(grammar_df, self.Load.min_count)
            
            #Get costs for new grammar
            print("Recalculating encoding costs")
            mdl = Minimum_Description_Length(self.Load, self.Parse)
            grammar_cost, grammar_df = mdl.get_grammar_cost(grammar)
            grammar_df.loc[:,"Construction"] = self.decode(grammar_df.loc[:,"Chunk"].values)
            slot_df = mdl.cost_df
            
            #Save
            grammar_df.to_csv(grammar_file)
            self.Load.save_file(self.Load.clips, self.nickname + "." + grammar_type + ".grammar_clipping_indexes.p")
        
        #Load grammar
        else:
            print("Loading clipped grammar")
            grammar_df = pd.read_csv(grammar_file, index_col = 0)
            self.Load.clips = self.Load.load_file(self.nickname + "." + grammar_type + ".grammar_clipping_indexes.p")
            
        print(grammar_df)

        #Get examples if requested
        if get_examples == True:
            example_file = os.path.join(self.out_dir, self.nickname + "." + grammar_type + ".examples.txt")
            if not os.path.exists(example_file):
                self.print_examples(grammar = grammar_df.loc[:,"Chunk"].values, input_file = input_data, output = self.nickname + "." + grammar_type + ".examples.txt", n = 100)
                
        return grammar_df, self.Load.clips

    #------------------------------------------------------------------
    def clip_constructions(self, grammar_df, min_count):
    
        #First generate all possible merged constructions, recursively
        grammar = {}
        print("Preparing for clipping search")
        
        #Iterate over grammar to get constructions and frequency
        for row in grammar_df.itertuples():
            chunk = row[1]
            freq = row[2]

            #Input may be a string rather than tuple
            if isinstance(chunk, str):
                chunk = eval(chunk)
            
            #Add to dict
            grammar[chunk] = freq

        #Get starting size
        starting_size = len(grammar)
        
        #Dictionary for saving clip index
        clips = {}
        reject_list = []
        current_search = False
        
        #Continue merging until no new constructions
        while True:
            
            #Initialize for new clipped constructions
            round_counter = 0
            new_constructions = {}
            candidate_pool = []
            clip_pool = []
            
            #Double loop for comparing constructions
            grammar_list = list(grammar.keys())
            
            #Multi-process by construction
            starting = time.time()
            
            if current_search == False:
                print("\t\tChecking that there are no duplicates (a full grammar search)")
                results = list(ct.concat([[(i, j) for j in range(len(grammar_list)) if j < i] for i in range(len(grammar_list))]))
            else:
                print("\t\tChecking in current search with " + str(len(current_search)))
                results = list(ct.concat([[(i, j) for j in range(len(grammar_list))] for i in range(len(grammar_list)) if grammar_list[i] in current_search]))
            
            print("\t\tTotal of " + str(len(results)) + " pairs to check.")
            pool_instance = mp.Pool(processes = max(20, mp.cpu_count()), maxtasksperchild = None)
            results = pool_instance.map(partial(process_clipping, grammar_list = grammar_list), results, chunksize = 10000)
            pool_instance.close()
            pool_instance.join()            
            
            print("\t\tNow Reducing. Size of results: " + str(len(results)) + " and size of grammar: " + str(len(grammar_list)))
            results = [x for x in results if x != None]
            results = list(set(results))
            
            print("\t\tMerging " + str(len(results)) + " search results.")
            #Separate into candidates and clip indexes
            candidate_pool = [x[0] for x in results]
            clip_pool = [x[1] for x in results]
            del results
            print("\t\tDone in total: " + str(time.time() - starting))
            
            #Done with loop; now keep only observed second-order constructions
            #Parse candidates in data
            starting = time.time()
            print("\t\tFinished generating clips; now parsing " + str(len(candidate_pool)) + " possible clips.")
            candidate_pool = list(ct.partition_all(20000, candidate_pool))
            clip_pool = list(ct.partition_all(20000, clip_pool))
            
            #Reformat for multi-processing
            input_tuples = []
            for i in range(len(candidate_pool)):
                input_tuples.append((candidate_pool[i], clip_pool[i]))
                
            del candidate_pool
            del clip_pool
            
            pool_instance = mp.Pool(processes = min(15, mp.cpu_count()), maxtasksperchild = 1)
            output = pool_instance.map(partial(process_clipping_parsing, data = self.Load.data, min_count = min_count), input_tuples, chunksize = 1)
            pool_instance.close()
            pool_instance.join()   
            
            del input_tuples
            
            #Combine from multi-processing
            for i in range(len(output)):
                temp_new_constructions = output[i][0]
                temp_clips = output[i][1]
                new_constructions = ct.merge(new_constructions, temp_new_constructions)
                clips = ct.merge(clips, temp_clips)
                
            del output
            
            #Count new constructions
            round_counter = len(new_constructions)    
            print("\t\tDone in total: " + str(time.time() - starting))
            
            #Merge and display results
            print("\t\t New clipped constructions this round: " + str(round_counter))
            grammar = ct.merge(grammar, new_constructions)
            current_search = list(new_constructions.keys())
            
            #End loop check
            if round_counter == 0:
                break
                    
        print("Finished clipping constructions, from " + str(starting_size) + " to " + str(len(grammar)))

        return grammar, clips
    
    #------------------------------------------------------------------
    def decode(self, constructions, clips = None):
    
        #No specific clips passed, use default
        if clips == None:
            clips = self.Load.clips
            
        return_constructions = []
        
        #Iterate over items in grammar
        for construction in constructions:
        
            #Decode current construction
            construction = self.Load.decode_construction(construction, clips)
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
            
            print("\tStarting delta ", delta_threshold)
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
    def print_examples(self, grammar, input_file, n, output = False, send_back = False):
  
        output_dict = {} #For returning examples
        n = 50
        
        #Temp file if necessary
        if output == False:
            output = "temp.txt"
        
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
                
                #Input may be a string rather than tuple
                if isinstance(x, str):
                    x = eval(x)
                    
                #Prune to actual constraints
                construction = self.Load.decode_construction(x)

                #Example holder
                output_dict[x] = []
                
                #print(i, construction)
                if output != False:
                    fw.write(str(i) + "\t")
                    fw.write(construction)
                    fw.write("\n")
                
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
                                    
                                    if output != False:
                                        fw.write("\t" + str(counter) + "\t" + str(text) + "\n")
                                    
                                    if send_back == True:
                                        output_dict[x].append(" ".join(text))
                        
                        #Stop looking for examples at threshold
                        if counter > n:
                            break
                
                #End of examples for this construction
                fw.write("\n\n")
        
        #Return if necessary
        if send_back == True:
            return output_dict
    #-------------------------------------------------------------------------------
    
    def print_examples_clusters(self, examples_dict, clusters_df, clips = None, output_file = None):
    
        #Add default output
        if output_file == None:
            output_file = self.nickname + ".grammar_clusters.txt"
        #add path
        output_file = os.path.join(self.out_dir, output_file)
          
        #open file for writing
        with codecs.open(output_file, "w", encoding = "utf-8") as fw:
        
            #Iterate over type clusters then token clusters
            for cluster_type, cluster1_df in clusters_df.groupby("Type Cluster"):
                for cluster_token, cluster2_df in cluster1_df.groupby("Token Cluster"):
            
                    #Now look at each construction
                    for construction in cluster2_df.loc[:,"Chunk"].values:
                        
                        examples = examples_dict[construction]
                        writeable = self.Load.decode_construction(construction, clips = clips)
                        
                        fw.write("Type_Cluster:")
                        fw.write(str(cluster_type))
                        fw.write(" Token_Cluster:")
                        fw.write(str(cluster_token))
                        fw.write(" Constructions: [")
                        fw.write(writeable)
                        fw.write("]\n")
                        
                        for example in examples:
                            fw.write("\t")
                            fw.write(example)
                            fw.write("\n")
                        
                        fw.write("\n")

        return
    
    #------------------------------------------------------------------------------

    def get_association(self, freq_threshold = 1, normalization = True, grammar_type = "full", lex_only = False):
        
        #For smoothing, get discounts by constraint type
        if self.normalization == True:
            discount_dict = self.Association.find_discounts(self.Load.data)
            self.Load.save_file(discount_dict, self.nickname+ "." + grammar_type + ".discounts.p")
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
            word1 = str(row[1])
            word2 = str(row[2])
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
    
    def get_construction_similarity(self, grammar):
        
        print("\t Starting construction similarity")
        #Input may be a string rather than tuple
        grammar = [eval(chunk) if isinstance(chunk, str) else chunk for chunk in grammar]
        
        #Build np matrix for similarity
        similarity_matrix = np.array([[self.construction_similarity(construction1, construction2) for construction2 in grammar] for construction1 in grammar])
        print("\t Finished constructions similarity: " + str(similarity_matrix.shape))
        
        #Cluster
        cluster_df = self.Word_Classes.learn_construction_categories(grammar, similarity_matrix)

        return cluster_df
 
    #-----------------------------------------------
    
    def get_token_similarity(self, grammar_df, examples_dict, n_chunks = 100):
    
        #Initialize results holder
        results = []
    
        #Restrict the search to within type-based clusters
        for cluster, cluster_df in grammar_df.groupby("Cluster"):
            
            print("\t Starting token cluster " + str(cluster))

            #Input may be a string rather than tuple
            grammar = [eval(chunk) if isinstance(chunk, str) else chunk for chunk in cluster_df.loc[:,"Chunk"].tolist()]
            
            similarity_matrix = []
            
            #Build one construction at a time
            print("\t Starting token similarity")
            construction_pairs = list(ct.concat([[(construction1, construction2) for construction2 in grammar] for construction1 in grammar]))
  
            #Compare each construction with all other construction examples
            pool_instance = mp.Pool(processes = mp.cpu_count(), maxtasksperchild = 1)
            similarity_matrix = pool_instance.map(partial(token_similarity, examples_dict = examples_dict), construction_pairs, chunksize = n_chunks)
            pool_instance.close()
            pool_instance.join()
            
            #Convert to right dimensions
            similarity_matrix = list(ct.partition_all(len(grammar), similarity_matrix))
            
            #Convert to np matrix
            similarity_matrix = np.array(similarity_matrix)            
            print("\t Finished token similarity: " + str(similarity_matrix.shape))

            #Cluster
            cluster_df = self.Word_Classes.learn_construction_categories(grammar, similarity_matrix, num_clusters = range(int(len(grammar)/10)+11, 10, -10))
            cluster_df.loc[:,"Type Cluster"] = cluster
            results.append(cluster_df)
            
        cluster_df = pd.concat(results)
        cluster_df.columns = ["Chunk", "Token Cluster", "Type Cluster"]
        cluster_df.sort_values(by = ["Type Cluster", "Token Cluster"], ascending=True, inplace=True)
        
        return cluster_df
    
    #---------------------------------------------------
    
    def construction_similarity(self, construction1, construction2):

        #Exact match
        if construction1 == construction2:
            overlap = 0.0
            
        #Or fall back to highest overlap
        else:
            #Sequence matching algorithm (by slot constraints)
            s = difflib.SequenceMatcher(None, construction1, construction2)
            length = max(len(construction1), len(construction2))
            overlap = sum([x[2] for x in s.get_matching_blocks()]) / float(length)
            #Convert similarity to distance
            overlap = 1 - overlap

        return overlap

    #-----------------------------------------------    
    
    def forget_constructions(self, grammar, input_data, threshold = 1, adjustment = 0.20, rounds = 30, increment_size = 50000, name = "full", clips = False, temp_grammar = []):

        #Input may be a string rather than tuple
        grammar = [eval(chunk) if isinstance(chunk, str) else chunk for chunk in grammar]
        #grammar = [str(constraints)
        
        #Prepare existing grammar for reduced weight reduction
        if len(temp_grammar) > 5:
            temp_grammar = temp_grammar.loc[:,"Chunk"].tolist()
            if isinstance(temp_grammar[0], str):
                temp_grammar = [eval(chunk) for chunk in temp_grammar]
            
        #Clipped constructions decay at half the rate
        adjustment_clips = adjustment/2
            
        #Initialize round counter and construction weights
        print("Starting to prune grammar with construction forgetting.")
        round = 0
        weights = [1 for x in range(len(grammar))]
        history = [] #Save the forgetting rate
        
        #Track frequencies during pruning
        return_grammar = {} 
        for construction in grammar:
            return_grammar[construction] = 0
        
        #Iterate forgetting over specified number of rounds
        for i in range(rounds):
        
            print("\tStarting forgetting round " + str(round) + " with remaining constructions " + str(len(grammar)) + " for " + name)
            #Get current data
            data = self.Load.read_file(input_data, iterating = (self.max_words + (i*increment_size), self.max_words + ((i*increment_size)+increment_size)))
            data = [self.Load.clean(line) for line in data]

            round += 1
                            
            #Ensure enoguh data for pruning
            if len(data) > 10:
            
                #Get frequency for each construction
                detailed_grammar = detail_model(grammar)
                frequencies = self.Parse.parse_enriched(data, grammar = grammar, detailed_grammar = detailed_grammar)
                frequencies = np.sum(frequencies, axis=0)
                frequencies = frequencies.tolist()[0]
                
                #Update frequencies
                for i in range(len(grammar)):
                    return_grammar[grammar[i]] += frequencies[i]
                
                #Adjust weights given frequencies
                new_weights = []
                    
                #For each weight
                for i in range(len(weights)):
                    
                    #Get current weight and current construction
                    weight = weights[i]
                    construction = grammar[i]
                    
                    #Constructions from existing grammar are reduced less quickly
                    if temp_grammar != False:
                        if construction in temp_grammar:
                            temp_increment = 0.17
                        
                    #Second-order are reduced at half the rate
                    if construction in clips:
                        temp_increment = adjustment_clips
                    #First-order are reduced at full rate
                    else:
                        temp_increment = adjustment
   
                    #Return weight to 1 if above threshold 
                    if frequencies[i] >= threshold:
                        weight = 1
                    #Reduce weight if below threshold
                    else:
                        weight = weight - temp_increment
                            
                    #Store new weight
                    new_weights.append(weight)
                 
                #Replace weights array
                weights = new_weights
                
                #Prune grammar using weights
                grammar = [grammar[i] for i in range(len(grammar)) if weights[i] >= 0.0001]
                weights = [weights[i] for i in range(len(weights)) if weights[i] >= 0.0001]
                
                #Prune frequency dict
                allowed = lambda x: x in grammar
                return_grammar = ct.keyfilter(allowed, return_grammar)
                
                #Store rate info
                history.append([name, round, len(grammar)])
               
        #Finished with forgetting-based learning, now save grammar
        #Get costs for new grammar
        print("Finished. Now recalculating encoding costs")
        mdl = Minimum_Description_Length(self.Load, self.Parse)
        grammar_cost, grammar_df = mdl.get_grammar_cost(return_grammar)
        grammar_df.loc[:,"Construction"] = self.decode(grammar_df.loc[:,"Chunk"].values, clips = clips)
        
        #Save history
        history = pd.DataFrame(history, columns = ["Type", "Round", "Grammar Size"])
        history.to_csv(os.path.join(self.out_dir, self.nickname + ".forgetting_rates." + name + ".csv"))
        
        #Get reduced clips
        new_clips = {}
        for key in clips:
            if key in grammar:
                new_clips[key] = clips[key]
                
        print("\t\tValidating: grammar and clips: " + str(len(grammar_df)) + " and " + str(len(new_clips)))

        return grammar_df, new_clips
    #-----------------------------------------------