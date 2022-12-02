import math
import time
import itertools
import random
import copy
from collections import deque
import cytoolz as ct
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
                      
#---------------------------------------------------------------------------
    
class Minimum_Description_Length(object):

    def __init__(self, Load, Parse):
    
        print("Initializing MDL Learner for this round (loading data).")
        
        #Initialize
        self.Parse = Parse
        self.Load = Load
        
        #Find cost of various slot constraints
        self.cost_df = self.get_constraint_cost()
        
        return
    
#---------------------------------------------------------------------------    
        
    def get_constraint_cost(self):
        
        #Get fixed units costs per representation type
        self.type_cost = -math.log2(float(1.0/3.0))
        cost_df = []
        
        #Get lexical probabilities and cost
        total_words = sum(self.Load.lexicon.values())
        in_lexicon = list(self.Load.cbow_df.loc[:,"Word"].values)
        cost_lex = {}
        
        #Iterate over the full lexicon
        for word in self.Load.lexicon:
            if word in in_lexicon:
                freq = self.Load.lexicon[word]
                prob = freq/float(total_words)
                cost = -math.log2(prob) + self.type_cost
                cost_lex[self.Load.word_dict[word]] = cost
                cost_df.append([1, self.Load.word_dict[word], cost])
        
        #Get cbow probabilities and cost
        cost_syn = {}
        #Iterate over categories
        for category, category_df in self.Load.cbow_df.groupby("Category"):
            cat_freq = 0
            #Add frequencies for each word in category
            for word in category_df.loc[:,"Word"].values:
                cat_freq += self.Load.lexicon[word]
            #Find the probability of this category 
            cost = -math.log2(float(cat_freq)/len(self.Load.cbow_df)) + self.type_cost
            cost_syn[category] = cost
            cost_df.append([2, category, cost])
         
        #Get sg probabilities and cost
        cost_sem = {}
        #Iterate over categories
        for category, category_df in self.Load.sg_df.groupby("Category"):
            cat_freq = 0
            #Add frequencies for each word in category
            for word in category_df.loc[:,"Word"].values:
                cat_freq += self.Load.lexicon[word]
            #Find the probability of this category
            cost = -math.log2(float(cat_freq)/len(self.Load.sg_df)) + self.type_cost
            cost_sem[category] = cost
            cost_df.append([3, category, cost])
        
        #Save slot encoding costs
        self.cost_lex = cost_lex
        self.cost_syn = cost_syn
        self.cost_sem = cost_sem
        cost_df = pd.DataFrame(cost_df, columns = ["Type", "Value", "Cost"])
        
        return cost_df
        
    #---------------------------------------------------------------------------

    def get_grammar_cost(self, chunks):

        #Get total of construction instances
        total = sum(chunks.values())
        chunk_cost = {}
        chunk_df = []
        print("Total observed constructions")
        
        #Find the cost for each potential construction
        for chunk in chunks:
            
            #First, more likely constructions should cost less
            prob = chunks[chunk] / float(total)
            cost = -math.log2(prob)
            chunk_cost[chunk] = {}
            chunk_cost[chunk]["Pointer"] = cost
            cost = 0
            
            #Second accumulate slot-specific constraints
            for constraint in chunk:
            
                #Lexical
                if constraint[0] == 1:
                    add = self.cost_lex[constraint[1]]
                #Syntactic
                elif constraint[0] == 2:
                    add = self.cost_syn[constraint[1]]
                #Semantic
                elif constraint[0] == 3:
                    add = self.cost_sem[constraint[1]]
                #Add to cost
                cost += add
                
            #Done with construction
            chunk_cost[chunk]["Encoding"] = cost
            chunk_df.append([chunk, chunks[chunk], chunk_cost[chunk]["Pointer"], chunk_cost[chunk]["Encoding"]])
        
        #Create readable df
        chunk_df = pd.DataFrame(chunk_df, columns = ["Chunk", "Frequency", "Pointer", "Encoding"])
            
        return chunk_cost, chunk_df
    
    #---------------------------------------------------------------------------
    
    def evaluate_grammar(self, grammar, grammar_cost, input_data = None):

        starting = time.time()  #For timing purposes
        
        #Use pre-loaded data or enrich a new corpus
        if input_data != None:
            data = self.Load.load(input_data)
        else:
            data = self.Load.data
        
        grammar_list = list(grammar.keys()) #Fix order of keys
        
        #Parse candidates to determine encoding cost
        candidates, indexes, matches, vector_list = self.Parse.parse_mdl(data, grammar_list)
        print("\tParsed " + str(len(data)) + " lines with " + str(len(grammar)) + " constructions in " + str(time.time() - starting) + " seconds.")

        #Add pre-calculated construction encoding cost
        starting = time.time()
        l1_list = []
        l2_list = []
        
        #Make empty array for total indexes covered
        covered_indexes = []
        for i in range(len(data)):
            covered_indexes.append([0 for x in range(len(data[i]))])
        
        #Get encoding cost for each construction
        for i in range(len(grammar_list)):
            construction = grammar_list[i]
            
            #L1 is the cost of the construction in the grammar
            l1_cost = grammar_cost[construction]["Encoding"]
            l1_list.append(l1_cost)
            
            #L2 is the cost of encoding uses of the construction in the corpus
            current_pointer_cost = grammar_cost[construction]["Pointer"]
            l2_cost = current_pointer_cost*matches[i]
            l2_list.append(l2_cost)
            
            #Add encoded indexes
            for index in indexes[i]:
                #Find the line of the current matches
                k = index[0]
                #Switch covered indexes to 1
                for m in list(ct.concat(index[1:])):
                    covered_indexes[k][m] = 1
                
        #Find unencoded indexes
        unencoded_indexes = 0
        encoded_indexes = 0
        for k in covered_indexes:
            unencoded_indexes += k.count(0)
            encoded_indexes += k.count(1)

        #Use unencoded indexes to get regret cost
        #Regret cost applied twice, once for encoding and once for grammar
        if unencoded_indexes > 0:
            unencoded_cost = -math.log2(float(1.0/(unencoded_indexes)))
            l2_regret_cost = (unencoded_cost * unencoded_indexes) * 2
        
        else:
            l2_regret_cost = 0
            
        #Sum the cost lists
        l1_cost = sum(l1_list)
        l2_match_cost = sum(l2_list)

        #Total all terms
        total_mdl = l1_cost + l2_match_cost + l2_regret_cost
                
        #DEBUGGING
        print("\t\tMDL: " + str(total_mdl))
        print("\t\tL1 Cost: " + str(l1_cost))
        print("\t\tL2 Match Cost: " + str(l2_match_cost))
        print("\t\tL2 Regret Cost: " + str(l2_regret_cost))
        print("\t\tEncoded: " + str(encoded_indexes))
        print("\t\tUnencoded: " + str(unencoded_indexes))
        
        return total_mdl, l1_cost, l2_match_cost, l2_regret_cost
        
    #---------------------------------------------------------------------------