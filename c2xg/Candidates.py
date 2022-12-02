import time
import operator
import os
import numpy as np
import pandas as pd
import cytoolz as ct
import multiprocessing as mp
from functools import partial
from collections import defaultdict
from collections import deque
import operator
import difflib
import copy

from .Parser import Parser

#--------------------------------------------------------------#

class BeamSearch(object):

    def __init__(self, delta_threshold, freq_threshold, association_dict):
        
        #Initialize empty candidate stack
        self.candidate_stack = defaultdict(list)
        self.candidates = []
        self.association_dict = association_dict
        self.delta_threshold = delta_threshold
        self.freq_threshold = freq_threshold
        
        return
    #--------------------------------------------------------------#
    
    def beam_search(self, line):

        self.candidates = []
        
        #PART 1: Search for candidates left-to-right across the line
        for i in range(len(line)):

            #Start path from each of the current slot-constraints
            for current_start in [(1, line[i][0]), (2, line[i][1]), (2, line[i][2])]:
                
                #Ignore out of vocabulary representations
                if current_start[1] != -1:

                    #Recursive search from each available path
                    self.recursive_beam(current_start, line, i, len(line))
         
        #PART 2: Evaluate candidate stack
        for index in self.candidate_stack.keys():
            
            #Search for each starting node
            top_score = 0.0
            
            #Get top total association, allowing for weak links
            for candidate in self.candidate_stack[index]:
                current_score = self.get_score(candidate)
                if current_score > top_score:
                    top_score = current_score
                    top_candidate = candidate
             
            #Keep only the best candidate for each node
            self.candidates.append(top_candidate)
            
        #Reduce duplicate candidates
        self.candidates = list(set(self.candidates))
        
        #PART 3: Horizontal pruning to find nested candidates
        to_pop = []
        
        #Check each combination of candidates
        for i in range(len(self.candidates)):
            for j in range(len(self.candidates)):
            
                #Only check later and larger pairs
                if j > i and len(self.candidates[i]) != len(self.candidates[j]):
                
                    #Get candidates
                    candidate1 = self.candidates[i]
                    candidate2 = self.candidates[j]
                    
                    #Convert to string for test
                    test1 = str(candidate1).replace("((","(").replace("))",")")
                    test2 = str(candidate2).replace("((","(").replace("))",")")
                    
                    #If nested, remove smaller
                    if test1 in test2 or test2 in test1:
                        if len(candidate1) > len(candidate2):
                            to_pop.append(candidate2)
                        elif len(candidate1) > len(candidate2):
                            to_pop.append(candidate1)
         
        #Remove smaller nested candidates
        self.candidates = [x for x in self.candidates if x not in to_pop]
        
        #Reset state to prepare for next line
        self.candidate_stack = defaultdict(list)

        return self.candidates
    #--------------------------------------------------------------#
    
    def recursive_beam(self, previous_start, line, i, line_length):

        #Progress down the line
        i += 1

        #Stop at the end
        if i < line_length:
        
                
            #For each available next path; examine each constraint on its own because overlapping constructions are possible
            for start in [(1, line[i][0]), (2, line[i][1]), (3, line[i][2])]:
                    
                #Create larger path if the input is not a root node
                #If previous_start has multiple slots, will join
                try:
                    previous_start = list(ct.concat(previous_start))
                #But single slot inputs will throw an error
                except Exception as e:
                    pass
                        
                #Join and reform into a tuple of (type, constraint)
                current_path = list(ct.concat([previous_start, start]))
                current_path = tuple(ct.partition(2, current_path))
                
                #Association is pairwise, so for longer sequences only look at final two slots
                if len(current_path) > 2:
                    test_path = current_path[-2:]
                else:
                    test_path = current_path
  
                #Get association statistics for the relevant pair
                try:
                    current_dict = self.association_dict[test_path[0]][test_path[1]]
                #Errors reflect missing pairs which are below the frequency threshold    
                except Exception as e:
                    current_dict = {"Max": 0.0, "Frequency": 0.0}
                 
                #If the current pair is above the frequency and association thresholds, continue this line of search
                if current_dict["Max"] > self.delta_threshold and current_dict["Frequency"] > self.freq_threshold:
                    
                    #Continue search
                    self.recursive_beam(current_path, line, i, line_length)
                                                            
                #Search is over, save candidate if possible
                else:
                    #Has to be between two and nine slots
                    if len(current_path) > 2 and len(current_path) < 10:
                                        
                        #Remove the weak link
                        current_path = current_path[0:-1]
                                    
                        #Add to candidate_stack
                        if current_path not in self.candidate_stack[i - len(current_path) + 1]:
                            self.candidate_stack[i - len(current_path) + 1].append(current_path)
   
            return
            
    #--------------------------------------------------------------#
    
    def get_score(self, current_candidate):
    
        #Initialize score
        total_score = 0.0
        
        #Iterate over pairs of slots constraints
        for pair in ct.sliding_window(2, current_candidate):
        
            #Accumulate both directions of association
            current_dict = self.association_dict[pair[0]][pair[1]]
            current_score = current_dict["LR"] +  current_dict["RL"]
            total_score += current_score

        return total_score
    #--------------------------------------------------------------#

class Candidates(object):

    def __init__(self, language, Load, association_dict = None, freq_threshold = 1, delta_threshold = 0.10):
    
        #Initialize
        self.language = language
        self.Load = Load
        self.freq_threshold = freq_threshold
        self.delta_threshold = delta_threshold
        self.association_dict = association_dict
        self.Parse = Parser(self.Load)
        
    #------------------------------------------------------------------
    
    def get_candidates(self, input_data):
        
        candidates = []
        starting = time.time()
        
        #Initialize Beam Search class
        BS = BeamSearch(self.delta_threshold, self.freq_threshold, self.association_dict)
        
        #Beam Search extraction
        candidates = list(ct.concat([BS.beam_search(x) for x in input_data]))
        print("Before duplicate removal: ", len(candidates))
        candidates = list(set(candidates))
        print("After duplicate removal: ", len(candidates))

        #Parse candidates in data because extraction won't estimate frequencies
        frequencies = np.array(self.Parse.parse_enriched(input_data, grammar = candidates))
        frequencies = np.sum(frequencies, axis=0)
        
        #Reduce candidates
        final_candidates = {}
        for i in range(len(candidates)):
            if frequencies[i] > self.freq_threshold:
                final_candidates[candidates[i]] = frequencies[i]
            
        #Print time and number of remaining candidates
        print("After frequency threshold: " + str(len(final_candidates)) + " in " + str(time.time() - starting) + " seconds.")
    
        return final_candidates

    #--------------------------------------------------------------#