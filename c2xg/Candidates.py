import time
import operator
import os
import numpy as np
import pandas as pd
import cytoolz as ct
import multiprocessing as mp
from functools import partial
from numba import jit
from collections import defaultdict
from collections import deque
import operator
import difflib
import copy

from .Association import Association
from .Association import calculate_measures

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
            top_score = 0.0
            for candidate in self.candidate_stack[index]:
                current_score = self.get_score(candidate)
                if current_score > top_score:
                    top_score = current_score
                    top_candidate = candidate
                    
            self.candidates.append(top_candidate)
        
        #PART 3: Horizontal pruning to find nested candidates
        to_pop = []
        for i in range(len(self.candidates)):
            for j in range(len(self.candidates)):
                if i != j and j > i:
                    candidate1 = self.candidates[i]
                    candidate2 = self.candidates[j]
                    
                    s = difflib.SequenceMatcher(None, candidate1, candidate2)
                    largest = max([x[2] for x in s.get_matching_blocks()])
                    
                    if largest > 2:
                        shortest = min(len(candidate1), len(candidate2))
                        
                        if float(largest / shortest) < 0.75:
                            score1 = self.get_score(candidate1)
                            score2 = self.get_score(candidate2)
                            
                            if score1 < score2:
                                if candidate1 not in to_pop:
                                    to_pop.append(candidate1)
                            elif candidate2 not in to_pop:
                                to_pop.append(candidate2)
        
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
                        self.candidate_stack[i - len(current_path) + 1].append(current_path)
   
            return
            
    #--------------------------------------------------------------#
    
    def get_score(self, current_candidate):
    
        total_score = 0.0
        
        for pair in ct.sliding_window(2, current_candidate):
        
            try:
                current_dict = self.association_dict[pair[0]][pair[1]]
            except:
                current_dict = {}

            current_score = max(current_dict["RL"], current_dict["LR"])
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
        
    
    #------------------------------------------------------------------
    
    def get_candidates(self, input_data):
        
        candidates = []
        starting = time.time()
        
        #Initialize Beam Search class
        BS = BeamSearch(self.delta_threshold, self.freq_threshold, self.association_dict)
        
        #Beam Search extraction
        candidates = list(ct.concat([BS.beam_search(x) for x in input_data]))

        #Count each candidate, get dictionary with candidate frequencies
        candidates = ct.frequencies(candidates)
        print("\t" + str(len(candidates)) + " candidates before pruning.")
        
        #Reduce candidates
        above_zero = lambda x: x > self.freq_threshold
        candidates = ct.valfilter(above_zero, candidates)        
            
        #Print time and number of remaining candidates
        print("\t" + str(len(candidates)) + " candidates in " + str(time.time() - starting) + " seconds.")
    
        return candidates

    #--------------------------------------------------------------#