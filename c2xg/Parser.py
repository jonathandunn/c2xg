import time
import copy
import numpy as np
import cytoolz as ct
import multiprocessing as mp
from functools import partial
from scipy import sparse
from collections import defaultdict

#--------------------------------------------------------------#
def parse_for_examples(construction, line):

    indexes = [-1]
    matches = 0
    
    #Iterate over line from left to right
    for i in range(len(line)):
        
        unit = line[i]

        #Check if the first unit matches, to merit further consideration
        if construction[0][1] == unit[construction[0][0]-1]:
                        
            match = True    #Initiate match flag to True

            #Check each future unit in candidate
            for j in range(1, len(construction)):
                            
                #If we reach the padded part of the construction, break it off
                if construction[j] == (0,0):
                    break
                            
                #If this unit doesn't match, stop looking
                if i+j < len(line):
                    if line[i+j][construction[j][0] - 1] != construction[j][1]:
                                        
                        match = False
                        break
                        
                #This construction is longer than the remaining line
                else:
                    match = False
                    break

            #Done with candidate
            if match == True:
                matches += 1
                indexes.append(i)    #Save indexes covered by construction match
                
    return construction, indexes[1:], matches

#--------------------------------------------------------------#

def parse_mdl_support(construction, lines):

    indexes = [-1]
    matches = 0
    
    #Iterate over input lines
    for k in range(len(lines)):
    
        #Iterate over line from left to right
        line = lines[k]
        
        for i in range(len(line)):
            
            unit = line[i]

            #Check if the first unit matches, to merit further consideration
            if construction[0][1] == unit[construction[0][0]-1]:
                            
                match = True    #Initiate match flag to True

                #Check each future unit in candidate
                for j in range(1, len(construction)):
                                
                    #If we reach the padded part of the construction, break it off
                    if construction[j] == (0,0):
                        break
                                
                    #If this unit doesn't match, stop looking
                    if i+j < len(line):
                        if line[i+j][construction[j][0] - 1] != construction[j][1]:
                                            
                            match = False
                            break
                            
                    #This construction is longer than the remaining line
                    else:
                        match = False
                        break

                #Done with candidate
                if match == True:
                    matches += 1
                    indexes.append(tuple((k, list(range(i, i + len(construction))))))    #Save indexes covered by construction match
          
    return construction, indexes[1:], matches

#--------------------------------------------------------------#

def parse_clipping_support(line, constructions):

    indexes = [-1]
    matches = []
    
    #Iterate over input constructions in current line
    for k in range(len(constructions)):
    
        construction = constructions[k]
        current_matches = 0
    
        #Iterate over line from left to right
        for i in range(len(line)):
            
            unit = line[i]

            #Check if the first unit matches, to merit further consideration
            if construction[0][1] == unit[construction[0][0]-1]:
                            
                match = True    #Initiate match flag to True

                #Check each future unit in candidate
                for j in range(1, len(construction)):
                                
                    #If we reach the padded part of the construction, break it off
                    if construction[j] == (0,0):
                        break
                                
                    #If this unit doesn't match, stop looking
                    if i+j < len(line):
                        if line[i+j][construction[j][0] - 1] != construction[j][1]:
                                            
                            match = False
                            break
                            
                    #This construction is longer than the remaining line
                    else:
                        match = False
                        break

                #Done with candidate
                if match == True:
                    current_matches += 1
                    indexes.append(tuple((k, list(range(i, i + len(construction))))))    #Save indexes covered by construction match
        
        #Add frequency of this construction
        matches.append(current_matches)
        
    return indexes[1:], matches

#--------------------------------------------------------------#
def _get_candidates( unit, grammar ) : 
        
    # Check for: if construction[0][1] == unit[construction[0][0]-1]:
    # model_expanded[ (possible elem[0][0]) ][ elem[0][1] ].append( elem ) 
    all_plausible = list()
    X, grammar = grammar
    for elem_0_0 in X : 
        plausible = grammar[ elem_0_0 ][ unit[ elem_0_0 - 1 ] ]
        plausible = [ ( i[0], i[2] ) for i in plausible  ]
        all_plausible += plausible
    
    return all_plausible 
                
#--------------------------------------------------------------#

def parse_fast( line, grammar, grammar_len, sparse_matches=False ) : 
        
    matches = None
    if sparse_matches : 
        matches = list()
    else : 
        matches = [0 for x in range( grammar_len )]
    
    #Iterate over line from left to right
    for line_index in range( len( line ) ) : 
        unit = line[ line_index ] 
        #Get plausible candidates 
        candidates = _get_candidates( unit, grammar )
        for k in range(len(candidates)) : 
            construction, grammar_index = candidates[k]
            ## Below this is the same as the parse() function
            match = True    #Initiate match flag to True
            #Check each future unit in candidate
            for j in range(1, len(construction)):
                #If we reach the padded part of the construction, break it off
                if construction[j] == (0,0):
                    break
                            
                #If this unit doesn't match, stop looking
                if line_index+j < len(line):
                    if line[line_index+j][construction[j][0] - 1] != construction[j][1]:
                        match = False
                        break
                        
                #This construction is longer than the remaining line
                else:
                    match = False
                    break
                
            #Done with candidate
            if match == True:
                if sparse_matches : 
                    matches.append( grammar_index ) 
                else : 
                    matches[grammar_index] += 1
        
    return matches 
#--------------------------------------------------------------#

def parse(line, grammar):

    matches = [0 for x in range(len(grammar))]

    #Iterate over line from left to right
    for i in range(len(line)):
            
        unit = line[i]

        #Check for plausible candidates moving forward
        for k in range(len(grammar)):

            construction = grammar[k]    #Get construction by index
            
            #Check if the first unit matches, to merit further consideration
            if construction[0][1] == unit[construction[0][0]-1]:
                        
                match = True    #Initiate match flag to True

                #Check each future unit in candidate
                for j in range(1, len(construction)):
                            
                    #If we reach the padded part of the construction, break it off
                    if construction[j] == (0,0):
                        break
                            
                    #If this unit doesn't match, stop looking
                    if i+j < len(line):
                        if line[i+j][construction[j][0] - 1] != construction[j][1]:
                                        
                            match = False
                            break
                        
                    #This construction is longer than the remaining line
                    else:
                        match = False
                        break

                #Done with candidate
                if match == True:
                    matches[k] += 1
    
    return matches
#--------------------------------------------------------------#

def detail_model(model): 

    ## Update model so we can access grammar faster ... 
    ## Want to make `if construction[0][1] == unit[construction[0][0]-1]` faster
    ## Dict on construction[0][1] which is self.model[i][0][1] (Call this Y)
    ## BUT unit[ construction[0][0] - 1 ] changes with unit ... 
    ## construction[0][0] values are very limited.  (call this X)
    ## dict[ construction[0][0] ][ construction[0][1] ] = list of constructions
        
    model_expanded = dict()

    X = list( set( [ model[i][0][0] for i in range(len(model)) ] ) )
        
    for x in X : 
        model_expanded[ x ] = defaultdict( list ) 
        this_x_elems = list()
        for k, elem in enumerate( model ) : 
            if elem[0][0] != x : 
                continue
            elem_trunc = [ i for i in elem if i != (0,0) ]
            model_expanded[ x ][ elem[0][1] ].append( ( elem, elem_trunc, k ) )
        
    return ( X, model_expanded ) 
    
#--------------------------------------------------------------#

def _validate(lines, grammar, grammar_detailed): 

    from tqdm import tqdm

    for line in tqdm( lines, desc="Validating" ) : 
        matches_parse      = parse(      line, grammar=grammar )
        matches_parse_fast = parse_fast( line, grammar=grammar_detailed, grammar_len=len(grammar), sparse_matches=False )
        print( sum( matches_parse_fast ), flush=True )
        assert matches_parse == matches_parse_fast 
    return 

#--------------------------------------------------------------#

class Parser(object):

    def __init__(self, Load = None):
    
        #Initialize Parser
        if Load != None:
            self.language = Load.language
        
        if Load != None:
            self.Load = Load
    
    #--------------------------------------------------------------#
    
    def parse_stream(self, files, grammar, detailed_grammar=None):
        
        for line in self.Encoder.load_stream(files):
            if not detailed_grammar is None :
                matches = parse_fast( line, grammar = detailed_grammar, grammar_len = len( grammar ), sparse_matches=False )
            else : 
                matches = parse(line, grammar)

            yield matches
                
    #--------------------------------------------------------------#
    def parse_examples(self, construction, line):
    
            construction_thing, indexes, matches = parse_for_examples(construction, line)
            return construction_thing, indexes, matches
    #--------------------------------------------------------------#
    
    def parse_enriched(self, lines, grammar, detailed_grammar = None):
    
        #Prepare grammar
        if detailed_grammar is None:
            detailed_grammar = detail_model(grammar)
            
        #Fast parsing
        lines = [sparse.coo_matrix(parse_fast(line, grammar = detailed_grammar, grammar_len = len(grammar))) for line in lines]
        lines = sparse.vstack(lines)

        return lines
    #--------------------------------------------------------------#
    
    def parse_mdl(self, lines, grammar):
        
        #Chunk array for workers
        total_count = len(lines)
    
        #Multi-process by construction
        pool_instance = mp.Pool(processes = mp.cpu_count(), maxtasksperchild = None)
        results = pool_instance.map(partial(parse_mdl_support, lines = lines), grammar, chunksize = 2500)
        pool_instance.close()
        pool_instance.join()

        #Find fixed max value for match indexes
        max_matches = max([len(indexes) for construction, indexes, matches in results])
        
        #Initialize lists
        construction_list = []
        indexes_list = []
        matches_list = []
        vector_list = []
        
        #Create fixed-length arrays
        for i in range(len(results)):
            construction, indexes, matches = results[i]

            vector_list.append(i)
            construction_list.append(construction)
            matches_list.append(matches)
            indexes_list.append(indexes)
    
        #results contains a tuple for each construction in the grammar (indexes[list], matches[int])
        return construction_list, indexes_list, np.array(matches_list), vector_list
        
    #---------------------------------------------------------------#
    
    def parse_clipping(self, lines, grammar):
        
        #Chunk array for workers
        total_count = len(lines)
    
        #Multi-process by construction
        pool_instance = mp.Pool(processes = mp.cpu_count(), maxtasksperchild = None)
        results = pool_instance.map(partial(parse_clipping_support, constructions = grammar), lines, chunksize = 10)
        pool_instance.close()
        pool_instance.join()
        
        #Initialize lists
        construction_list = grammar
        indexes_list = []
        matches_list = []
        
        #Create fixed-length arrays
        for i in range(len(results)):
            indexes, matches = results[i]
            matches_list.append(matches)
            indexes_list.append(indexes)
          
        matches_list = np.array(matches_list)
        matches_list = np.sum(matches_list, axis = 0)
    
        #results contains a tuple for each construction in the grammar (indexes[list], matches[int])
        return construction_list, indexes_list, matches_list
    