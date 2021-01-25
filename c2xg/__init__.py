import os
import sys

from .modules.Association import Association
from .modules.Candidates import Candidates
from .modules.Encoder import Encoder
from .modules.Loader import Loader
from .modules.MDL_Learner import MDL_Learner
from .modules.Parser import Parser
from .modules.Word_Classes import Word_Classes

from .modules.clustering.pyc_xmeans import xmeans
from .modules.clustering.pyc_center_initializer import kmeans_plusplus_initializer
from .modules.clustering.pyc_encoder import type_encoding
from .modules.clustering.pyc_utils import euclidean_distance_sqrt, euclidean_distance
from .modules.clustering.pyc_utils import list_math_addition_number, list_math_addition, list_math_division_number
from .modules.clustering.pyc_utils import euclidean_distance

from .modules.rdrpos_tagger.SCRDRlearner.SCRDRTree import SCRDRTree as SCRDRTree
from .modules.rdrpos_tagger.Utility.Utils import getWordTag
from .modules.rdrpos_tagger.SCRDRlearner.Node import Node
from .modules.rdrpos_tagger.SCRDRlearner.Object import getObjectDictionary
from .modules.rdrpos_tagger.SCRDRlearner.Object import FWObject
from .modules.rdrpos_tagger.InitialTagger.InitialTagger import initializeCorpus, initializeSentence
from .modules.rdrpos_tagger.Utility.Config import NUMBER_OF_PROCESSES, THRESHOLD
from .modules.rdrpos_tagger.Utility.Utils import getWordTag, getRawText, readDictionary
from .modules.rdrpos_tagger.Utility.LexiconCreator import createLexicon
from .modules.rdrpos_tagger.Utility.Utils import readDictionary
from .modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import unwrap_self_RDRPOSTagger
from .modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import printHelp
from .modules.rdrpos_tagger.SCRDRlearner.SCRDRTreeLearner import SCRDRTreeLearner
from .modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger

sys.setrecursionlimit(100000)
sys.path.append(os.path.abspath(""))