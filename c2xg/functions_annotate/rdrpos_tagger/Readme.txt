#1:	Please read the "License.txt" file before proceeding.

#2:	Please find the general architecture of RDRPOSTagger in our EACL2014 paper:

	@InProceedings{NguyenNPP2014,
	  author    = {Nguyen, Dat Quoc  and  Nguyen, Dai Quoc  and  Pham, Dang Duc  and  Pham, Son Bao},
	  title     = {{RDRPOSTagger: A Ripple Down Rules-based Part-Of-Speech Tagger}},
	  booktitle = {Proceedings of the Demonstrations at the 14th Conference of the European Chapter of the Association for Computational Linguistics},
	  year      = {2014},
	  pages     = {17--20},
	  url       = {http://www.aclweb.org/anthology/E14-2005}
	}

	Please cite our EACL2014 paper whenever RDRPOSTagger is used to produce published results or incorporated into other software!!!

#3:	FULL usage of RDRPOSTagger can be found at: http://rdrpostagger.sourceforge.net/

#4:	For a short description of the usage: please follow the "printHelp()" function in the "RDRPOSTagger.py" module in the "pSCRDRtagger" package, and the "printHelp()" method in the "RDRPOSTagger.java" class in the "jSCRDRtagger" package. Or simply run from command line/terminal:

	Example 1: pSCRDRtagger$ python RDRPOSTagger.py

	Example 2: jSCRDRtagger$ java RDRPOSTagger

#5:	To obtain faster tagging process in Python: set a higher value for the "NUMBER_OF_PROCESSES" variable in the "Config.py" module in the "Utility" package. The value should not larger than the number of CPU cores which your computer has.