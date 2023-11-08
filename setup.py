import os
from setuptools import setup, find_packages

# Utility function to read the README file.

def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name = "c2xg",
	version = "1.0",
	author = "Jonathan Dunn",
	author_email = "jonathan.dunn@canterbury.ac.nz",
	description = ("Construction Grammars for Natural Language Processing and Computational Linguistics"),
	license = "LGPL 3.0",
	keywords = "grammar induction, syntax, cxg, unsupervised learning, natural language processing, computational linguistics, construction grammar, cognitive linguistics, usage-based grammar",
	url = "http://www.c2xg.io",
	packages = find_packages(exclude=["*.pyc", "__pycache__"]),
	install_requires=["clean-text", #==0.6.0"
			  "cytoolz", #==0.12.2"
			  "gensim", #==4.3.0"
			  "kmedoids", #==0.4.3"
			  "matplotlib", #==3.7.2"
			  "numpy", #==1.25.2"
			  "pandas", #==2.0.3"
			  "scipy", #==1.11.1"
			  "scikit-learn", #==1.3.0"
			  "statsmodels"], #==0.14.0"],
	include_package_data=True,
	long_description=read('README.md'),
	long_description_content_type='text/markdown',
	)
