import os
from setuptools import setup, find_packages

# Utility function to read the README file.

def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name = "c2xg",
	version = "0.2",
	author = "Jonathan Dunn",
	author_email = "jdunn8@iit.edu",
	description = ("Learn, vectorize, and annotate Construction Grammars"),
	license = "LGPL 3.0",
	keywords = "grammar induction, unsupervised language processing, construction grammar, cognitive linguistics",
	url = "http://www.c2xg.io",
	packages = find_packages(exclude=["*.pyc", "__pycache__"]),
	package_data={'': ['c2xg.files_data.*']},
	install_requires=["cytoolz",
						"gensim",
						"matplotlib",
						"seaborn",
						"numexpr",
						"numpy",
						"pandas",
						"scipy",
						"sklearn"
						],
	include_package_data=True,
	long_description=read('README.md'),
	)