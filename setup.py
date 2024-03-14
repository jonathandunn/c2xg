import os
from setuptools import setup, find_packages

# Utility function to read the README file.

def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name = "c2xg",
	version = "2.02",
	author = "Jonathan Dunn",
	author_email = "jedunn@illinois.edu",
	description = ("Construction Grammars for Natural Language Processing and Computational Linguistics"),
	license = "LGPL 3.0",
	keywords = "grammar induction, syntax, cxg, unsupervised learning, natural language processing, computational linguistics, construction grammar, cognitive linguistics, usage-based grammar",
	url = "http://www.c2xg.io",
	packages = find_packages(exclude=["*.pyc", "__pycache__"]),
	install_requires=[
              "clean-text >= 0.6.0",
			  "cytoolz >= 0.12.2",
			  "gensim >= 4.0.1",
			  "kmedoids >= 0.3.3",
			  "matplotlib >= 3.5.3",
			  "numpy >= 1.21.6",
			  "pandas >= 1.3.5",
			  "scipy >= 1.7.3",
			  "scikit-learn >= 1.0.2",
			  "statsmodels >= 0.13.5",
			 ],
    python_requires='>=3.7',
	include_package_data=True,
	long_description=read('README.md'),
	long_description_content_type='text/markdown',
	)
