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
	description = ("Learn, vectorize, and annotate Construction Grammars"),
	license = "LGPL 3.0",
	keywords = "grammar induction, unsupervised language processing, construction grammar, cognitive linguistics, usage-based grammar",
	url = "http://www.c2xg.io",
	packages = find_packages(exclude=["*.pyc", "__pycache__"]),
	package_data={'': ['c2xg.data.*']},
	install_requires=["cytoolz",
						"gensim",
						"matplotlib",
						"seaborn",
						"numexpr",
						"numba",
						"numpy",
						"pandas",
						"scipy",
						"sklearn",
						"clean-text",
						],
	include_package_data=True,
	long_description=read('README.md'),
	)
