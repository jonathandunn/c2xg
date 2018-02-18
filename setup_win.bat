python setup.py sdist
python setup.py bdist_wheel
@RD /S /Q "C2xG.egg-info"
@RD /S /Q "build"