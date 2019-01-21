python setup.py sdist
python setup.py bdist_wheel
@RD /S /Q "c2xg.egg-info"
@RD /S /Q "build"