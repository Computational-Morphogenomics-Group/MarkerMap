# MarkerMap

## Setup 

### MacOS 

- Clone the repository `git clone https://github.com/Computational-Morphogenomics-Group/MarkerMap.git`
- Install the dependencies from pip: `pip install pytorch-lightning scGeneFit scanpy anndata lassonet smashpy`
	- Note if your system has both python2 and python3 installed, you will need to use `pip3` to get the python3 versions of the packages
- Install the dependencies with homebrew: `brew install libomp`

#### For Developers

- If you are going to be developing this package, also install the following: `pip install pre-commit pytest`
- In the root directory, run `pre-commit install`. You should see a line like `pre-commit installed at .git/hooks/pre-commit`. Now when you commit to your local branch, it will run `jupyter nbconvert --clean-output` on all the local jupyter notebooks on that branch. This ensures that only clean notebooks are uploaded to the github.
- To run tests, simply run pytest: `pytest`.
