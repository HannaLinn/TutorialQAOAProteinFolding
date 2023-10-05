# TutorialQAOAProteinFolding

A tutorial in protein folding of coordinate-based HP-lattice model using Pennylane.

## Setup
1. Copy the conda environment (this may take a while):
conda env create -f environment.yml
2. Then activate the environment:
conda activate tutorialproteinenv
3. Activate Jypyter Notebook:
jupyter notebook
4. Open the file: Tutotial_HPCoordinateBasedLattice.ipynb

## Contains
To be able to run the jupyter notebook, you need the following files in the same directory: (you don't have to touch these, but have a look if you are interested)
1. protein_folding.py: Code for generating the Hamiltonian.
2. protein_utils.py: Code for nice plots, getting information about your Hamiltonian and generating fancy parameters.
3. CoordinateBased_HPLattice.py: Code for generating a protein instance with the coordinate-based HP-lattice model.
