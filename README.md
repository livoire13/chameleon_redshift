[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14846920.svg)](https://doi.org/10.5281/zenodo.14846920)

# Chameleon Redshift
Python codes accompanying the article "Testing screened scalar-tensor theories of gravity with atomic clocks" by Hugo Lévy (author of the code) and Jean-Philippe Uzan. To run the scripts, the *femtoscope* package must be installed.
- The paper: https://arxiv.org/abs/2410.17292
- *femtoscope*: https://github.com/onera/femtoscope/

## optimal_constraints.py
This script was used to produce Figs. 1 and 9, and to check the condition given by Eq. (35).

## foil_experiment.py
This script was used to produce Fig. 3, showing potential constraints on the chameleon parameter space resulting from the experimental design involving a foil surrounding some of the atoms. Details about the assumptions used for this numerical study can be found in Sec. V A 3 of the article.

## nucleus.py
This script was used to produce Fig. 4, where we study the influence of the atom's nucleus on the chameleon field value at the location of the electron cloud (i.e. a few Bohr radii away from the nucleus).
