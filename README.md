Code for "Non-equilibrium whole-brain dynamics arise from pairwise interactions"
Sebastian M. Geli, Christopher W. Lynn Morten L. Kringelbach Gustavo Deco and 
Yonatan Sanz Perl.

The codes are organized in the following way:

 - "transition_matrix_random_areas.py" Computes the joint transition matrices to decompose the entropy production. The decomposition was done using the codes provided in Lynn, C. W., Holmes, C. M., Bialek, W. & Schwab, D. J. Decomposing the Local Arrow of Time in Interacting Systems. Phys Rev Lett 129, (2022).

- "autoencoder.py" Trains an autoencoder neural network on time series data to  obtain a latent space representation.

- "s2_matrices_generate_and_classify" Computes the entropy production of pairs of regions and trains a random forest classifier to identify cognitive tasks.

The neuroimaging data used in this paper are publicly available from HCP. In addition, we provide a function within each code to generate data of equivalent shape that can be used for testing.

For queries or issues, please contact Sebastian Geli at sebastianmanuel.geli@upf.edu

Cite the code [![DOI](https://zenodo.org/badge/861690469.svg)](https://zenodo.org/doi/10.5281/zenodo.13838040)
