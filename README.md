# Exact Shapley Values for Local and Model-True Explanations of Decision Tree Ensembles

This repository is the official implementation of [Exact Shapley Values for Local and Model-True Explanations of Decision Tree Ensembles](https://github.com/Biodesix/EjectShapley) (link will be updated when paper is on posted). 

## Requirements

basic libraries (numpy, matplotlib, etc), seaborn, shap

Download the data and results files from [here](https://github.com/Biodesix/EjectShapley) (link to be updated), and put everything in the directory named 'data' for the data files and 'results' for the results files in the top directory.


## Training, Evaluation, and Results

In the top directory, there are scripts to train, validate, and calculate Shapley values on all data sets considered in the manuscript.  Scripts that have 'draw' prepended to their name produce plots, some of which are those from the manuscript.  These scripts are labeled by which figure in the manuscript they correspond to.  For figures 2, and 5, the models are very lightweight and are trained in the drawing scripts.  For figures 3, 4, and E1, there are separate scripts for training/validating/SV calculation named fig3.py, etc.  These models are less lightweight (or at least the SV calculation is).  These will need to be run before the corresponding draw_fixX.py scripts, but please note that by default, they will only produce abbreviated results.  The full result files can be downloaded from [here](https://github.com/Biodesix/EjectShapley) (link to be updated), so the draw scripts can be run on those (and will by default) to produce the plots from the manuscript.  Figure E2 will also produce abbreviated results by default, but this can be overwritten in the script. 

The classes that do the heavy lifting live in the 'crast' (Classification, Regression, and Survival Tree) directory.

## Contributing

Please see License.txt for more information, and feel free to contact the authors with any questions!