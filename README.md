# Exact Shapley Values for Local and Model-True Explanations of Decision Tree Ensembles

This repository is the official implementation of [Exact Shapley Values for Local and Model-True Explanations of Decision Tree Ensembles](https://github.com/Biodesix/EjectShapley) (link will be updated when paper is posted). 

## Requirements

basic libraries (numpy, matplotlib, etc), seaborn, shap

Download the data and results files from [here](https://data.mendeley.com/datasets/f7hnnyrssm/1), and put everything in the directory named 'data' for the data files and 'results' for the results files in the top directory.

The results from the paper were produced using Python 3.8.5, xgboost 1.3.3, shap 0.39.0, and scikit-learn 0.15.2

## Training, Evaluation, and Results

In the top directory, there are scripts to train and validate models and calculate Shapley values on the validation set on all data sets considered in the manuscript.
- *fig3.py*: trains, validates, and calculates SVs necessary for figure 3 results, produces abbreviated results by default
- *fig4.py*: trains, validates, and calculates SVs necessary for figure 4 results, produces abbreviated results by default
- *figE1.py*: trains, validates, and calculates SVs necessary for figure E1 results, produces abbreviated results by default
- *draw_fig2.py*: trains, validates, calculates SVs, and produces plots for figure 2.
- *draw_fig3.py*: produces plots for figure 3 on either downloaded full data set or data resulting data from *fig3.py*.
- *draw_fig4AB.py*: produces plots for figure 4 (A and B) on either downloaded full data set or data resulting data from *fig4.py*.
- *draw_fig4CD.py*: produces plots for figure 4 (C and D) on either downloaded full data set or data resulting data from *fig4.py*.
- *draw_fig5.py*: trains, validates, calculates SVs, and produces plots for figure 5.
- *draw_figE1.py*: produces plots for figure E1 on either downloaded full data set or data resulting data from *figE1.py*.
- *draw_figE2.py*: trains, validates, calculates SVs, and produces plots for figure E2, produces abbreviated results by default.

The classes that do the heavy lifting live in the 'crast' (Classification, Regression, and Survival Tree) directory.

## Contributing

Please see License.txt for more information, and feel free to contact the authors with any questions!