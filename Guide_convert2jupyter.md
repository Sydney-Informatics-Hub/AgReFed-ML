# Guide how to convert python scripts to jupyter notebooks

1) Make copy of original python file to folder example_notebooks

2) Remove in copied file all functions except main() and instead import them from python mother module, e.g.:

sys.path.append("../python_scripts/")
from featureimportance import (create_simulated_features, 
	plot_feature_correlation_spearman, 
	calc_new_correlation, 
	blr_factor_importance, 
	plot_correlationbar,
	rf_factor_importance,
	gradientbars)

3) unindent all code below if __name__ == '__main__': and remove this line.

4) Set settings yaml file name directly and remove args loading part

5) move code in main() function to end (after loading settings file)

6) convert the shorten python script to a jupyter notebook via the tool `p2j` 

7) open new generated .ipynb file and make following changes:
    - replace show = False with show = True
    - add header as markdown
    - Add in header note to mother module, e.g.: "This Jupyter notebook is build upon the python module `featureimportance.py` and imports its core functions. To edit any of this functions, please do so in `featureimportance.py`"
