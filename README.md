# Machine-Learning-Enron-data
Please refer to Project_5_polished.ipynb for the detailed report and code

Building an algorithm for predicting point of interest persons based on emails and compensation

The goal of the project is to develop an algorithm that will identify Enron employee who lkeily committed fraud. The dataset cosnsits of Enron email and financial data compiled into a dictionary.  Each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features in the data fall into three major types, namely financial features, email features and POI labels. 

I selected Gaussian NB algorithm because it showed great results on selected features and without even applying PCA. Initially, I thought that the GradientBoosting will work however it did not on tester.py data, probably because of cross validations issues and the fact that there were too many features to be used.

Final top five features selected were as below:

1. salary 
2. deferral_payments
3. total_payments 
4. exercised_stock_options 
5. bonus 

poi_id.py is the final code file

pkl files are files created as a result of dumper in poi_id.py



