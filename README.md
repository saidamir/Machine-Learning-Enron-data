# Machine-Learning-Enron-data
Please refer to Summary_project.txt for the detailed Q&A

Building an algorithm for predicting point of interest persons based on emails and compensation

The goal of the project is to develop an algorithm that will identify Enron employee who lkeily committed fraud. The dataset cosnsits of Enron email and financial data compiled into a dictionary.  Each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features in the data fall into three major types, namely financial features, email features and POI labels. 

I selected GradientBoosting algorithm which helps combining the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator. The module provides methods for both classification and regression via gradient boosted regression trees. Maximum scores are under 'max_depth' parameter value of 10 in our list and they stay the same once they increase.

Project 5.ipynb is the IPython document with detailed codes and explaination of my thought process

Prorject 5_polished.ipynb is the IPython document where the code can be hidden or open - use the button on the top left corner

poi_id.py is the final code file

pkl files are files created as a result of dumper in poi_id.py

Summary_project.txt is the file which answers the questions raised in the project


