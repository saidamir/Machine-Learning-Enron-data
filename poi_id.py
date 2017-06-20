import sys
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import math
from tester import test_classifier, dump_classifier_and_data
import numpy as np
from numpy import log
from numpy import sqrt
from numpy import float64
from numpy import nan
from time import time

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import matplotlib
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from tester import dump_classifier_and_data
from sklearn import preprocessing
from tester import main

sys.path.append("../tools/")

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'email_address', 'from_poi_to_this_person'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['LOCKHART EUGENE E']
### Task 3: Create new feature(s)
data_dict_log = data_dict
#dividing features to financial, non-financial and label
financial_features = ['salary', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 
                      'restricted_stock', 'restricted_stock_deferred', 'total_stock_value', 'expenses',
                      'loan_advances','other', 'director_fees', 'deferred_income', 'long_term_incentive']
non_financials_features = ['to_messages', 'shared_receipt_with_poi', 'from_messages', 
                           'from_this_person_to_poi', 'from_poi_to_this_person']
poi_label = ['poi']
#creating new feature names
features_new = ["poi_ratio_from", 'poi_ratio_to',"total_payments_log", "salary_log","bonus_log",
                "total_stock_value_log", "exercised_stock_options_log"]

NANvalue = 'NaN'

for key in data_dict_log:
#creating new financial features, by log the compensation data          
    for feat in financial_features:
        try:
            data_dict_log[key][feat + '_log'] = math.log(data_dict_log[key][feat])
        except:
            data_dict_log[key][feat + '_log'] = NANvalue   
            
#creating new non_financial features   
    try: 
        data_dict_log[key]['poi_ratio_from'] = \
        1. * data_dict_log[key]['from_this_person_to_poi'] / data_dict_log[key]['from_messages']
        data_dict_log[key]['poi_ratio_to'] = \
        1. * data_dict_log[key]['from_poi_to_this_person'] / data_dict_log[key]['to_messages'] * 1.
    except:
        data_dict_log[key]['poi_ratio_from'] = NANvalue
        data_dict_log[key]['poi_ratio_to'] = NANvalue
            
# combining all to one feature list
features_list = poi_label + financial_features + non_financials_features + features_new
print 'No of all features', len(features_list)

### Store to my_dataset for easy export below.
my_dataset = data_dict_log
print "Length of the dataset", len(data_dict_log)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

GNB = GaussianNB()
DTC = DecisionTreeClassifier()
LR = LogisticRegression()

#this returned k-9 and acceptable results
#param_grid = {'selectkbest__k': range(5,27)}
#below returned k-16 and the best results
param_grid = {'selectkbest__k': range(10,27)} 

# Create the pipline to use in Task 5
pipeline = Pipeline([#('reduce_dim', PCA()),
                     ('selectkbest', SelectKBest()),
                    ('naive_bayes', GNB), #('min_max_scaler', scaler)
                    ])
#('min_max_scaler', scaler)]
grid_search = GridSearchCV(pipeline, cv=10, n_jobs=-1, param_grid = param_grid)
grid_search.fit(features_train, labels_train)
clf = grid_search.best_estimator_
pred = clf.predict(features_test)
#acc=accuracy_score(labels_test, pred)
#precision = precision_score(labels_test,pred)
#recall = recall_score(labels_test,pred)

print clf
print grid_search.best_estimator_.score(features_test, labels_test)
print grid_search.best_params_,
dump_classifier_and_data(clf, my_dataset, features_list)

main()
