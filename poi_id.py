#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
from pprint import pprint
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from pprint import pprint

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi","total_payments", "total_stock_value",
                 "from_poi_to_this_person","from_this_person_to_poi"
    , "to_messages","shared_receipt_with_poi","from_messages"]

# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict = pd.DataFrame(data_dict)
data_dict =  data_dict.T
data_dict_poi = data_dict[["poi"]]
data_dict = data_dict[features_list]
### Task 2: Remove outliers
data_dict = data_dict.drop(['TOTAL'],axis = 'index')
data_dict = data_dict.drop(data_dict.loc[ ((data_dict['poi']== True)
                                           & (data_dict['from_poi_to_this_person'] == 'NaN'))].index
                           , axis = 0)
data_dict = data_dict.replace(to_replace = "NaN", value = "0")

### Task 3: Create new feature(s)
data_dict = data_dict.applymap(float)
data_dict["from_this_person_to_poi_ratio"]=data_dict["from_this_person_to_poi"]/data_dict["from_messages"]
data_dict["from_poi_to_this_person_ratio"]=data_dict["from_poi_to_this_person"]/data_dict["to_messages"]

data_dict = data_dict.replace(to_replace = "NaN", value = 0)
data_dict_norm = (data_dict - data_dict.mean()) / (data_dict.max() - data_dict.min())

###reverse normalized poi
data_dict_norm["poi"] = data_dict_poi

data_dict =  data_dict_norm
data_dict = data_dict.drop(['from_this_person_to_poi', u'to_messages', u'from_messages'
                               ,'from_poi_to_this_person'], axis = 1)

features_list = data_dict.columns

### Store to my_dataset for easy export below.
###convert DataFrame to dict
new_data_dict = {}
for i_row in data_dict.index:
    new_data_dict[i_row] =  {}
    for i_column  in data_dict.columns:
        new_data_dict[i_row][i_column] = data_dict.loc[i_row,i_column]

my_dataset = new_data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion ='entropy',max_features = 'auto',class_weight='balanced' )

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.1, random_state=42)

from sklearn import metrics
print "start of decision tree"
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(features_train, labels_train)
pred_tree = clf_tree.predict(features_test)
print metrics.precision_score(labels_test,pred_tree )
print metrics.recall_score(labels_test, pred_tree)
print "end of decision tree"

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
