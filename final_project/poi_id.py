#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
from time import time
import numpy as np
from numpy import mean
sys.path.append("../tools/")

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi"."others" and "email_address" left out.

features_list = ['poi', 
                 'salary', 
                 'deferral_payments', 
                 'total_payments', 
                 'loan_advances',
                 'bonus', 
                 'restricted_stock_deferred', 
                 'deferred_income', 
                 'total_stock_value',
                 'expenses', 
                 'exercised_stock_options', 
                 'long_term_incentive',
                 'restricted_stock', 
                 'director_fees', 
                 'to_messages', 
                 'from_poi_to_this_person',
                 'from_messages', 
                 'from_this_person_to_poi', 
                 'shared_receipt_with_poi'] # You will need to use more features

print 'feature count: ', len(features_list)-1

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Explore the dataset
print 'people count in the dataset:', len(data_dict)

### Number of poi in the dataset
num_poi = 0
for i in data_dict:
    if data_dict[i]['poi']==True:
        num_poi=num_poi+1
print 'poi count in the dataset:', num_poi
print 'person count who are not poi:', len(data_dict)-num_poi

### Task 2: Remove outliers
features =['salary', 'bonus']
data = featureFormat(data_dict, features)
for point in data:
    salary=point[0]
    bonus=point[1]
    plt.scatter(salary, bonus)
plt.xlabel('salary')
plt.ylabel('bonus')

plt.show() # There is an outlier

#for i, v in data_dict.items():
#    if v['salary'] != 'NaN' and v['salary'] > 10000000:
#        print i

data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0 )

print "datapoints after removing outliers:", len(data_dict)

features =['salary', 'bonus']
data = featureFormat(data_dict, features)
for point in data:
    salary=point[0]
    bonus=point[1]
    plt.scatter(salary, bonus)
plt.xlabel('salary')
plt.ylabel('bonus')

plt.show()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# Bonus-salary ratio
for employee, features in data_dict.iteritems():
	if features['bonus'] == "NaN" or features['salary'] == "NaN":
		features['bonus_salary_ratio'] = "NaN"
	else:
		features['bonus_salary_ratio'] = float(features['bonus']) / float(features['salary'])

# from_this_person_to_poi as a percentage of from_messages
for employee, features in data_dict.iteritems():
	if features['from_this_person_to_poi'] == "NaN" or features['from_messages'] == "NaN":
		features['from_this_person_to_poi_percentage'] = "NaN"
	else:
		features['from_this_person_to_poi_percentage'] = float(features['from_this_person_to_poi']) / float(features['from_messages'])

# from_poi_to_this_person as a percentage of to_messages
for employee, features in data_dict.iteritems():
	if features['from_poi_to_this_person'] == "NaN" or features['to_messages'] == "NaN":
		features['from_poi_to_this_person_percentage'] = "NaN"
	else:
		features['from_poi_to_this_person_percentage'] = float(features['from_poi_to_this_person']) / float(features['to_messages'])

                    
features_list.extend(['bonus_salary_ratio', 'from_this_person_to_poi_percentage','from_poi_to_this_person_percentage'])

print "\nThree new features created: \n'bonus_salary_ratio'\n'from_this_person_to_poi_percentage'\n'from_poi_to_this_person_percentage'"

print "\nNaNs per feature: "

### Missing values in each feature
nan = [0 for i in range(len(features_list))]
for i, person in my_dataset.iteritems():
    for j, feature in enumerate(features_list):
        if person[feature] == 'NaN':
            nan[j] += 1
for i, feature in enumerate(features_list):
    print feature, nan[i]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Improved featuer selection
selection=SelectKBest(k=7)
selection.fit(features, labels)

results = zip(selection.get_support(), features_list[1:], selection.scores_)
results = sorted(results, key=lambda x: x[2], reverse=True)
print "\n K-best features:"
for result in results:
    print result

## update features list chosen manually and by SelectKBest
features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus',
                 'salary', 'from_this_person_to_poi_percentage', 'deferred_income']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Set up pipelines for classifiers
scaler = MinMaxScaler()
select = SelectKBest()

dct = DecisionTreeClassifier()
svc = SVC()
knc = KNeighborsClassifier()
rnf = RandomForestClassifier()

# Load pipeline steps into list
steps_dct = [
         ('feature_selection', select),
         ('classifier', dct)
         ]
steps_svc = [
         ('min_max_scaler', scaler),
         ('feature_selection', select),
         ('classifier', svc)
         ]
steps_knc = [
         ('min_max_scaler', scaler),
         ('feature_selection', select),
         ('classifier', knc)
         ]

steps_rnf = [
         ('feature_selection', select),
         ('classifier', rnf)
         ]


# Create pipeline
pipeline_dct = Pipeline(steps_dct)
pipeline_svc = Pipeline(steps_svc)
pipeline_knc = Pipeline(steps_knc)
pipeline_rnf = Pipeline(steps_rnf)

# Set pipleline parameters to be used in gridsearch
parameters_dct = dict(
                  feature_selection__k=[2, 3, 5, 6], 
                  classifier__criterion=['gini', 'entropy'],
                  classifier__max_depth=[0,1,2,3, 4, 5, 8],
                  classifier__min_samples_split=[1, 2, 3, 4],
                  classifier__class_weight=[None, 'balanced'],
                  classifier__random_state=[42]
                  )

parameters_svc = dict(
                  feature_selection__k=[2, 3, 5, 6], 
                  classifier__C=[0.1, 1, 10, 100, 1000],
                  classifier__kernel=['rbf'],
                  classifier__gamma=[0.001, 0.0001]
                  )

parameters_knc = dict(
                  feature_selection__k=[2, 3, 5, 6], 
                  classifier__n_neighbors=[1, 2, 3, 4, 5],
                  classifier__leaf_size=[1, 10, 30, 60],
                  classifier__algorithm=['auto', 'ball_tree', 'kd_tree', 'brute']
                  )

parameters_rnf = dict(
                  feature_selection__k=[2, 3, 5, 6], 
                  classifier__n_estimators=[5,10],
                  classifier__criterion=['gini', 'entropy'],
                  classifier__max_depth=[0,1,2,3, 4, 5, 8],
                  classifier__min_samples_split=[1, 2, 3, 4],
                  classifier__class_weight=[None, 'balanced'],
                  classifier__random_state=[42]
                  )


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



# Create training sets and test sets
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Cross-validation for parameter tuning in grid search 
strat_split = StratifiedShuffleSplit(
    labels_train,
    n_iter = 20,
    test_size = 0.5,
    random_state = 42
    )

# Create, fit, and make predictions with grid search

#select_pipeline
pipeline = pipeline_dct
parameters = parameters_dct

gs = GridSearchCV(pipeline,
	              param_grid=parameters,
	              scoring="f1",
	              cv=strat_split,
	              error_score=0)
gs.fit(features_train, labels_train)
labels_predictions = gs.predict(features_test)

# Pick the classifier with the best tuned parameters
clf = gs.best_estimator_
print "\n", "Best parameters are: ", gs.best_params_, "\n"

if pipeline == pipeline_dct:
# Print features selected and their importances
    features_selected=[features_list[i+1] for i in clf.named_steps['feature_selection'].get_support(indices=True)]
    scores = clf.named_steps['feature_selection'].scores_
    importances = clf.named_steps['classifier'].feature_importances_
    
    indices = np.argsort(importances)[::-1]
    print 'The ', len(features_selected), " features selected and their importances:"
    for i in range(len(features_selected)):
        print "feature no. {}: {} ({}) ({})".format(i+1,features_selected[indices[i]],importances[indices[i]], scores[indices[i]])

# Print classification report (focus on precision and recall)
report = classification_report( labels_test, labels_predictions )
print(report)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)