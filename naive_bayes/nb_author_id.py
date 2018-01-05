#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

t0 = time()
model.fit(features_train, labels_train)
print "predict time:", round(time()-t0, 3), "s"

print(model)


expected = labels_test
t0 = time()
predicted = model.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))



#########################################################


