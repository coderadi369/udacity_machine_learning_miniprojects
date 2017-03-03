#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from numpy import array
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train=array(features_train)
labels_train=array(labels_train)
features_test=array(features_test)
labels_test=array(labels_test)

print features_train


#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.svm import SVC
clf=SVC(kernel="linear")
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print pred
print (accuracy_score(labels_test, pred))

#########################################################


