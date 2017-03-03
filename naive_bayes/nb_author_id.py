#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
import numpy
import sklearn
import nltk    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB 
from numpy import array
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train=array(features_train)
features_test=array(features_test)
labels_train=array(labels_train)
labels_test=array(labels_test)


'''
print type(features_train)

print features_train.shape
'''
t0=time()

#########################################################
### your code goes here ###
clf=GaussianNB()
clf.fit(features_train,labels_train)
ans=clf.predict(features_test)
print (ans)
print (accuracy_score(labels_test, ans))


#########################################################

print round(time()-t0,3)

