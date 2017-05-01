#!/usr/bin/python

import sys
sys.path.append("../tools/")
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

print 'training size: ' + str(features_train.size)
print 'feature size: ' + str(features_train[0].size)

import numpy
a = numpy.asarray(features_train)
numpy.savetxt("features_train.csv", a, delimiter=",")

b = numpy.asarray(labels_train)
numpy.savetxt("labels_train.csv", b, delimiter=",")

