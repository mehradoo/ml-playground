#!/usr/bin/python
import sys
import numpy

sys.path.append("../ML2-SVM/")
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

print 'features_train size: ' + str(len(features_train))
print 'labels_train size: ' + str(len(labels_train))

print 'features_test size: ' + str(len(features_test))
print 'labels_test size: ' + str(len(labels_test))

features_train = features_train[:len(features_train) / 100]
labels_train = labels_train[:len(labels_train) / 100]

aws_training_matrix = []

for index in range(0, len(features_train)):
    aws_training_matrix.append(numpy.append(features_train[index][0:999], labels_train[index]))

numpy.savetxt("data-source.csv", aws_training_matrix, delimiter=",")

