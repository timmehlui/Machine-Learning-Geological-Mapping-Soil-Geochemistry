# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 10:06:57 2020

@author: Timothy Lui

Create a function that checks that the order of the classes in all 9 algorithms is the same.
Hence we will know that we can add their probabilities.
"""
import numpy as np

def sameClassOrderNine(lr, qda, nn, rbfsvm, nb, ann, rf, ab, gb):
    # lr, qda, nn, lsvm, rbfsvm, nb, ann, rf, ab, gb are all algorithms.
    truth_matrix = [False, False, False, False, False, False, False, False]
    if np.all(lr.classes_ == qda.classes_):
        truth_matrix[0] = True
    if np.all(lr.classes_ == nn.classes_):
        truth_matrix[1] = True
    if np.all(lr.classes_ == rbfsvm.classes_):
        truth_matrix[2] = True
    if np.all(lr.classes_ == nb.classes_):
        truth_matrix[3] = True
    if np.all(lr.classes_ == ann.classes_):
        truth_matrix[4] = True
    if np.all(lr.classes_ == rf.classes_):
        truth_matrix[5] = True
    if np.all(lr.classes_ == ab.classes_):
        truth_matrix[6] = True
    if np.all(lr.classes_ == gb.classes_):
        truth_matrix[7] = True
    if all(truth_matrix):
        return True
    else:
        return False
        