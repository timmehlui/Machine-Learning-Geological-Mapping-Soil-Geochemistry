# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:23:46 2020

@author: Timothy Lui

Create a function that does a multi classifier system with any number of models. 
Takes in the probabilities of models and a numpy array in order of classes
Outputs the predictions.

Adds probabilities and then chooses the highest probability.
"""
import numpy as np

def mcsprob(classOrder, prob1, *probs):
    sumprobs = prob1
    for element in probs:
        sumprobs = sumprobs + element
    indices = np.argmax(sumprobs, axis=1)
    num_rows = np.size(prob1, axis = 0)
    predictions = []
    for i in range(num_rows):
        predictions.append(classOrder[indices[i]])
    return np.array(predictions)