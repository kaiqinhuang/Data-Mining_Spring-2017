# -*- coding: utf-8 -*-
"""
Data Mining_Assignment 5

Name: Kaiqin Huang
Date Due: Mar 14, 2017

Use Python to build a Naive Bayes classifier for the spam data provided here,
and report the classification accuracy for 10 fold cross validation.
You may not use any machine learning libraries for this task.

Submit your well-commented code by 3/14 at start of class.

"""


import pandas as pd
import math
from random import shuffle


def readData(file_name, seperation):
    df = pd.read_csv(file_name, header = None, sep = seperation)
    return df


def probCol(data, col):
    prob = sum(data[col]) / len(data[col])
    return prob


def probWordGivenSpam(data, target, col):
    counter = 0
    for i in range(len(data[col])):
        if target[0][i] == 1 and data[col][i] == 1:
            counter += 1
    prob_word_spam = counter / sum(target[0])
    return prob_word_spam


def probWordGivenNotSpam(data, target, col):
    counter = 0
    for i in range(len(data[col])):
        if target[0][i] == 0 and data[col][i] == 1:
            counter += 1
    prob_word_not_spam = counter / (len(target[0]) - sum(target[0]))
    return prob_word_not_spam


def probNotWordGivenSpam(data, target, col):
    counter = 0
    for i in range(len(data[col])):
        if target[0][i] == 1 and data[col][i] == 0:
            counter += 1
    prob_not_word_spam = counter / sum(target[0])
    return prob_not_word_spam


def probNotWordGivenNotSpam(data, target, col):
    counter = 0
    for i in range(len(data[col])):
        if target[0][i] == 0 and data[col][i] == 0:
            counter += 1
    prob_not_word_not_spam = counter / (len(target[0]) - sum(target[0]))
    return prob_not_word_not_spam



"""
def getProbList(data, target, prob_function):
    prob_list = []
    for i in data.columns:
        prob = prob_function(data, target, i)
        prob_list.append(prob)
    return prob_list
"""




def getProbWordGivenSpamList(data, target):
    prob_list = []
    for i in data.columns:
        prob = probWordGivenSpam(data, target, i)
        prob_list.append(prob)
    return prob_list


def getProbWordGivenNotSpamList(data, target):
    prob_list = []
    for i in data.columns:
        prob = probWordGivenNotSpam(data, target, i)
        prob_list.append(prob)
    return prob_list


def getProbNotWordGivenSpamList(data, target):
    prob_list = []
    for i in data.columns:
        prob = probNotWordGivenSpam(data, target, i)
        prob_list.append(prob)
    return prob_list


def getProbNotWordGivenNotSpamList(data, target):
    prob_list = []
    for i in data.columns:
        prob = probNotWordGivenNotSpam(data, target, i)
        prob_list.append(prob)
    return prob_list






def probSpamGivenDataNumerator(data, target, row):
    binary = data.ix[row]
    prob_word_spam = getProbWordGivenSpamList(data, target)
    prob_not_word_spam = getProbNotWordGivenSpamList(data, target)

    if len(binary) != len(prob_word_spam):
        print("The lengths of binary list and prob list do not equal.")
    else:
        prob_data_given_spam_log = 0
        for i in range(len(binary)):
            prob_data_given_spam_log += math.log(math.pow(prob_word_spam[i], binary[i]), 2) + math.log(math.pow(prob_not_word_spam[i], (1 - binary[i])), 2)
        prob_data_given_spam = math.pow(2, prob_data_given_spam_log)

        prob_spam = probCol(target, 0)
        prob_spam_given_data_numerator = prob_data_given_spam * prob_spam
        return prob_spam_given_data_numerator


def probNotSpamGivenDataNumerator(data, target, row):
    binary = data.ix[row]
    prob_word_not_spam = getProbWordGivenNotSpamList(data, target)
    prob_not_word_not_spam = getProbNotWordGivenNotSpamList(data, target)

    if len(binary) != len(prob_word_not_spam):
        print("The lengths of binary list and prob list do not equal.")
    else:
        prob_data_given_not_spam_log = 0
        for i in range(len(binary)):
            prob_data_given_not_spam_log += math.log(math.pow(prob_word_not_spam[i], binary[i]), 2) + math.log(math.pow(prob_not_word_not_spam[i], (1 - binary[i])), 2)
        prob_data_given_not_spam = math.pow(2, prob_data_given_not_spam_log)

        prob_not_spam = 1 - probCol(target, 0)
        prob_not_spam_given_data_numerator = prob_data_given_not_spam * prob_not_spam
        return prob_not_spam_given_data_numerator


def spamPredict(data, target, row):
    prob_spam_numer = probSpamGivenDataNumerator(data, target, row)
    prob_not_spam_numer = probNotSpamGivenDataNumerator(data, target, row)
    if prob_spam_numer > prob_not_spam_numer:
        return 1
    else:
        return 0


def probSpamGivenData(data, target, row):
    binary = data.ix[row]
    numerator = probSpamGivenDataNumerator(data, target, row)
    denominator_log = 0
    for i in range(len(binary)):
        denominator_log += math.log(math.pow(probCol(data, i), binary[i]), 2) + math.log(math.pow((1 - probCol(data, i)), (1 - binary[i])), 2)        
    denominator = math.pow(2, denominator_log)
    prob_spam_given_data = numerator / denominator
    return prob_spam_given_data


def probNotSpamGivenData(data, target, row):
    binary = data.ix[row]
    numerator = probNotSpamGivenDataNumerator(data, target, row)
    denominator_log = 0
    for i in range(len(binary)):
        denominator_log += math.log(math.pow(probCol(data, i), binary[i]), 2) + math.log(math.pow((1 - probCol(data, i)), (1 - binary[i])), 2)        
    denominator = math.pow(2, denominator_log)
    prob_not_spam_given_data = numerator / denominator
    return prob_not_spam_given_data









def getDataSplitter(data, num_fold):
    total_row = len(data[0])
    size_fold = math.floor(total_row / num_fold)

    random_num_list = [i for i in range(total_row)]
    shuffle(random_num_list)

    splitter = []
    for i in range(num_fold):
        fold = [random_num_list[i * size_fold : ((i + 1) * size_fold - 1)]]
        splitter.append(fold)
    return splitter


#==============================================================================
# def splitData(data, num_fold):
#     splitter = getDataSplitter(data, num_fold)
#
#
# def getTestFold(data, num_fold, num_test_fold):
#==============================================================================








def main():
    df = readData("spamdata_binary.txt", "\t")
    target = readData("spamlabels.txt", "\t")

    """ Test
    """
    value = spamPredict(df, target, 4600)
    print(value)
    value1 = spamPredict(df, target, 0)
    print(value1)
    print("These are the prediction results for the last and first rows.  \n")

    value = probSpamGivenData(df, target, 0) + probNotSpamGivenData(df, target, 0)    
    print(value)
    print("This is the sum of probSpam and probNotSpam.  \n")
  
    print(probSpamGivenData(df, target, 0))
    print(probNotSpamGivenData(df, target, 0))    
    print("These are probSpam and probNotSpam.  \n")

    print(probSpamGivenDataNumerator(df, target, 0))
    print(probNotSpamGivenDataNumerator(df, target, 0))
    print("These are probSpamNumerator and probNotSpamNumerator.  \n")
    
    
    
    
    
    
if __name__ == "__main__":
    main()











