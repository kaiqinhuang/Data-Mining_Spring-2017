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


def probWordGivenSpam(data, target, col, splitter_training):
    counter = 0
    for i in splitter_training:
        if target[0][i] == 1 and data[col][i] == 1:
            counter += 1
    prob_word_spam = counter / sum(target[0])
    return prob_word_spam


def probWordGivenNotSpam(data, target, col, splitter_training):
    counter = 0
    for i in splitter_training:
        if target[0][i] == 0 and data[col][i] == 1:
            counter += 1
    prob_word_not_spam = counter / (len(target[0]) - sum(target[0]))
    return prob_word_not_spam


def probNotWordGivenSpam(data, target, col, splitter_training):
    counter = 0
    for i in splitter_training:
        if target[0][i] == 1 and data[col][i] == 0:
            counter += 1
    prob_not_word_spam = counter / sum(target[0])
    return prob_not_word_spam


def probNotWordGivenNotSpam(data, target, col, splitter_training):
    counter = 0
    for i in splitter_training:
        if target[0][i] == 0 and data[col][i] == 0:
            counter += 1
    prob_not_word_not_spam = counter / (len(target[0]) - sum(target[0]))
    return prob_not_word_not_spam






def getProbList(data, target, prob_function, splitter_training):
    prob_list = []
    for i in data.columns:
        prob = prob_function(data, target, i, splitter_training)
        prob_list.append(prob)
    return prob_list



def getProbWordGivenSpamList(data, target, splitter_training):
    prob_list = []
    for i in data.columns:
        prob = probWordGivenSpam(data, target, i, splitter_training)
        prob_list.append(prob)
    return prob_list


def getProbWordGivenNotSpamList(data, target, splitter_training):
    prob_list = []
    for i in data.columns:
        prob = probWordGivenNotSpam(data, target, i, splitter_training)
        prob_list.append(prob)
    return prob_list


def getProbNotWordGivenSpamList(data, target, splitter_training):
    prob_list = []
    for i in data.columns:
        prob = probNotWordGivenSpam(data, target, i, splitter_training)
        prob_list.append(prob)
    return prob_list


def getProbNotWordGivenNotSpamList(data, target, splitter_training):
    prob_list = []
    for i in data.columns:
        prob = probNotWordGivenNotSpam(data, target, i, splitter_training)
        prob_list.append(prob)
    return prob_list









def probSpamGivenDataNumerator(prob_word_spam_list, prob_not_word_spam_list, prob_spam, data_test, row):
    binary = data_test.ix[row]

    if len(binary) != len(prob_word_spam_list):
        print("The lengths of binary list and prob list do not equal.")
    else:
        prob_data_given_spam_log = 0
        for i in range(len(binary)):
            prob_data_given_spam_log += math.log(math.pow(prob_word_spam_list[i], binary[i]), 2) + math.log(math.pow(prob_not_word_spam_list[i], (1 - binary[i])), 2)
        prob_data_given_spam = math.pow(2, prob_data_given_spam_log)
        prob_spam_given_data_numerator = prob_data_given_spam * prob_spam
        return prob_spam_given_data_numerator


def probNotSpamGivenDataNumerator(prob_word_not_spam_list, prob_not_word_not_spam_list, prob_not_spam, data_test, row):
    binary = data_test.ix[row]

    if len(binary) != len(prob_word_not_spam_list):
        print("The lengths of binary list and prob list do not equal.")
    else:
        prob_data_given_not_spam_log = 0
        for i in range(len(binary)):
            prob_data_given_not_spam_log += math.log(math.pow(prob_word_not_spam_list[i], binary[i]), 2) + math.log(math.pow(prob_not_word_not_spam_list[i], (1 - binary[i])), 2)
        prob_data_given_not_spam = math.pow(2, prob_data_given_not_spam_log)
        prob_not_spam_given_data_numerator = prob_data_given_not_spam * prob_not_spam
        return prob_not_spam_given_data_numerator


"""
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
"""


def spamPredict(prob_word_spam_list, prob_not_word_spam_list, prob_spam, prob_word_not_spam_list, prob_not_word_not_spam_list, prob_not_spam, data_test, row):
    prob_spam_numer = probSpamGivenDataNumerator(prob_word_spam_list, prob_not_word_spam_list, prob_spam, data_test, row)
    prob_not_spam_numer = probNotSpamGivenDataNumerator(prob_word_not_spam_list, prob_not_word_not_spam_list, prob_not_spam, data_test, row)
    if prob_spam_numer > prob_not_spam_numer:
        return 1
    else:
        return 0


def getPredictTarget(prob_word_spam_list, prob_not_word_spam_list, prob_spam, prob_word_not_spam_list, prob_not_word_not_spam_list, prob_not_spam, data_test, splitter_test):
    predict_target = []
    for i in splitter_test:
        spam_predict = spamPredict(prob_word_spam_list, prob_not_word_spam_list, prob_spam, prob_word_not_spam_list, prob_not_word_not_spam_list, prob_not_spam, data_test, i)
        predict_target.append(spam_predict)
    return predict_target
    

def getAccuracy(target_test, target_predict, splitter_test):
    if len(target_test) != len(target_predict):
        print("The lengths of target_test and target_predict do not equal.")
    else:
        counter = 0
        for i in range(len(splitter_test)):
            if target_test.iloc[i, 0] == target_predict[i]:
                counter += 1
        accuracy = counter / len(target_test)
        return accuracy


"""
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
"""








def getDataSplitter(data, num_fold):
    total_row = len(data[0])
    size_fold = math.floor(total_row / num_fold)

    random_num_list = [i for i in range(total_row)]
    shuffle(random_num_list)

    splitter = []
    for i in range(num_fold):
        fold = random_num_list[i * size_fold : (i + 1) * size_fold]
        splitter.append(fold)
    return splitter


def getTestSplitter(splitter, num_test_fold):
    splitter_test = splitter[num_test_fold]
    return splitter_test
    
    
def getTrainingSplitter(splitter, num_test_fold):
    splitter_training = []
    for i in range(len(splitter)):
        if i != num_test_fold:
            splitter_training += splitter[i]
    return splitter_training

    
def getTest(data, splitter_test):
    data_test = data[data.index.isin(splitter_test)]
    return data_test


def getTraining(data, splitter_training):
    data_training = data[data.index.isin(splitter_training)]
    return data_training








def main():
    df = readData("spamdata_binary.txt", "\t")
    target = readData("spamlabels.txt", "\t")

    """ Test
    
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
    """
   
    num_fold = 10
    num_test_fold = 1
    splitter = getDataSplitter(target, num_fold)
    splitter_test = getTestSplitter(splitter, num_test_fold)
    splitter_training = getTrainingSplitter(splitter, num_test_fold)
        
    df_test = getTest(df, splitter_test)
    target_test = getTest(target, splitter_test)
    df_training = getTraining(df, splitter_training)
    target_training = getTraining(target, splitter_training)
    print(target_test)

    prob_word_spam_list = getProbWordGivenSpamList(df_training, target_training, splitter_training)    
    prob_not_word_spam_list = getProbList(df_training, target_training, probNotWordGivenSpam, splitter_training)
    prob_word_not_spam_list = getProbList(df_training, target_training, probWordGivenNotSpam, splitter_training)
    prob_not_word_not_spam_list = getProbList(df_training, target_training, probNotWordGivenNotSpam, splitter_training)
    prob_spam = probCol(target_training, 0)
    prob_not_spam = 1 - probCol(target_training, 0)
       
    target_predict = getPredictTarget(prob_word_spam_list, prob_not_word_spam_list, prob_spam, prob_word_not_spam_list, prob_not_word_not_spam_list, prob_not_spam, df_test, splitter_test)    
    accuracy = getAccuracy(target_test, target_predict, splitter_test)
    print(accuracy)


if __name__ == "__main__":
    main()

