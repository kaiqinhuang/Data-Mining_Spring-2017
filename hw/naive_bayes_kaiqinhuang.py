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
    """Read in data
    
    Args:
        file_name: the file name
        seperation: the seperation used in the file
        
    Return:
        df: the data in a DataFrame
    """
    
    df = pd.read_csv(file_name, header = None, sep = seperation)
    return df


def probCol(data, col):
    """Get the probability of a column
    
    Args:
        data: the DataFrame
        col: the index of the column
        
    Return:
        prob: probability of 1 in the column of binary numbers
    """
    
    prob = sum(data[col]) / len(data[col])
    return prob


def probWordGivenSpam(data, target, col, splitter_training):
    """Get the conditional probability of word included given it is a spam
    
    Args:
        data: the DataFrame of data
        target: the DataFrame of target
        col: the column index of the word
        splitter_training: a list of row index numbers to get the training data
        
    Return:
        prob_word_spam: the conditional probability of word included given it is a spam
    """
    
    counter = 0
    for i in splitter_training:
        if target[0][i] == 1 and data[col][i] == 1:
            counter += 1
    prob_word_spam = counter / sum(target[0])
    return prob_word_spam


def probWordGivenNotSpam(data, target, col, splitter_training):
    """Get the conditional probability of word included given it is not a spam
    
    Args:
        data: the DataFrame of data
        target: the DataFrame of target
        col: the column index of the word
        splitter_training: a list of row index numbers to get the training data
        
    Return:
        prob_word_not_spam: the conditional probability of word included given it is not a spam
    """

    counter = 0
    for i in splitter_training:
        if target[0][i] == 0 and data[col][i] == 1:
            counter += 1
    prob_word_not_spam = counter / (len(target[0]) - sum(target[0]))
    return prob_word_not_spam


def probNotWordGivenSpam(data, target, col, splitter_training):
    """Get the conditional probability of word not included given it is a spam
    
    Args:
        data: the DataFrame of data
        target: the DataFrame of target
        col: the column index of the word
        splitter_training: a list of row index numbers to get the training data
        
    Return:
        prob_not_word_spam: the conditional probability of word not included given it is a spam
    """

    counter = 0
    for i in splitter_training:
        if target[0][i] == 1 and data[col][i] == 0:
            counter += 1
    prob_not_word_spam = counter / sum(target[0])
    return prob_not_word_spam


def probNotWordGivenNotSpam(data, target, col, splitter_training):
    """Get the conditional probability of word not included given it is not a spam
    
    Args:
        data: the DataFrame of data
        target: the DataFrame of target
        col: the column index of the word
        splitter_training: a list of row index numbers to get the training data
        
    Return:
        prob_not_word_not_spam: the conditional probability of word not included given it is not a spam
    """

    counter = 0
    for i in splitter_training:
        if target[0][i] == 0 and data[col][i] == 0:
            counter += 1
    prob_not_word_not_spam = counter / (len(target[0]) - sum(target[0]))
    return prob_not_word_not_spam










def getProbList(data, target, prob_function, splitter_training):
    """Get the probability list
    
    Args:
        data: the DataFrame of data
        target: the DataFrame of target
        prob_function: identify which probability it is calculating
        splitter_training: a list of row index numbers to get the training data

    Return:
        prob_list: a list of probabilities            
    """

    prob_list = []
    for i in data.columns:
        prob = prob_function(data, target, i, splitter_training)
        prob_list.append(prob)
    return prob_list


def getProbWordGivenSpamList(data, target, splitter_training):
    """Get the probability list
    
    Args:
        data: the DataFrame of data
        target: the DataFrame of target
        splitter_training: a list of row index numbers to get the training data

    Return:
        prob_list: a list of probabilities            
    """
    
    prob_list = []
    for i in data.columns:
        prob = probWordGivenSpam(data, target, i, splitter_training)
        prob_list.append(prob)
    return prob_list


def getProbWordGivenNotSpamList(data, target, splitter_training):
    """Get the probability list
    
    Args:
        data: the DataFrame of data
        target: the DataFrame of target
        splitter_training: a list of row index numbers to get the training data

    Return:
        prob_list: a list of probabilities            
    """
    
    prob_list = []
    for i in data.columns:
        prob = probWordGivenNotSpam(data, target, i, splitter_training)
        prob_list.append(prob)
    return prob_list


def getProbNotWordGivenSpamList(data, target, splitter_training):
    """Get the probability list
    
    Args:
        data: the DataFrame of data
        target: the DataFrame of target
        splitter_training: a list of row index numbers to get the training data

    Return:
        prob_list: a list of probabilities            
    """
    
    prob_list = []
    for i in data.columns:
        prob = probNotWordGivenSpam(data, target, i, splitter_training)
        prob_list.append(prob)
    return prob_list


def getProbNotWordGivenNotSpamList(data, target, splitter_training):
    """Get the probability list
    
    Args:
        data: the DataFrame of data
        target: the DataFrame of target
        splitter_training: a list of row index numbers to get the training data

    Return:
        prob_list: a list of probabilities            
    """
    
    prob_list = []
    for i in data.columns:
        prob = probNotWordGivenNotSpam(data, target, i, splitter_training)
        prob_list.append(prob)
    return prob_list










def probSpamGivenDataNumerator(prob_word_spam_list, prob_not_word_spam_list, prob_spam, data_test, row):
    """Get the numerator of the conditional probability it is a spam given data
    
    Args:
        prob_word_spam_list: a list of probabilities word is included given it is a spam
        prob_not_word_spam_list: a list of probabilities word is not included given it is a spam
        prob_spam: the probability it is a spam
        data_test: the DataFrame of test data
        row: the row index
            
    Return:
        prob_spam_given_data_numerator: the numerator part of the conditional probability got from Naive Bayes
    """
        
    binary = data_test.ix[row]
    if len(binary) != len(prob_word_spam_list):
        print("The lengths of binary list and prob list do not equal.")
    else:
        prob_data_given_spam_log = 0
        for i in range(len(binary)):
            prob_data_given_spam_log += math.log(math.pow(prob_word_spam_list[i], binary[i]) + 0.000001, 2) + math.log(math.pow(prob_not_word_spam_list[i], (1 - binary[i])) + 0.000001, 2)
        prob_data_given_spam = math.pow(2, prob_data_given_spam_log)
        prob_spam_given_data_numerator = prob_data_given_spam * prob_spam
        return prob_spam_given_data_numerator


def probNotSpamGivenDataNumerator(prob_word_not_spam_list, prob_not_word_not_spam_list, prob_not_spam, data_test, row):
    """Get the numerator of the conditional probability it is a spam given data
    
    Args:
        prob_word_spam_list: a list of probabilities word is included given it is a spam
        prob_not_word_spam_list: a list of probabilities word is not included given it is a spam
        prob_not_spam: the probability it is not a spam
        data_test: the DataFrame of test data
        row: the row index
            
    Return:
        prob_not_spam_given_data_numerator: the numerator part of the conditional probability got from Naive Bayes
    """
        
    binary = data_test.ix[row]
    if len(binary) != len(prob_word_not_spam_list):
        print("The lengths of binary list and prob list do not equal.")
    else:
        prob_data_given_not_spam_log = 0
        for i in range(len(binary)):
            prob_data_given_not_spam_log += math.log(math.pow(prob_word_not_spam_list[i], binary[i]) + 0.000001, 2) + math.log(math.pow(prob_not_word_not_spam_list[i], (1 - binary[i])) + 0.000001, 2)
        prob_data_given_not_spam = math.pow(2, prob_data_given_not_spam_log)
        prob_not_spam_given_data_numerator = prob_data_given_not_spam * prob_not_spam
        return prob_not_spam_given_data_numerator


def spamPredict(prob_word_spam_list, prob_not_word_spam_list, prob_spam, prob_word_not_spam_list, prob_not_word_not_spam_list, prob_not_spam, data_test, row):
    """Get the predicted target
    
    Args:
        prob_word_spam_list: a list of probabilities word is included given it is a spam
        prob_not_word_spam_list: a list of probabilities word is not included given it is a spam
        prob_spam: the probability it is a spam
        prob_word_spam_list: a list of probabilities word is included given it is a spam
        prob_not_word_spam_list: a list of probabilities word is not included given it is a spam
        prob_not_spam: the probability it is not a spam
        data_test: the DataFrame of test data
        row: the row index

    Return:
        1: if it is predicted to be a spam, returns 1
        0: if it is not predicted to be a spam, returns 0
    """        

    prob_spam_numer = probSpamGivenDataNumerator(prob_word_spam_list, prob_not_word_spam_list, prob_spam, data_test, row)
    prob_not_spam_numer = probNotSpamGivenDataNumerator(prob_word_not_spam_list, prob_not_word_not_spam_list, prob_not_spam, data_test, row)
    if prob_spam_numer > prob_not_spam_numer:
        return 1
    else:
        return 0


def getPredictTarget(prob_word_spam_list, prob_not_word_spam_list, prob_spam, prob_word_not_spam_list, prob_not_word_not_spam_list, prob_not_spam, data_test, splitter_test):
    """Get a list of predicted targets
    
    Args:
        prob_word_spam_list: a list of probabilities word is included given it is a spam
        prob_not_word_spam_list: a list of probabilities word is not included given it is a spam
        prob_spam: the probability it is a spam
        prob_word_spam_list: a list of probabilities word is included given it is a spam
        prob_not_word_spam_list: a list of probabilities word is not included given it is a spam
        prob_not_spam: the probability it is not a spam
        data_test: the DataFrame of test data
        splitter_test: a list of row index numbers to get the test data

    Return:
        predict_target: a list of predicted targets
    """

    predict_target = []
    for i in splitter_test:
        spam_predict = spamPredict(prob_word_spam_list, prob_not_word_spam_list, prob_spam, prob_word_not_spam_list, prob_not_word_not_spam_list, prob_not_spam, data_test, i)
        predict_target.append(spam_predict)
    return predict_target
    

def getAccuracy(target_test, target_predict, splitter_test):
    """Get the accuracy of the predictions
    
    Args:
        target_test: the DataFrame of test data targets
        target_predict: the DataFrame of the predicted targets

    Return:
        accuracy: the accuracy of predictions
    """
    
    if len(target_test) != len(target_predict):
        print("The lengths of target_test and target_predict do not equal.")
    else:
        counter = 0
        for i in range(len(splitter_test)):
            if target_test.iloc[i, 0] == target_predict[i]:
                counter += 1
        accuracy = counter / len(target_test)
        return accuracy










def getDataSplitter(data, num_fold):
    """Get the splitter
    
    Args:
        data: DataFrame needs to be splitted
        num_fold: number of folds
        
    Return:
        splitter: a list of lists of row index
    """
    
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
    """Get the splitter to get the test data
    
    Args:
        splitter: a list of lists of row index
        num_test_fold: the index of list chosen for test
        
    Return:
        splitter_test: a list of row index for the test data
    """     
        
    splitter_test = splitter[num_test_fold]
    return splitter_test
    
    
def getTrainingSplitter(splitter, num_test_fold):
    """Get the splitter to get the training data
    
    Args:
        splitter: a list of lists of row index
        num_test_fold: the index of list chosen for test
        
    Return:
        splitter_training: a list of row index for the training data
    """     
        
    splitter_training = []
    for i in range(len(splitter)):
        if i != num_test_fold:
            splitter_training += splitter[i]
    return splitter_training

    
def getTest(data, splitter_test):
    """Get the test DataFrame
    
    Args:
        data: the data to be splitted
        splitter_test: a list of row index
        
    Return:
        data_test: the DataFrame of the test data
    """
    
    data_test = data[data.index.isin(splitter_test)]
    return data_test


def getTraining(data, splitter_training):
    """Get the training DataFrame
    
    Args:
        data: the data to be splitted
        splitter_training: a list of row index
        
    Return:
        data_training: the DataFrame of the training data
    """
    
    data_training = data[data.index.isin(splitter_training)]
    return data_training










def main():
     
    df = readData("spamdata_binary.txt", "\t")
    target = readData("spamlabels.txt", "\t")
    
    num_fold = 10
    splitter = getDataSplitter(target, num_fold)
    accuracy_sum = 0
    
    for i in range(num_fold):
        num_test_fold = i
        splitter_test = getTestSplitter(splitter, num_test_fold)
        splitter_training = getTrainingSplitter(splitter, num_test_fold)
        splitter_test.sort()
        splitter_training.sort()
        
        df_test = getTest(df, splitter_test)
        target_test = getTest(target, splitter_test)
        df_training = getTraining(df, splitter_training)
        target_training = getTraining(target, splitter_training)
    
        prob_word_spam_list = getProbWordGivenSpamList(df_training, target_training, splitter_training)    
        prob_not_word_spam_list = getProbList(df_training, target_training, probNotWordGivenSpam, splitter_training)
        prob_word_not_spam_list = getProbList(df_training, target_training, probWordGivenNotSpam, splitter_training)
        prob_not_word_not_spam_list = getProbList(df_training, target_training, probNotWordGivenNotSpam, splitter_training)
        prob_spam = probCol(target_training, 0)
        prob_not_spam = 1 - probCol(target_training, 0)
           
        target_predict = getPredictTarget(prob_word_spam_list, prob_not_word_spam_list, prob_spam, prob_word_not_spam_list, prob_not_word_not_spam_list, prob_not_spam, df_test, splitter_test)    
        accuracy = getAccuracy(target_test, target_predict, splitter_test)
        accuracy_sum += accuracy
    
    accuracy_average = accuracy_sum / num_fold
    print(accuracy_average)
    return accuracy_average
    
       
    """ Some tests
        
    splitter = list(range(len(df)))
    splitter.reverse()
    prob_word_spam_list = getProbWordGivenSpamList(df, target, splitter)    
    prob_not_word_spam_list = getProbList(df, target, probNotWordGivenSpam, splitter)
    prob_word_not_spam_list = getProbList(df, target, probWordGivenNotSpam, splitter)
    prob_not_word_not_spam_list = getProbList(df, target, probNotWordGivenNotSpam, splitter)
    prob_spam = probCol(target, 0)
    prob_not_spam = 1 - probCol(target, 0)
       
    target_predict = getPredictTarget(prob_word_spam_list, prob_not_word_spam_list, prob_spam, prob_word_not_spam_list, prob_not_word_not_spam_list, prob_not_spam, df, splitter)    
    accuracy = getAccuracy(target, target_predict, splitter)
    print(accuracy)
    """
      
    
    """
    data = pd.DataFrame([[1,0],[0,0],[1,0],[0,1],[0,0],[1,1],[1,1],[1,0]])
    target = pd.DataFrame([[1],[1],[0],[0],[1],[0],[0],[0]])
    
    splitter = getDataSplitter(data, 4)
    print(splitter)
    
    splitter_test = getTestSplitter(splitter, 0)
    print(splitter_test)
    
    splitter_training = getTrainingSplitter(splitter, 0)
    print(splitter_training)
    
    data_test = getTest(data, splitter_test)
    print(data)
    print(data_test)    
       
    data_training = getTraining(data, splitter_training)
    print(data)
    print(data_training)    
    """
    
    
    """
    splitter = [[3, 2], [5, 4], [0, 1],[6,7]]
    splitter_test = [5, 4]
    splitter_training = [3, 2, 0, 1, 6, 7]
    
    data_test = getTest(data,splitter_test)
    target_test = getTest(target,splitter_test)
    data_training = getTraining(data,splitter_training)
    target_training = getTraining(target,splitter_training)
        
    list1 = getProbWordGivenSpamList(data_training, target_training, splitter_training)
    list2 = getProbWordGivenNotSpamList(data_training, target_training, splitter_training)
    list3 = getProbNotWordGivenSpamList(data_training, target_training, splitter_training)
    list4 = getProbNotWordGivenNotSpamList(data_training, target_training, splitter_training)
    
    prob_spam = probCol(target_test,0)
    prob_not_spam = 1 - probCol(target_test,0)
            
    print(data_test)
    print(list1, list2, list3, list4, prob_spam, prob_not_spam)
    
    prob = probSpamGivenDataNumerator(list1, list3, prob_spam, data_test, splitter_test[1])
    print(prob)
    print(data_training)
    print(target_training)

    prob_not = probNotSpamGivenDataNumerator(list2, list4, prob_not_spam, data_test, splitter_test[1])
    print(prob_not)

    list_target = getPredictTarget(list1, list3, prob_spam, list2, list4, prob_not_spam, data_test, splitter_test)
    print(list_target)
    
    spam_predict = spamPredict(list1, list3, prob_spam, list2, list4, prob_not_spam, data_test, 0)
    print(spam_predict)
    
    target_predict = getPredictTarget(list1, list3, prob_spam, list2, list4, prob_not_spam, data_test, splitter_test)  
    accuracy = getAccuracy(target_test, target_predict, splitter_test)
    
    print(accuracy)
    """
 
    
    """
    splitter_training = range(len(df))
    list1 = getProbWordGivenSpamList(df, target, splitter_training)
    list2 = getProbWordGivenNotSpamList(df, target, splitter_training)
    list3 = getProbNotWordGivenSpamList(df, target, splitter_training)
    list4 = getProbNotWordGivenNotSpamList(df, target, splitter_training)
    
    prob_spam = probCol(target,0)
    prob_not_spam = 1 - probCol(target,0)
    
    list_target = getPredictTarget(list1, list3, prob_spam, list2, list4, prob_not_spam, df, splitter_test)
    print(list_target)
    """
 
    
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

    
if __name__ == "__main__":
    main()

