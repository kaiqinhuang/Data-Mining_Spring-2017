#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Mining Assignment 4

Name: Kaiqin Huang
Date: Mar 2, 2017
"""


import math


def main():
    with open("/Users/kaiqinhuang/Desktop/tennis.csv","r") as file:
        tennis_data = file.readlines()
    
    tennis_data = [list(map(str.strip, line.split(','))) for line in tennis_data]

    buildTree(tennis_data, 0)


def entropy(distinct_list):
    """Calculate entropy
    
    Args:
        distinct_list: a list of two numbers, number of "Yes" and number of "No"
        
    Returns:
        ent: entropy
    """
    
    positive = distinct_list[0]
    negative = distinct_list[1]
   
    if positive == 0 and negative == 0:
        ent = 1    
    elif positive == 0 or negative == 0:
        ent = 0
    else:
        total = positive + negative
        prop_posi = positive / total
        prop_nega = negative / total
        ent = - prop_posi * math.log(prop_posi, 2) - prop_nega * math.log(prop_nega, 2)
    
    return ent


def entropy2(data):
    """Calculate entropy
    
    Args:
        data: a list of lists
        
    Returns:
        entropy_of_col: entropy of the specific column
    """
 
    col = getColumn(data, 4)
    entropy_of_col = entropy([col.count("Yes"), col.count("No")])   
    
    return entropy_of_col


def getColumn(data, col_id):
    """Get a column 
    
    Args:
        data: a list of lists
        col_id: the index of the column I want to get
        
    Returns:
        a specific column
    """
    
    return [line[col_id] for line in data]


def splitBy(data, attribute_id):
    """Split data by options of an attribute
    
    Args:
        data: a list of lists
        attribute_id: the index of the attribute that I want to use for spliting 
        
    Returns:
        split_data: a list of subsets of the original data
    """
    
    col = getColumn(data, attribute_id)
    values = set(col)
    split_data = []    
    for i in values:
        subset = [row for row in data if row[attribute_id] == i]
        split_data.append(subset)
   
    return split_data  
   

def informationGain2(data, attribute):
    """Calculate the information gain
    
    Args:
        data: a list of lists
        attribute: the attribute that I would like to calculate the IG
        
    Returns:
        columnIG: the IG of the specific attribute
    """
    
    split_data = splitBy(data, attribute)  
    weighted_entropies = 0
    
    for set in split_data:
        weighted_entropies += len(set) / len(data) * entropy2(set)     
    
    columnIG = entropy2(data) - weighted_entropies
    
    return columnIG


def maxIG(data):
    """Get the attribute with the highest IG
    
    Args:
        data: a list of lists
        
    Returns:
        index: the index of the attribute
        max_gain: the largest IG value
    """
    
    index = -1
    max_gain = -1
    
    for i in range(len(data[0]) - 1):
        gain = informationGain2(data, i)
        if gain > max_gain:
            index = i
            max_gain = gain
            
    return (index, max_gain)


def buildTree(data, level):
    """Build the decision tree
    
    Args:
        data: a list of list
        level: level of the current node
        
    Returns:
        none (just print-outs)
    """
    
    node = maxIG(data)
    subsets = splitBy(data, node[0])
    header = ["Outlook", "Temp", "Humidity", "Wind", "Play"]
        
    if node[1] == 0:
        print("\t" * level, level, getColumn(data, node[0])[0], ":", getColumn(data, -1)[0])              
    elif level < 4:
        print("\t" * level, level, getColumn(data, level - 1)[0], "->", header[node[0]])        
        rec = [buildTree(subset, level + 1) for subset in subsets]
    else:
        print("\t" * level, level, getColumn(data, level - 1)[0], ":", getColumn(data, -1))              
        

if __name__ == '__main__':
    main() 










#==============================================================================
# Tests:
#    
# print(data[0])
# print(data[4])
# 
# test = getColumn(tennisData, 0)
# 
# lastColumn = getColumn(tennisData, 4)
# print(lastColumn)
# print(lastColumn.count("Yes"))
# 
# a = entropy2(tennisData)
# print(a)
# 
# print(splitBy(tennisData, 0))
# 
# print(maxIG(tennisData))
# 
# firstNode = maxIG(tennisData)[0]
# 
# firstBranch = splitBy(tennisData, firstNode)
# 
# print(maxIG(firstBranch[0]))
#==============================================================================

