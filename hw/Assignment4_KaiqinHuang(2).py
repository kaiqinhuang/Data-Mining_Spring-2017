#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Mining Assignment 4 (Version 2)

Name: Kaiqin Huang
Due Date: Feb 28, 2017
"""

import pandas as pd
import math


def entropy(distinct_list):
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


def informationGain(attribute_list):
    list_sum = sum(attribute_list)
    list_len = len(attribute_list)
    
    if list_len % 2 == 0:
        positive = 0
        negative = 0
        sum_entropy = 0
        
        for i in range(0, list_len, 2):
            sublist = attribute_list[i: i + 2]
            positive += sublist[0]
            negative += sublist[1]
            sublist_sum = sum(sublist)
            prop_sublist = sublist_sum / list_sum
            sum_entropy += prop_sublist * entropy(sublist)
            
        list_start = [positive, negative]
        entropy_start = entropy(list_start)
        ig = entropy_start - sum_entropy
        
        return ig
    
    else:
        print("The length of the attribute list must be an even number.")
        
        
def readData(filename):
    df = pd.read_csv(filename, header = None)
    return df


def getAttributeOptions(column):
    options = list(set(column))
    return options
 
    
def getAttributeList(column_attr, column_play):
    attr_opt = getAttributeOptions(column_attr)  
    attr_list = [0] * len(attr_opt)
    for i in range(len(column_attr)):
        for n in range(len(attr_opt)):
            if column_attr[i] == attr_opt[n] and column_play[i] == "Yes":
                attr_list[2 * n] += 1
            if column_attr[i] == attr_opt[n] and column_play[i] == "No":
                attr_list[2 * n + 1] += 1                        
    return attr_list


def decisionTree():
    df = readData("tennis.csv")
    df.columns = ["Outlook","Temp","Humidity","Wind","Play"]

    col_outlook = df["Outlook"]
    col_temp = df["Temp"]
    col_humidity = df["Humidity"]
    col_wind = df["Wind"]
    col_play = df["Play"]

    outlook = getAttributeList(col_outlook, col_play)
    temp = getAttributeList(col_temp, col_play)
    humidity = getAttributeList(col_humidity, col_play)
    wind = getAttributeList(col_wind, col_play)
    
    









#==============================================================================
#     sunny_posi = 0
#     sunny_nega = 0
#     overcast_posi = 0
#     overcast_nega = 0
#     rainy_posi = 0
#     rainy_nega = 0
#     for i in range(len(df["Outlook"])):
#         if df["Outlook"][i] == "Sunny" and df["Play"][i] == "Yes":
#             sunny_posi += 1
#         if df["Outlook"][i] == "Sunny" and df["Play"][i] == "No":
#             sunny_nega += 1
#         if df["Outlook"][i] == "Overcast" and df["Play"][i] == "Yes":
#             overcast_posi += 1
#         if df["Outlook"][i] == "Overcast" and df["Play"][i] == "No":
#             overcast_nega += 1
#         if df["Outlook"][i] == "Rainy" and df["Play"][i] == "Yes":
#             rainy_posi += 1
#         if df["Outlook"][i] == "Rainy" and df["Play"][i] == "No":
#             rainy_nega += 1
#     outlook = [sunny_posi, sunny_nega, overcast_posi, overcast_nega, rainy_posi, rainy_nega]    
#==============================================================================
    
