#!/usr/bin/env python


"""
Name: Kaiqin Huang

Train a FF ANN using backprop to learn the XOR function. Use a 3-4-1 architecture.

Submit your code here, as well as a txt file that lists the weights learned by your model, by 2:30pm on 4-4.
"""


import numpy as np
import math
import random
import sys
import pdb


def genWeight(number):
    weight = []
    for _ in range(number):
        weight.append(random.random())
    return weight


def dotProduct(value, weight):
    value = np.array(value)
    weight = np.array(weight)
    product = value * weight
    dot_product = sum(product)
    return dot_product


def sigmoidFunction(x):
    sigmoid = 1 / (1 + math.e ** (-x))
    return sigmoid


def errorOut(target_b, output_b):
    error_b = output_b * (1 - output_b) * (target_b - output_b)
    return error_b


def weightAdjust(output_a, error_b, weight):
    weight_updated = weight + error_b * output_a
    return weight_updated


def errorHidden(output_a, error_b, weight):
    error_a = output_a * (1 - output_a) * (error_b * weight)
    return error_a















def main():
    input = np.matrix([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
    target = [0,1,1,0]
    weight12 = np.reshape(genWeight(12), (4,3))
    weight4 = np.array(genWeight(4))

    hidden = []
    for i in range(4):
        node = dotProduct(input[i], weight12[i])
        node_sigmoid = sigmoidFunction(node)
        hidden.append(node_sigmoid)

    hidden_np = np.array(hidden)
    output = dotProduct(hidden_np, weight4)
    output_sigmoid = sigmoidFunction(output)

    print(output_sigmoid)


if __name__ == "__main__":
    main()

