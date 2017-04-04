#!/usr/bin/env python


"""
Name: Kaiqin Huang

Train a FF ANN using backprop to learn the XOR function. Use a 3-4-1 architecture.

Submit your code here, as well as a txt file that lists the weights learned by your model, by 2:30pm on 4-4.
"""


import numpy as np
import math
import random


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


def weightUpdate(output_a, error_b, weight):
    weight_updated = weight + error_b * output_a
    return weight_updated


def errorHidden(output_a, error_b, weight):
    error_a = output_a * (1 - output_a) * (error_b * weight)
    return error_a


def main():
    input = np.reshape([0,0,1, 0,1,1, 1,0,1, 1,1,1], (4,3))
    target = [0,1,1,0]
    weight12 = np.reshape(genWeight(12), (4,3))
    weight4 = np.array(genWeight(4))

    num_training = 12000
    for _ in range(num_training):

        hidden = []
        for i in range(4):
            for j in range(4):
                node = dotProduct(input[i], weight12[j])
                node_sigmoid = sigmoidFunction(node)
                hidden.append(node_sigmoid)
        hidden = np.reshape(hidden, (4,4))

        output = []
        for i in range(4):
            node_output = dotProduct(hidden[i], weight4)
            output_sigmoid = sigmoidFunction(node_output)
            output.append(output_sigmoid)

        weight4_new = weight4
        weight12_new = weight12

        for i in range(4):  # Counter for four sets of input
            error_out = errorOut(target[i], output[i])
            weight4_new_temp = []
            weight12_new_temp = []
            for j in range(4):  # Counter for four hidder layer nodes and weights connecting output
                weight4_new_element = weightUpdate(hidden[i][j], error_out, weight4_new[j])
                weight4_new_temp.append(weight4_new_element)

                error_hidden = errorHidden(hidden[i][j], error_out, weight4_new[j])
                for r in range(3):  # Counter for three input variables
                    weight12_new_element = weightUpdate(input[i][r], error_hidden, weight12_new[j][r])
                    weight12_new_temp.append(weight12_new_element)

            weight4_new = weight4_new_temp
            weight12_new = np.reshape(weight12_new_temp, (4,3))


    print(weight4_new)
    print(target)
    print(output)
    print(weight12_new)


if __name__ == "__main__":
    main()

