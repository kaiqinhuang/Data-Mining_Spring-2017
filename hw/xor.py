#!/usr/bin/env python

"""
Data Mining_Neural Network
Name: Kaiqin Huang

Train a FF ANN using backprop to learn the XOR function. Use a 3-4-1 architecture.

Submit your code here, as well as a txt file that lists the weights learned by your model, by 2:30pm on 4-4.
"""


import numpy as np
import math
import random


def genWeight(number):
    """ Generate a list of random weights ranging from 0 to 1:
        
    Args:
        number: the number of weights
        
    Returns:
        weight: a list of weights 
    """
        
    weight = []
    for _ in range(number):
        weight.append(random.random())
    return weight


def dotProduct(value, weight):
    """ Get the summation of products:
        
    Args:
        value: a list of values
        weight: a list of weights
        
    Returns:
        dot_product: the summation of products
    """
       
    value = np.array(value)
    weight = np.array(weight)
    product = value * weight
    dot_product = sum(product)
    return dot_product


def sigmoidFunction(x):
    """ Apply the Sigmoid function:
        
    Args:
        x: the value
        
    Returns:
        sigmoid: the value after applying the sigmoid function
    """     
            
    sigmoid = 1 / (1 + math.e ** (-x))
    return sigmoid


def errorOut(target_b, output_b):
    """ Calculate the error of the output node:
        
    Args:
        target_b: the true value
        output_b: the returned value
        
    Returns:
        error_b: the error of the output node
    """
    
    error_b = output_b * (1 - output_b) * (target_b - output_b)
    return error_b


def weightUpdate(output_a, error_b, weight):
    """ Update the weight:
        
    Args:
        output_a: the output of the start node
        error_b: the error of the end node
        weight: the weight connecting two nodes
        
    Returns:
        weight_updated: the updated weight
    """     
            
    weight_updated = weight + error_b * output_a
    return weight_updated


def errorHidden(output_a, error_b, weight):
    """ Calculate the error of the hidden layer nodes:
        
    Args:
        output_a: the output of the hidden layer node
        error_b: the error of the output node
        weight: the weight connecting two nodes
        
    Returns:
        error_a: the error of the hidden layer node
    """
            
    error_a = output_a * (1 - output_a) * (error_b * weight)
    return error_a


def hiddenLayerOutput(inputs, input_hidden_weights):
    """ Get the hidden layer node output:
        
    Args:
        inputs: the x1, x2, x0
        input_hidden_weights: a list of weights connecting inputs and hidden layer nodes
    
    Returns:
        hidden: a list of hidden layer nodes outputs
    """
    
    hidden = []
    for i in range(len(inputs)):
        input_hidden = []
        for j in range(len(input_hidden_weights)):
            node = dotProduct(inputs[i], input_hidden_weights[j])
            node_sigmoid = sigmoidFunction(node)
            input_hidden.append(node_sigmoid)
        hidden.append(input_hidden)
    return hidden


def outputLayerOutput(hidden_layer_outputs, hidden_output_weights):
    """ Get the output layer node output:
        
    Args:
        hidden_layer_outputs: a list of hidden layer nodes outputs 
        hidden_output_weights: a list of weights connecting hidden layer nodes and output nodes
        
    Returns:
        output: a list of output values
    """
    
    output = []
    for i in range(len(hidden_layer_outputs)):
        node_output = dotProduct(hidden_layer_outputs[i], hidden_output_weights)
        output_sigmoid = sigmoidFunction(node_output)
        output.append(output_sigmoid)
    return output


def main():
    inputs = np.reshape([0,0,1, 0,1,1, 1,0,1, 1,1,1], (4,3))
    target = [0,1,1,0]
    
# Test:
#    inputs = [[0.35,0.9]]
#    target = [0.5]
#    input_hidden_weights = [[0.1,0.8],[0.4,0.6]]
#    hidden_output_weights = [0.3,0.9]

    n_hidden_nodes = 4
    n_inputs = len(inputs)
    n_variables = len(inputs[0])
    input_hidden_weights = np.reshape(genWeight(n_hidden_nodes*n_variables), (n_hidden_nodes,n_variables))
    hidden_output_weights = np.array(genWeight(n_hidden_nodes))

    hidden_output_weights_new = hidden_output_weights
    input_hidden_weights_new = input_hidden_weights

    n_training = 20000
    for _ in range(n_training):
        
        hidden = hiddenLayerOutput(inputs, input_hidden_weights_new)
        output = outputLayerOutput(hidden, hidden_output_weights_new)
       
        for i in range(n_inputs):  # Counter for four sets of inputs
            error_out = errorOut(target[i], output[i])
            hidden_output_weights_new_temp = []
            input_hidden_weights_new_temp = []
            for j in range(n_hidden_nodes):  # Counter for four hidder layer nodes and weights connecting output
                hidden_output_weights_new_element = weightUpdate(hidden[i][j], error_out, hidden_output_weights_new[j])
                hidden_output_weights_new_temp.append(hidden_output_weights_new_element)            
     
                error_hidden = errorHidden(hidden[i][j], error_out, hidden_output_weights_new[j])
                for r in range(n_variables):  # Counter for three inputs variables
                    input_hidden_weights_new_element = weightUpdate(inputs[i][r], error_hidden, input_hidden_weights_new[j][r])
                    input_hidden_weights_new_temp.append(input_hidden_weights_new_element)

            hidden_output_weights_new = hidden_output_weights_new_temp
            input_hidden_weights_new = np.reshape(input_hidden_weights_new_temp, (n_hidden_nodes,n_variables))

    print("Number of iterations is %d" % n_training)
    print("The target is: ", target)
    print("The output is: ", output)
    print("The 4 weights connecting hidden layer nodes and output nodes are: ")
    print(hidden_output_weights_new)
    print("The 12 weights connecting inputs and hidden layer nodes are: ")
    print(input_hidden_weights_new)
           

if __name__ == "__main__":
    main()


"""
What I have learned from this assignment:
    
    1. The best way to find error is to do test, using smaller dataset
    2. Should avoid having too many (maybe more than 2) "for" loops iterating in each other
    3. For the "for" loop, be careful about where to put the initialization
    4. Should try to break long code into smaller functions so I can do test on each small part
    5. Should avoid using magic numeric numbers, like 4 weights or 12 weights, but should give it as a common / general variable,
        and then it is much easier to change its value.

"""

