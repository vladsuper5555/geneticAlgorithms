import math
from random import getrandbits
import time

MAX_DATA_GATHERINGS = 30 # the number of times we gather data
T_MAX_HILL = 3000 # the maximum number of iterations we will do for hill climbing
EPSILON = 0.00001

# @jit(target_backend='cuda')   
def decodeBitStringValue(bitString, rangeInterval, dimension):
    bitStringLength = int(len(bitString) / dimension)
    bitStrings = [bitString[i:i+bitStringLength] for i in range(0, len(bitString), bitStringLength)]

    rangeValue = rangeInterval[1] - rangeInterval[0]
    denominator = 2 ** bitStringLength - 1

    values = []

    for bitSubString in bitStrings:
        value = 0
        for char in bitSubString:
            value = (value << 1) + char
        normalized_value = value / denominator
        value_in_range = normalized_value * rangeValue + rangeInterval[0]
        values.append(value_in_range)
    # now the values are in the range [range[0], range[1]]
    return values

# @jit(target_backend='cuda')   
def flip_bit(bitString, position):
    new_bitString = bitString.copy()
    new_bitString[position] = 1 if new_bitString[position] == 0 else 0
    return new_bitString


# @jit(target_backend='cuda')   
def calculateNeighbours (bitString):
    return [flip_bit(bitString, i) for i in range(len(bitString))]

def chooseNextNeighbour (function, neighBours, method, initialFunctionValue, rangeInterval, dimension):
    if method == 'best':
        lowestValue = math.inf
        neighBourForLowestValue = []
        for neighbour in neighBours: 
            functionValue = function['function'](decodeBitStringValue(neighbour, rangeInterval, dimension))
            if functionValue < lowestValue:
                lowestValue = functionValue
                neighBourForLowestValue = neighbour
        return neighBourForLowestValue
    elif method == 'first':
        # here we need some kind of random shuffel
        for neighbour in neighBours: 
            functionValue = function['function'](decodeBitStringValue(neighbour, rangeInterval, dimension))
            if functionValue < initialFunctionValue:
                return neighbour
        return neighBours[0]
    elif method == 'worst':
        lowestValue = -math.inf
        for neighbour in neighBours: 
            functionValue = function['function'](decodeBitStringValue(neighbour, rangeInterval, dimension))
            if functionValue > lowestValue and functionValue < initialFunctionValue:
                lowestValue = functionValue
                neighBourForLowestValue = neighbour
        if lowestValue == -math.inf:
            return neighBours[0]
        else:
            return neighBourForLowestValue
    
def hill_climb_algorithm (function, dimension, method):
    #start a timer and also return the time it took to complete this action
    best_function_resposne = math.inf
    bitStringLength = dimension * math.ceil(math.log2((function['range'][1] - function['range'][0]) * EPSILON ** (-1)))
    rangeInterval = function['range']
    start_time = time.time()
    for i in range(T_MAX_HILL): # this is the number of times we select at random and try to improve
        # this is basically the first value of the bitstring and we will improve on this one
        bitString = [not getrandbits(1) for _ in range(bitStringLength)]

        # calculated the pure value for the bitstring 
        local = False
        current_value = function['function'](decodeBitStringValue(bitString, function['range'], dimension))
        while not local:
            nextNeighbour = chooseNextNeighbour(function, calculateNeighbours(bitString), method, current_value, rangeInterval, dimension)
            nextNeighbourValue = function['function'](decodeBitStringValue(nextNeighbour, rangeInterval, dimension))
            if nextNeighbourValue < current_value:
                bitString = nextNeighbour
                current_value = nextNeighbourValue
            else:
                local = True
        if current_value < best_function_resposne:
            best_function_resposne = current_value

    end_time = time.time()
    execution_time = end_time - start_time
    return [round(best_function_resposne, 5), round(execution_time, 5)]

def calculate_data_for_function_and_dimension (function, dimension, method): 
    data = []
    for i in range (MAX_DATA_GATHERINGS):
        # we now try to obtain a minimum using hill climbing
        # in the end we append to data a vector of value and time
        [min_value, time] = hill_climb_algorithm(function, dimension, method)
        data.append([min_value, time])
        print(f"Test {i + 1}/{MAX_DATA_GATHERINGS} completed for function {function['functionName']} and dimension {dimension} data got is: {[min_value, time]}")
    return data

def compute_hill_climbing_data (functions, dimensions, method):
    # here we expect to get back a matrix basically an array of arrays
    data = []
    for function in functions:
        for dimension in dimensions:
            data.append([function['functionName'], dimension])
            rows = calculate_data_for_function_and_dimension(function, dimension, method)
            for row in rows: 
                data.append(row)
    return data