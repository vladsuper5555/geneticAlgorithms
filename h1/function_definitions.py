import math
functionDefinitions = []

def Michalewicz(inputs):
    sum = 0
    m = 10
    for index, value in enumerate(inputs):
        sum = sum + math.sin(value) * (math.sin((index * value ** 2) / math.pi) ** (2 * m))
    return -sum

def Schewefel(inputs):
    sum = 0
    for value in inputs:
        sum = sum + (-value * math.sin(math.sqrt(abs(value))))
    return sum

def Rastrigin(inputs):
    sum = 10 * len(inputs)
    for value in inputs:
        sum = sum + (value * value - 10 * math.cos(2 * math.pi * value))

    return sum

def De_Jong(inputs): 
    sum = 0
    for value in inputs:
        sum = sum + value * value
    return sum


functionDefinitions = [
    {
        'function': Schewefel,
        'range': [-500.0, 500.0],
        'functionName': 'Schewefel'
    },
    {
        'function': Michalewicz,
        'range': [0.0, math.pi],
        'functionName': 'Michalewicz'
    },
    {
        'function': Rastrigin,
        'range': [-5.12, 5.12],
        'functionName': 'Rastrigin'
    },
    {
        'function': De_Jong,
        'range': [-5.12, 5.12],
        'functionName': 'De_Jong'
    }
]