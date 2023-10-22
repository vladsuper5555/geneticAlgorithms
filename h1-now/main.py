# import pandas as pd
import sys
from function_definitions import functionDefinitions
from hill_climbing import compute_hill_climbing_data
# from simulated_annealing import compute_simulated_annealing
dimensions = [30] #100 for later

method = sys.argv[-1]

if __name__ == '__main__':
    # both this are matrixes of data
    # hill_climbing_data_method_best = compute_hill_climbing_data(functionDefinitions, dimensions, 'best')
    # hill_climbing_data_method_first = compute_hill_climbing_data(functionDefinitions, dimensions, 'first')
    # hill_climbing_data_method_worst = compute_hill_climbing_data(functionDefinitions, dimensions, 'worst')
    hill_climbing_data = compute_hill_climbing_data(functionDefinitions, dimensions, method)
    # simulated_annealing_data = compute_simulated_annealing(functionDefinitions, dimensions)
    fileName = 'hill_climbing_data_method_' + method + '.txt'
    with open(fileName, 'w') as f:
        for row in hill_climbing_data:
            f.write(str(row[0]) + ',' + str(row[1]) + "\n")

    # with open('hill_climbing_data_method_first.txt', 'w') as f:
    #     for row in hill_climbing_data_method_first:
    #         f.write(str(row[0]) + ', ' + str(row[1]) + "\n")

    # with open('hill_climbing_data_method_worst.txt', 'w') as f:
    #     for row in hill_climbing_data_method_worst:
    #         f.write(str(row[0]) + ', ' + str(row[1]) + "\n")
