import math
import time
import random


for j in range(10):
    start_time = time.time()
    n = 10
    m = 10

    def calc (value, index):
        return math.sin(value) * math.sin(index * value ** 2 / math.pi) ** (2*m)
    
    min1 = [calc(random.uniform(0, math.pi), i + 1) for i in range(n)]
    mins = 0 
    for i in range(n):
        mins = mins + min1[i]
    for i in range(0, 200000): 
        random_numbers = [calc(random.uniform(0, math.pi), i + 1) for i in range(n)]
        
        s = 0
        for i in range(n):
            s = s + random_numbers[i]
        
        if s > mins:
            s = mins
            min1 = random_numbers


    end_time = time.time()
    elapsed_time = end_time - start_time

    print(str(-mins).replace('.', ','))
    print(str(elapsed_time).replace('.', ','))




    