import math
import numpy as np
import time

for j in range(10):
    start_time = time.time()
    n = 10
    m = 10

    def calc (value, index):
        return math.sin(value) * math.sin(index * value ** 2 / math.pi) ** (2*m)

    #because the values are independent one from another we can just search for 1

    iterator = 0
    min1 = []
    for i in range(0, n):
        min1.append(calc(0, i + 1))

    while iterator < math.pi:
        for i in range (0, n):
            value = calc(iterator, i + 1)
            if value > min1[i]:
                min1[i] = value
        iterator += 0.000001
    s = 0
    for i in range(0, n):
        s = s + min1[i]
    s = -s
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(str(s).replace('.', ','))
    print(str(elapsed_time).replace('.', ','))
