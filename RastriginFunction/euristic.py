import math
import numpy as np
import time

n = 2

start_time = time.time()

def calc (arr):
    s = 0
    for i in arr:
        s = s + i * i - 10 * math.cos(2 * math.pi * i)
    return s

min2 = calc(np.random.uniform(-5.12, 5.12, n))
for i in range(0, 20000): 
    min2 = min(min2, calc(np.random.uniform(-5.12, 5.12, n)))

end_time = time.time()
elapsed_time = end_time - start_time

print(f"{str(10 * n + min2).replace('.', ',')} {str(elapsed_time).replace('.', ',')} seconds")
