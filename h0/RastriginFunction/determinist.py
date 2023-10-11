import math
import time


start_time = time.time()
n = 2
step = 0.000001
iterator = -5.12
# optimizing the calculation by just getting one minimum
min1 = iterator**2 - 10 * math.cos(2 * math.pi * iterator)
while iterator <= 5.12:
    min1 = min(iterator**2 - 10 * math.cos(2 * math.pi * iterator), min1)
    iterator = iterator + step

end_time = time.time()
elapsed_time = end_time - start_time

print((10 + min1) * n) # we always would expect to get the correct value    
print(f"Elapsed time: {elapsed_time} seconds")

