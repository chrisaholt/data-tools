import multiprocessing
import time
import torch

def foo(ind):
    return torch.randn(1).item()

if __name__ == "__main__":
    num = 1000

    print("***Non-parallel")
    start = time.time()
    counter = 0
    for ind in range(num):
        counter = counter + foo(ind)
    end = time.time()
    print("***Counter: ", counter)
    print("***Non-parallel time: ", end - start)

    print("***Parallel")
    start = time.time()
    pool = multiprocessing.Pool(2)
    counter = sum(pool.map(foo, range(num)))
    end = time.time()
    print("***Counter: ", counter)
    print("***Parallel time: ", end - start)
